# Emulator architecture

The photospheric emulators (Si I 10827, Fe I 6301, Fe I 6302, Ca II 8542)
share one design, implemented in `modules/emulator.py` as the class
`StokesEmulator`. The He I slab emulator follows the same idea without the
depth axis and is described on [the emulators page](emulators.md#he-i-10830).

## Pipeline

For a batch of $B$ atmospheres, the emulator maps

$$
\underbrace{\mathbf{x}\in\mathbb{R}^{B\times 6\times D}}_{T,\,v_z,\,v_\mathrm{mic},\,B_x,\,B_y,\,B_z \text{ on } D=80 \text{ depths}},\;
\mu\in\mathbb{R}^{B}
\;\longmapsto\;
\underbrace{\mathbf{S}\in\mathbb{R}^{B\times 4\times L}}_{I,\,Q,\,U,\,V \text{ on } L \text{ wavelengths}}
$$

in four stages:

```
encode ─────────────────────────────────────────────────────────────────────────────
  for every physical channel k:     c_k = P_k x_k                 (B-spline projection, 10 coeffs)
                                    ĉ_k = (c_k − offset_k)/scale_k
  depth features:                   d_k = Δ_k c_k  → standardized   (derivative-spline coeffs, 9 each)
  angle:                            μ̂  = (μ − offset)/scale
model ──────────────────────────────────────────────────────────────────────────────
  y = ResidualMLP([ĉ_T … ĉ_Bz, d_T … d_Bz, μ̂])                     (115 → 512 → … → n_PCA)
decode ─────────────────────────────────────────────────────────────────────────────
  for every Stokes channel s:       a_s = y_s · scale_s + offset_s
                                    z_s = mean_s + E_s a_s          (PCA reconstruction)
  inverse transform:                I = z_I,   X = I · symlog⁻¹(z_X)/a   for X = Q, U, V
```

All the matrices ($P_k$, $\Delta_k$, $E_s$, the mean profiles and
every offset/scale) are stored as frozen `nn.Parameter`s
(`requires_grad=False`) in the module's `state_dict`. Only the network
weights were trained. The emulator is therefore a single differentiable
`nn.Module`: gradients flow from the Stokes profiles back to the
stratification, which is what the inversion relies on.

```python
from modules.emulator import load_emulator
emulator, meta = load_emulator("weights/emulator_si10827_pca_resmlp512_cosine_deriv.pt")
stokes = emulator(physical, mu)     # physical (B, 6, 80), mu (B,)  ->  (B, 4, 350)
```

Internally `forward` is `decode(model(*encode(physical, mu)), physical)`. The
three stages can also be called separately.

## Stage 1: depth stratification to B-spline coefficients

Each physical channel is projected onto its own cubic B-spline basis on the
$\log\tau$ grid by least squares, $c_k = P_k x_k$ with
$P_k = \mathrm{pinv}(B_k)$. The trained emulators use 10 clamped cubic
B-splines with uniform knots on $[-7.5, 1.5]$ for every channel. The
details are on the [B-spline page](bsplines.md).

The coefficients are scaled with fixed constants, so that the
inputs are $\mathcal{O}(1)$:

| Channel | scale | offset |
|---------|-------|--------|
| `T` | 5000 K | 0 |
| `vz`, `vmic` | 2 km/s | 0 |
| `Bx`, `By`, `Bz` | 500 G | 0 |

### Depth-derivative features

The Stokes asymmetries come from depth *gradients*, for example the product of
velocity and field gradients that produces net circular polarization. To
expose them to the network directly, each channel also contributes the
coefficients of its **derivative spline** (`DepthFeatureConfig(derivatives=[...],
mode="diff")`):

$$
c'_i = \frac{p\,(c_{i+1}-c_i)}{t_{i+p+1}-t_{i+1}},\qquad i = 0,\dots,n-2,
$$

where $p=3$ is the degree and $t$ the knot vector. These are exact
B-spline coefficients of $\mathrm{d}x/\mathrm{d}\log\tau$, giving 9 extra
values per channel. They are standardized per element on the training set.

The network input is therefore $6\times10$ coefficients $+\;6\times9$
derivative coefficients $+\;\mu$ $=$ **115 values**.

## Stage 2: the network

`ResidualMLP` (`modules/model.py`) consists of:

* a linear embedding to width 512,
* 6 pre-norm residual blocks $x \leftarrow x + W_2\,\mathrm{SiLU}(W_1\,\mathrm{LN}(x))$
  with a hidden size of $2\times512$; $W_2$ is zero-initialized so
  every block starts as the identity,
* a LayerNorm and a linear head producing the concatenated PCA coefficients
  of I, Q, U and V.

Training (`si10827.py`, `fe6301.py`, `fe6302.py`) used AdamW (lr 10⁻³,
weight decay 10⁻⁵), batch size 1024, and 300 epochs with a linear warmup
followed by cosine decay. The loss is the MSE in scaled PCA-coefficient space
with equal Stokes weights.

## Stage 3: PCA coefficients to Stokes profiles

Each Stokes channel $s$ has its own orthonormal PCA basis $E_s$
($L\times n_s$) and mean profile $\bar z_s$, learned from the
training profiles, and

$$
z_s = \bar z_s + E_s\,a_s .
$$

The number of components $n_s$ is chosen per channel to keep 99.999 % of
the variance. Because the basis is orthonormal, a single scale per channel
(`per_coeff=False`) keeps the loss proportional to the profile-space MSE. See
the [PCA page](pca.md).

## Stage 4: inverse Stokes transform

The PCA is not applied to Q, U and V directly. It is applied to the
**symlog-compressed polarization ratios** (`StokesTransformConfig`,
`modules/transforms.py`):

$$
z_I = I,\qquad z_X = \operatorname{sign}(aX/I)\,\log\!\left(1 + \left|\frac{aX}{I}\right|\right),\quad X\in\{Q,U,V\}.
$$

Dividing by I removes the overall intensity scale. The symlog is linear and
sign-preserving for $|X/I|\ll 1/a$ (it is exactly zero without
polarization) and logarithmic for strong signals, which compresses the long
tail of strong-field profiles. The decoder inverts it exactly:

$$
X = \frac{I}{a}\,\operatorname{sign}(z_X)\left(e^{|z_X|}-1\right).
$$

The amplitude is $a=100$ for Si I 10827 and $a=30$ for Fe I and
Ca II.

## The checkpoint

`save_emulator` writes `{"state_dict": ..., "meta": ...}`. The `meta`
dictionary is enough to rebuild the module. Its keys for the Si I emulator
are:

```python
{'phys_names': ['T', 'vz', 'vmic', 'Bx', 'By', 'Bz'],
 'phys_grid_sizes':  {'T': 80, ...},          # depth points
 'phys_coeff_sizes': {'T': 10, ...},          # B-spline coefficients per channel
 'stokes_names': ['I', 'Q', 'U', 'V'],
 'stokes_grid_sizes':  {'I': 350, ...},       # wavelengths
 'stokes_coeff_sizes': {'I': 18, 'Q': 49, 'U': 49, 'V': 44},   # PCA components
 'in_dim': 115, 'out_dim': 160,
 'model_type': 'resmlp', 'model_kwargs': {'width': 512, 'num_blocks': 6, 'expansion': 2, ...},
 'stokes_basis': 'pca',
 'stokes_transform': {'amplitude': 100.0, 'intensity_channel': 'I', 'ratio_channels': None},
 'depth_features': {'derivatives': ('T', 'vz', 'vmic', 'Bx', 'By', 'Bz'), 'mode': 'diff', ...},
 'depth_feature_eval_points': [-7.5, -6.21, -4.93, -3.64, -2.36, -1.07, 0.21, 1.5],
 'weak_field': None}
```

`state_dict` holds `P_phys.<ch>` (10×80), `B_stokes.<s>` (the PCA
eigenvectors, $L\times n_s$), `stokes_mean.<s>`,
`phys_offset/scale.<ch>`, `stokes_offset/scale.<s>`, `mu_offset/scale`, the
depth-feature matrices, and the network weights.

```{note}
The checkpoint does **not** store the absolute wavelength grid. When an
emulator is used through `LineEmulator(checkpoint, wave)`, the absolute
wavelengths of its output must be supplied. `SpectralSynthesizer` checks that
their number matches the emulator output.
```

## Validity domain

* **Velocity.** The training models had $v_z(\log\tau=-1)=0$, so an
  emulator cannot shift a line as a whole. Bulk shifts are applied outside
  the emulator as a Doppler shift (see {ref}`Synthesis <velocities>`).
* **Angle.** The training range is $\mu\in[0.2,1]$. The inverter warns if
  the observations fall outside it.
* **Physical range.** The inversion bounds each node to the 0.1–99.9
  percentile range of the training models at that height, so the emulators
  are never evaluated far outside their training distribution.
* **Frame.** $B_z$ is along the line of sight. The Q/U reference
  direction follows SIR/Hazel photospheres (positive Q rotated by 90° with
  respect to the textbook convention).
