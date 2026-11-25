# inv_emulator

**Neural emulators for fast synthesis and spatially regularized inversion of
Stokes profiles, with examples for Si I 10827 Å, He I 10830 Å and Fe I 6301.5/6302.5 Å.**

Classical inversion codes (SIR, Hazel, ...) solve the radiative transfer
equation for every pixel at every iteration. That is accurate but slow, and it
limits how much of a field of view can be inverted with spatial coupling
between pixels. This repository replaces the forward problem with
*emulators*. These are small neural networks trained on large databases of
Hazel2 syntheses. Each one maps a model atmosphere to its emergent Stokes
profiles in one forward pass, and the pass is differentiable. With the
emulators in place, a full map (hundreds of thousands of pixels) is inverted
on a GPU by gradient descent, with every pixel solved at once, optional
spatial regularization, and an optional telescope PSF.

The forward model uses two compact representations:

| What | Representation | Why |
|------|----------------|-----|
| **Depth stratifications** `T, vz, vmic, Bx, By, Bz` (80 points in log τ) | **cubic B-splines**: 10 coefficients per variable inside the emulator, plus a few interpolating *nodes* per variable in the inversion | smooth, local, low-dimensional; a node parameterization gives control over how many degrees of freedom the inversion has |
| **Stokes profiles** `I, Q, U, V` | **PCA**: a per-parameter principal-component basis learned from the training profiles | 20–50 coefficients per Stokes parameter reproduce the profiles to ~10⁻⁴, so the network predicts a short vector rather than hundreds of wavelengths |

```
nodes ──interp. B-spline──▶ stratification [6, 80] ──B-spline projection──▶ 60 coefficients
                                                                          (+ depth derivatives, μ)
                                                                                   │
                                                                         residual MLP (6 × 512)
                                                                                   │
Stokes [4, L] ◀── inverse symlog(·)·I ◀── mean + PCA eigenvectors · coeffs ◀──────┘
```

📖 **Full documentation:** Sphinx sources in [`docs/`](docs/), published on
GitHub Pages by `.github/workflows/docs.yml`. To build locally:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html     # or: cd docs && make html
```

---

## Emulators

| Emulator | Line(s) | Inputs | Output basis | Checkpoint |
|----------|---------|--------|--------------|------------|
| **Si I 10827** | Si I 10827.089 Å (photosphere) | stratification on 80 log τ points + μ | PCA, 18/49/49/44 components (I/Q/U/V) on 350 λ in [10824, 10833] Å | `weights/emulator_si10827_pca_resmlp512_cosine_deriv.pt` |
| **Fe I 6301** | Fe I 6301.5012 Å | same | PCA, 22/39/40/36 on 70 λ | `weights/emulator_fe6301_pca_resmlp512_cosine_deriv.pt` |
| **Fe I 6302** | Fe I 6302.4936 Å | same | PCA, 20/34/34/32 on 80 λ | `weights/emulator_fe6302_pca_resmlp512_cosine_deriv.pt` |
| **He I 10830** | He I 10830 triplet (chromospheric slab) | `Bx, By, Bz, Δv, a, height, θ` | PCA, 30 components for each of ε (IQUV), η (IQUV), ρ (QUV) | `weights/emulator_he10830_pca.pt` |

The two Fe I emulators split the Hazel grid `linspace(6300.8921, 6303.2671,
150)` at the continuum point between the lines, and `SpectralSynthesizer`
joins them back together. The He I emulator does not output Stokes profiles.
It outputs the slab's emission, absorption and magneto-optical coefficients.
These enter an exact constant-property slab formal solution (as in Hazel2),
with the emulated Si I spectrum as the boundary condition. The repository also
contains a Ca II 8542 emulator (`invert_caii8542.py`), which uses the same
architecture.

The photospheric emulators share the following architecture:

* **Depth:** each of the 6 physical variables is projected by least squares
  onto 10 clamped cubic B-splines on log τ ∈ [−7.5, 1.5]. The derivative-spline
  coefficients (9 per variable) and μ are added to those inputs, so the
  network has 115 inputs.
* **Stokes:** the network fits `I`, `symlog(a·Q/I)`, `symlog(a·U/I)` and
  `symlog(a·V/I)`, with `a = 100` for Si I and `a = 30` for Fe I. Each one is
  expanded in its own PCA basis, which keeps 99.999 % of the training variance.
* **Network:** a residual MLP (width 512, 6 pre-norm blocks), trained with
  AdamW and a cosine schedule.
* Everything is stored inside the checkpoint: projection matrices,
  eigenvectors, mean profiles and scalers. `emulator(physical, mu)` takes
  physical stratifications and returns physical Stokes profiles.

> The emulators were trained with `vz(log τ = −1) = 0` and μ ∈ [0.2, 1]. The
> bulk line-of-sight velocity is applied afterwards as a Doppler shift of the
> synthetic spectrum (see below).

<!-- > `weights/` is excluded from git (`.gitignore`). Download the checkpoints -->
<!-- > separately and place them in `weights/`. -->

---

## Installation

```bash
git clone <this repository>
cd inv_emulator
pip install numpy scipy torch h5py matplotlib tqdm
# optional: astropy (FITS PSFs, CRISP data), PyWavelets (wavelet regularizer),
#           hazel (comparison with Hazel2), pyqtgraph + PyQt6 (view_inversion.py)
```

Run the scripts from the repository root, because they import the local
`modules` package. A CUDA GPU is used automatically when one is available.

Download the weights from [here](https://cloud.iac.es/index.php/s/sJJE7DSXMTwnntA)
and put them on the `weights/`.

---

## Synthesis

`synth_fe6301_si10827_he10830.py` synthesizes one atmosphere with the
emulators and compares the result with Hazel2. The essential part is:

```python
import numpy as np, torch
from modules.inversion import LineEmulator, SpectralSynthesizer
from modules.hei import SiHeSynthesizer
from modules.hei_pca import HeIPCAEmulator

# model: (80, 8) = logtau, T, Pe, vmic, v, Bx, By, Bz  (deepest point first)
v_ref = model[22, 4]                               # velocity at log tau = -1
physical = model[:, [1, 4, 3, 5, 6, 7]].T.copy()   # (6, 80): T, vz, vmic, Bx, By, Bz
physical[1] -= v_ref                               # emulators expect vz(-1) = 0

# Fe I 6301 + 6302 (photosphere only)
wave_fe = np.linspace(6300.8921, 6303.2671, 150)
fe = SpectralSynthesizer([
    LineEmulator("weights/emulator_fe6301_pca_resmlp512_cosine_deriv.pt", wave_fe[wave_fe - 6301.5012 < 0.5]),
    LineEmulator("weights/emulator_fe6302_pca_resmlp512_cosine_deriv.pt", wave_fe[wave_fe - 6302.4936 > -0.5])])

# Si I 10827 photosphere + He I 10830 slab
sihe = SiHeSynthesizer(
    SpectralSynthesizer([LineEmulator("weights/emulator_si10827_pca_resmlp512_cosine_deriv.pt",
                                      np.linspace(10824.0, 10833.0, 350))]),
    HeIPCAEmulator.load("weights/emulator_he10830_pca.pt"), step=0.02)
he = dict(He_Bx=200.0, He_By=-100.0, He_Bz=300.0, He_tau=1.0, He_v=2.0,
          He_deltav=8.0, He_a=0.3, He_beta=1.0, He_height=3.0)

wave = np.linspace(10824.3, 10832.7, 421)          # any wavelengths inside the grid
with torch.no_grad():
    stokes, valid = sihe(torch.tensor(physical[None], dtype=torch.float32),
                         torch.tensor([1.0]),                       # mu
                         torch.tensor([v_ref], dtype=torch.float32),
                         torch.tensor(sihe.relative_wavelengths(wave), dtype=torch.float32),
                         {k: torch.tensor([v], dtype=torch.float32) for k, v in he.items()})
# stokes: (1, 4, 421), in units of the quiet-Sun disk-centre continuum
```

All arguments are batched. Passing `B` atmospheres at once synthesizes `B`
spectra in one call.

---

## Inversion

Every `invert_*.py` script follows the same steps:

1. **Forward model.** A `SpectralSynthesizer` built from one or more
   `LineEmulator`s, or a `SiHeSynthesizer` that adds He I slabs. Optional
   pieces are an instrumental profile `psf`, telluric lines, and a spatial
   telescope PSF.
2. **Depth grid and bounds.** These come from the training models: the
   log τ grid, the median stratification (initial T and vmic), and the
   0.1–99.9 percentile range at every node height (box bounds, so the
   emulators never extrapolate). The lower bound of `Bx` is set to 0, which
   removes the 180° ambiguity.
3. **Variables.** One `VariableConfig(nodes, scale, bounds, regularization)`
   per physical variable. The *nodes* are log τ positions. A B-spline of
   degree `min(3, n−1)` that interpolates the node values gives the
   stratification: 1 node gives a constant, 2 nodes a linear profile, and 4 or
   more a cubic. Outside the outer nodes the stratification is held constant.
   Per-pixel quantities without depth dependence, such as He I slab and
   telluric parameters, use `slab_variable(...)`.
4. **Regularization.** Every node map can carry spatial regularizers:
   `Regularizer("l2" | "tv" | "l1" | "wavelet_l1", weight)`. The transverse
   field can instead be regularized through the ambiguity-invariant
   `TransverseRegularizer("Bx", "By", ...)`. The scripts use
   `smooth(jump) = Regularizer("l2", REG_STRENGTH / jump**2)`. With that
   choice, a pixel-to-pixel difference of size `jump` costs `REG_STRENGTH`
   units of χ².
5. **`InversionConfig`.** Holds the noise per Stokes parameter, Stokes
   weights, optimizer (Adam), learning rate with cosine decay, iteration
   count, and `chunk_size` (pixels per forward/backward pass, chosen to fit
   GPU memory).
6. **Initial guess.** The weak-field approximation gives `Bx, By, Bz`. The
   centre of gravity of `I` gives `vz`. The median stratification gives `T`
   and `vmic`. For He I there are dedicated estimates
   (`he_slabs_initial_guess`), and tellurics are initialized from the depth of
   the I minimum.
7. **`Inverter(...).run(...)`, then `save_result(...)`.** The result is written
   to HDF5.

The objective is

```
J = Σ_pixels Σ_s w_s Σ_λ ((syn − obs)/σ_s)²  +  Σ_variables weight · R(node maps)
```

and it is minimized with Adam. After each step comes a proximal step for the
spatial regularizers and a projection onto the bounds. The bulk velocity
`v_ref = vz(log τ = −1)` Doppler-shifts the synthetic spectrum, and the
emulators receive `vz − v_ref`.

### Script configurations

| Script | Data | Forward model | Nodes (log τ) | Notable settings |
|--------|------|---------------|---------------|------------------|
| `invert_fe6301_6302.py` | Hinode SP (AR 11967), normalized to quiet-Sun continuum, 400×400 px | Fe I 6301 + 6302 emulators | T: −7.5, −4, −2.5, −1, 0, 1.5 · vz, Bx, By, Bz: −7.5, 1.5 · vmic: −1 | `REG_STRENGTH = 5`; L2 on T (jump 20 K), vz, vmic, Bz (100 G); `TransverseRegularizer` on (Bx, By); optional Hinode spatial PSF (`PSF_FILE`) |
| `invert_si10827_he10830.py` | 10830 Å map (447×170 px) | Si I emulator + **two He I slabs in series** (`He1_`, `He2_`) + telluric line at 10832 Å | T: −7.5, −4, −2.5, −1, 1.5 · vz, Bx, By, Bz: −7.5, 1.5 · vmic: −1 | L2 on every variable; `He1_v ∈ [−20, 5]`, `He2_v ∈ [−5, 40]` km/s; `beta`, `height` fixed; Q, U weighted ×2 |
| `invert_si10827_he10830_l1.py` | same, cross-talk corrected | same | T adds a 0 node · Bx, By: single node at −1 | `TransverseRegularizer` on photosphere and both slabs; **L1 on `He2_tau`** (keeps the second component switched off unless the data need it); `He2_v ∈ [15, 40]` km/s; tighter Δv, a |
| `invert_multiregion.py` | synthetic Fe I + Si I/He I on the same pixels | one atmosphere shared by several `Region`s | T: −4.5, −3, −1.5, 0 · vz: −3, −1, 0.5 · B: −2.5, −0.5 | `MultiRegionInverter`; χ² per point weighted by region; variables shared by name |
| `invert_caii8542.py` | SST/CRISP Ca II 8542 | Ca II emulator + CRISP transmission profile | T: 7 nodes · vz: 3 · others: 2 | per-pixel μ from WCS; absolute intensity units |

### Running

```bash
python invert_fe6301_6302.py            # edit paths/configuration at the top of the script
python invert_si10827_he10830.py
python invert_multiregion.py fe630 sihe10830 --iterations 1000
python view_inversion.py inversions/inversion_result_si10827_he10830.h5 --telluric 10832.0
```

### Output (HDF5)

`nodes/<var>` (nx, ny, n_nodes) · `node_positions/<var>` ·
`stratification/<var>` (nx, ny, 80) · `logtau` · `v_ref` · `synthetic`
(nx, ny, 4, nl) · `chi2` · `wavelength` · `observations` · `history_chi2`.
He I parameters are stored as `nodes/He*_<par>` and telluric parameters as
`nodes/Tel_<par>`. The velocity `vz` is absolute, i.e. it includes `v_ref`.

---

## Repository layout

```
modules/
  bspline.py         B-spline knots, design matrix B and projector P = pinv(B)
  pca.py             per-channel PCA basis (mean + orthonormal eigenvectors)
  transforms.py      symlog(a·X/I) transform of Q, U, V and its inverse
  features.py        depth-derivative inputs computed from the B-spline coefficients
  model.py           MLP and ResidualMLP
  emulator.py        StokesEmulator (self-contained nn.Module), load_emulator
  inversion.py       nodes → stratification, SpectralSynthesizer, Inverter, MultiRegionInverter
  regularization.py  spatial regularizers and their proximal operators
  hei.py, hei_pca.py He I slab: PCA emulator of ε, η, ρ and exact formal solution
  telluric.py        Voigt telluric lines
  spatial_psf.py     FFT convolution of the synthetic maps with a telescope PSF
si10827.py, fe6301.py, fe6302.py, he10830.py   training scripts
invert_*.py                                    inversion scripts
synth_fe6301_si10827_he10830.py                synthesis example and Hazel2 comparison
view_inversion.py                              interactive viewer of inversion results
```
