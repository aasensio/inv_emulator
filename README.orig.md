# mlp_bspline

Fast MLP emulator for Stokes-profile synthesis from atmospheric stratifications.

Given a physical model cube `[N, 6, D]` (channels `T, vz, vmic, Bx, By, Bz`
sampled at `D` depth points) and the corresponding Stokes cube `[N, 4, L]`
(`I, Q, U, V` sampled at `L` wavelengths), this trains an MLP that maps one
to the other, conditioned on heliocentric angle `mu`. The mapping happens
entirely in B-spline coefficient space:

```
stratification [6, D]  --P_phys-->  coefficients  --MLP-->  coefficients  --B_stokes-->  Stokes [4, L]
```

`P_phys` (least-squares B-spline projection) and `B_stokes` (B-spline design
matrix) are precomputed once per channel and stored as frozen buffers inside
the deployable `StokesEmulator` module, so synthesis at inference time is
just two matrix multiplies around one MLP forward pass.

## Key features

- **Independent knots per channel.** Every one of the 6 physical channels
  and 4 Stokes channels gets its own `ChannelSplineConfig` (number of
  coefficients, degree, optionally an explicit knot vector).
- **Independent scaling per channel.** Every channel's B-spline
  coefficients get their own `ScalerConfig` (`standard`, `minmax`, a
  `fixed` user-supplied constant, or `none`), and each can be scaled
  per-coefficient or with one scalar for the whole channel — e.g. keep `Q`
  and `U` on their own fixed amplitude scale while `I` is standardized.
- **Precomputed, saved-with-the-model projection matrices.** `P_phys` and
  `B_stokes` (and every scaler's offset/scale) live inside the
  `StokesEmulator`'s `state_dict`, so a checkpoint is fully self-contained.
- **Per-Stokes-parameter loss weights** during training (e.g. upweight `V`).
- **Best-checkpoint saving.** When `checkpoint_path` is given, the emulator
  is written to disk every time validation loss reaches a new best (not
  just once at the end), so the file on disk is always the best model seen
  so far even if training is interrupted.
- **Live GPU stats on the progress bar.** Per-batch `tqdm` bar reports the
  running loss plus GPU utilization/memory (via `nvitop`), when a GPU is
  visible.
- **Periodic ground-truth-vs-emulator plots.** Every `TrainConfig.plot_every`
  epochs (default 10, requires `checkpoint_path`), one validation
  stratification is run through the emulator and the predicted vs.
  ground-truth Stokes profiles are saved as
  `{checkpoint_stem}_comparison_epoch{N}.png` / `.npy` (the `.npy` holding a
  `(2, C_stokes, L)` array: `[ground_truth, prediction]`) next to the
  checkpoint.

## Layout

- `mlp_bspline/bspline.py` — `BSplineProjector` / `BSplineGroup`: knot
  construction, design matrix `B`, projection matrix `P = pinv(B)`.
- `mlp_bspline/scaling.py` — `ChannelScaler` / `MultiChannelScaler`.
- `mlp_bspline/data.py` — projects + scales raw cubes into flat coefficient
  tensors (`prepare_dataset`) and a `Dataset` wrapper.
- `mlp_bspline/model.py` — the `MLP` and the `ResidualMLP`.
- `mlp_bspline/emulator.py` — `StokesEmulator`, the deployable end-to-end
  module, plus `save_emulator` / `load_emulator`.
- `mlp_bspline/train.py` — `train_emulator(...)` and a CLI (`python -m
  mlp_bspline.train ...`).
- `mlp_bspline/synthetic.py` — synthetic data generator used by tests/demo
  only (not physically accurate, just exercises the pipeline).

## Quick start (Python API)

```python
import numpy as np
from mlp_bspline.config import ChannelSplineConfig, ScalerConfig
from mlp_bspline.train import TrainConfig, train_emulator

# physical: [N, 6, D] in channel order T, vz, vmic, Bx, By, Bz
# stokes:   [N, 4, L] in channel order I, Q, U, V
# mu:       [N]
# depth, wave: [D], [L] grids

phys_cfg = {
    "T":    ChannelSplineConfig(num_coeffs=14, degree=3),
    "vz":   ChannelSplineConfig(num_coeffs=10, degree=3),
    "vmic": ChannelSplineConfig(num_coeffs=10, degree=3),
    "Bx":   ChannelSplineConfig(num_coeffs=10, degree=3),
    "By":   ChannelSplineConfig(num_coeffs=10, degree=3),
    "Bz":   ChannelSplineConfig(num_coeffs=14, degree=3),
}
stokes_cfg = {
    "I": ChannelSplineConfig(num_coeffs=24, degree=3),
    "Q": ChannelSplineConfig(num_coeffs=16, degree=3),
    "U": ChannelSplineConfig(num_coeffs=16, degree=3),
    "V": ChannelSplineConfig(num_coeffs=20, degree=3),
}
stokes_scaler_cfg = {
    "I": ScalerConfig(kind="standard"),
    "Q": ScalerConfig(kind="fixed", per_coeff=False, scale=1e-3, offset=0.0),
    "U": ScalerConfig(kind="fixed", per_coeff=False, scale=1e-3, offset=0.0),
    "V": ScalerConfig(kind="minmax"),
}

result = train_emulator(
    physical, stokes, mu, depth, wave, phys_cfg, stokes_cfg,
    stokes_scaler_configs=stokes_scaler_cfg,
    train_cfg=TrainConfig(epochs=300, hidden_dims=(256, 256, 256),
                           stokes_loss_weights={"I": 1.0, "Q": 2.0, "U": 2.0, "V": 3.0}),
    checkpoint_path="emulator.pt",
)

emulator = result.emulator  # ready-to-use StokesEmulator
```

## Inference / synthesis

```python
import torch
from mlp_bspline.emulator import load_emulator

emulator, meta = load_emulator("emulator.pt")
emulator.eval()

physical = torch.as_tensor(physical_batch, dtype=torch.float32)  # [B, 6, D]
mu = torch.as_tensor(mu_batch, dtype=torch.float32)              # [B]
with torch.no_grad():
    stokes = emulator(physical, mu)  # [B, 4, L]
```

## Explicit knot placement

By default each channel gets uniformly spaced interior knots
(`clamped_uniform_knots`). To place Stokes (or physical) knots yourself —
e.g. to concentrate resolution near a line core — use
`clamped_knots_from_positions`, which builds the same kind of clamped knot
vector from explicit interior breakpoint positions instead of spacing them
evenly:

```python
from mlp_bspline.bspline import clamped_knots_from_positions
from mlp_bspline.config import ChannelSplineConfig

degree = 3
positions = [-0.3, -0.1, 0.0, 0.1, 0.3]  # interior knots, dense near line center
knots = clamped_knots_from_positions(wave.min(), wave.max(), positions, degree)

stokes_cfg = {
    "I": ChannelSplineConfig(num_coeffs=len(positions) + degree + 1, degree=degree, knots=knots),
    # ...
}
```

`positions` must be strictly increasing and strictly inside `(x_min,
x_max)`; the boundaries are clamped with multiplicity `degree + 1` at each
end, same as `clamped_uniform_knots`.

## Weak-field residuals for Q, U, V

With `weak_field=WeakFieldConfig(...)`, Q/U/V are fit as residuals with
respect to the weak-field approximation (I is still fit directly):

```
V_WFA = -C gbar l0^2 Bz dI/dl
Q_WFA = -s (1/4) C^2 Gbar l0^4 (Bx^2 - By^2) d2I/dl2
U_WFA = -s (1/4) C^2 Gbar l0^4 (2 Bx By)     d2I/dl2
```

with `C = 4.6686e-13 A^-1 G^-1`, `Bx, By, Bz` the depth average of each
component of the stratification (LOS frame, optionally over a
`depth_range`), and derivatives by finite differences on the wavelength
grid. Training builds the WFA from the true I; the emulator rebuilds it from
its predicted I and adds it back in `decode`, so `emulator(physical, mu)`
still returns physical Stokes profiles. `second_order_lande(g_u, g_l, J_u,
J_l)` computes `Gbar`. `s = linear_polarization_sign` is `-1` for SIR/Hazel
(reference direction of positive Q rotated by 90 deg). It composes with
`stokes_transform`, which is then applied to the residuals.

```python
from mlp_bspline.config import WeakFieldConfig
from mlp_bspline.weak_field import lande_factor, effective_lande, second_order_lande

g = lande_factor(J=2, L=1, S=1)                      # Si I 10827: 3P2 - 3P2
weak_field = WeakFieldConfig(lambda0=10827.089,
                             geff=effective_lande(g, g, 2, 2),       # 1.5
                             Geff=second_order_lande(g, g, 2, 2),    # 2.25
                             linear_polarization_sign=-1.0)
result = train_emulator(..., weak_field=weak_field)
```

Several lines in the same window (e.g. the Fe I 6301.5/6302.5 doublet) each
get their own WFA, with their own `lambda0`, `gbar`, `Gbar` and a
`wave_range` (inclusive, in the units of the wavelength grid passed to
training) where it is applied. Windows may not overlap, and the baseline is
zero outside all of them, so put the boundary in the continuum between the
lines. A single line can also be restricted with `wave_range=`. See
`fe6301.py`.

```python
from mlp_bspline.config import WeakFieldConfig, WeakFieldLine

weak_field = WeakFieldConfig(
    lines=[WeakFieldLine(lambda0=6301.5012, geff=1.667, Geff=2.517, wave_range=(-0.61, 0.5)),
           WeakFieldLine(lambda0=6302.4936, geff=2.5,   Geff=6.25,  wave_range=(0.5001, 1.77))],
    linear_polarization_sign=-1.0)
```

From the CLI: `--wfa-lines '[{"lambda0": ..., "geff": ..., "Geff": ..., "wave_range": [lo, hi]}, ...]'`
(or `--wfa-wave-range lo hi` for a single line).

## PCA basis for the Stokes profiles

Instead of B-spline coefficients, the model can predict principal-component
coefficients of each Stokes parameter (`stokes_basis="pca"`); the physical
stratifications still use B-splines. The basis is fit per channel on the
training split of what is actually fit (after `weak_field` and
`stokes_transform`), diagonalizing whichever of the covariance `X^T X`
(L x L) or Gram `X X^T` (N x N) matrix is smaller. The mean profile and the
eigenvectors are stored in the emulator, so `emulator(physical, mu)` still
returns physical profiles.

```python
from mlp_bspline.config import ChannelPCAConfig, ScalerConfig

result = train_emulator(
    physical, stokes, mu, depth, wave, phys_cfg, None,   # no Stokes spline config needed
    stokes_basis="pca",
    stokes_pca_configs={n: ChannelPCAConfig(variance_fraction=0.99999) for n in "IQUV"},
    # or ChannelPCAConfig(num_components=25)
    stokes_scaler_configs={n: ScalerConfig(kind="standard", per_coeff=False) for n in "IQUV"},
    ...)
```

Use `per_coeff=False` scalers with PCA: the basis is orthonormal, so one
scale per channel keeps the loss proportional to the profile-space MSE,
while per-coefficient standardization gives the noisy trailing components
the same weight as the leading ones. From the CLI: `--stokes-basis pca`
with `--num-pca-stokes N`, `--pca-variance-stokes F` or
`--pca-config-stokes cfg.json`.

## Depth-gradient and cross-coupling features

`depth_features=DepthFeatureConfig(...)` appends extra inputs to the physical
coefficients, computed from them inside the emulator (`mlp_bspline/features.py`),
so `emulator(physical, mu)` and the inversion code are unchanged:

- `derivatives=[...]`: d/d(depth grid) of each listed channel (`"B"` = |B|
  allowed), as feature `"d<channel>"`. With `mode="spline"` it is the analytic
  derivative of the B-spline at `eval_points` (default: its distinct knots);
  with `mode="diff"`, the scaled consecutive-coefficient differences
  `p (c[i+1] - c[i]) / (t[i+p+1] - t[i+1])`, i.e. the derivative spline's
  coefficients.
- `products=[...]`: cross terms `"a*b"` evaluated at `eval_points`, each
  factor a channel value (`"Bz"`), a derivative (`"dvz"`), `"B"` (|B|) or
  `"dB"` (d|B|), e.g. `"dvz*dT"`, `"dvz*dB"`, `"dvz*B"`.

Each feature is standardized per element on the training set, and products
are passed through `asinh` (`squash_products=True`) because they are
very heavy-tailed. See `experiment_si10827.py` for an ablation.

```python
from mlp_bspline.config import DepthFeatureConfig

depth_features = DepthFeatureConfig(derivatives=["T", "vz", "vmic", "Bx", "By", "Bz"],
                                    products=["dvz*dT", "dvz*dB", "dvz*B"])
result = train_emulator(..., depth_features=depth_features)
```

From the CLI: `--depth-features '{"derivatives": ["T", "vz"], "products": ["dvz*dT"]}'`.

## Residual MLP and learning-rate schedule

`model_type="resmlp"` replaces the plain MLP by a residual one: a linear
embedding to `width`, `num_blocks` pre-norm blocks `x + W2 SiLU(W1 LN(x))`
(hidden size `expansion * width`, `W2` zero-initialized so each block starts
as the identity) and a LayerNorm + linear head. `optimizer="adamw"` and
`scheduler="cosine"` (linear warmup over `warmup_epochs`, then per-step
cosine decay to zero at `epochs`; disable early stopping with
`patience=None`) are alternatives to the default Adam + ReduceLROnPlateau.

```python
train_cfg = TrainConfig(epochs=300, batch_size=1024, lr=1e-3,
                        model_type="resmlp", model_kwargs=dict(width=512, num_blocks=6, expansion=2),
                        optimizer="adamw", weight_decay=1e-5, scheduler="cosine", patience=None)
```

On Si I 10827 (300 epochs, `experiment_si10827.py`) this 6.4M-parameter
model lowers the validation loss from 0.0117 (4x256 MLP) to 0.0023 and the
profile rms error by ~2.5x in every Stokes parameter; a plain 4x512 MLP of
similar size to a 256-wide resmlp reaches 0.0073 vs 0.0051.

## Inversion of observations

`invert_fe6301_6302.py` inverts Fe I 6301.5 + 6302.5 observations
(`stokes (nx, ny, 4, nl)`, absolute `wavelength (nl,)`, `mu` scalar or
`(nx, ny)`, in HDF5, same intensity units as the Hazel synthesis) with the
two PCA emulators; the machinery is in `mlp_bspline/inversion.py` and
`mlp_bspline/regularization.py`.

```bash
python invert_fe6301_6302.py observations.h5 inversion.h5
```

- **Nodes.** Each of `T, vz, vmic, Bx, By, Bz` has its own node positions in
  log(tau); an interpolating B-spline of degree `min(3, n-1)` through the
  node values gives the 80-point stratification (constant beyond the
  outer nodes).
- **Velocity.** The emulators were trained with `vz(log tau=-1) = 0`. The
  `vz` nodes are absolute; `v_ref = vz(log tau=-1)` Doppler-shifts the
  synthetic spectrum and the emulators get `vz - v_ref`. Positive = redshift.
- **Wavelengths.** The two emulators' outputs are concatenated, optionally
  convolved with an instrumental profile (`psf={"fwhm": ...}` or
  `(dlambda, profile)`), and linearly interpolated to the observed
  wavelengths. Observed points outside the synthetic range, or NaN, are ignored.
- **Objective.** `sum_pixels chi2 + sum weight * R(node maps)`, with
  `chi2 = sum_s w_s sum_l ((syn - obs) / sigma_s)^2`. Adam (or SGD) steps
  followed by a proximal step for the spatial regularizers and a
  projection onto per-node box bounds (default: the training range at each
  node's height). The prox step uses Adam's effective step size, so the
  weights keep their meaning in the objective.
- **Regularizers** (per variable, in its physical units, optionally one
  weight per node): `Regularizer("l2", w)` (`w ||grad||^2`),
  `Regularizer("tv", w)` (isotropic TV), `Regularizer("l1", w)`,
  `Regularizer("wavelet_l1", w, {"wavelet": "db2", "levels": 3})` (L1 of the
  detail coefficients of an orthonormal periodic wavelet transform; exact
  prox; map sizes divisible by `2**levels` regularize every pixel).
- **Initialization.** Median training stratification for T and vmic,
  centre-of-gravity velocity, weak-field `Bx, By, Bz` (with a floor on
  `B_perp`, since `Bx = By = 0` is a stationary point of Q and U).
- **Output** (HDF5): `nodes/<var>`, `stratification/<var>` (nx, ny, 80),
  `logtau`, `v_ref`, `synthetic`, `chi2` per pixel, `wavelength`,
  `observations`, `history_chi2`.

## CLI

```bash
python -m mlp_bspline.train \
  --physical physical.npy --stokes stokes.npy --mu mu.npy \
  --phys-grid depth.npy --stokes-grid wave.npy \
  --num-knots-phys 12 --num-knots-stokes 24 --degree 3 \
  --scaler-config-stokes stokes_scalers.json \
  --stokes-loss-weights '{"I":1.0,"Q":2.0,"U":2.0,"V":3.0}' \
  --epochs 300 --hidden-dims 256 256 256 \
  --checkpoint emulator.pt
```

Per-channel spline/scaler overrides are passed as JSON, e.g.
`stokes_scalers.json`:

```json
{"Q": {"kind": "fixed", "per_coeff": false, "scale": 1e-3, "offset": 0.0},
 "U": {"kind": "fixed", "per_coeff": false, "scale": 1e-3, "offset": 0.0}}
```

## Tests

```bash
python -m pytest tests/ -q
```


The two passes are there to keep the memory limit that chunking gives you while still getting the exact gradient.

**The problem.** The optimizer never holds an autograd graph for the whole field of view. It processes pixels in chunks (`chunk_size=65536`), and each chunk does a forward pass, `backward()`, and then frees its graph. Your 400×400 FOV is 160 000 pixels, so that's 3 chunks. The memory cost is the graph: all the emulator MLP activations for every pixel.

A spatial PSF breaks the assumption that chunks are independent. The convolved profile at one pixel depends on its neighbours, which may be in other chunks. So the χ² of one chunk can't be computed until every pixel has been synthesized. Keeping all the chunk graphs alive until then would rebuild the full-FOV graph, which is what chunking was avoiding.

**The two-pass trick.** It splits the chain rule at the synthetic maps S. The PSF convolution and χ² are cheap and only need S, not the emulator internals.

1. **Pass 1 (no graph):** synthesize S chunk by chunk under `no_grad`, giving the full maps (npix, 4, nl). That's only about 290 MB for your case.
2. **Full-map step:** make S a leaf tensor, convolve it with the PSF by FFT, compute χ², and call `backward()`. This gives ∂χ²/∂S for every pixel. The graph here covers only the FFT and χ², so it's small.
3. **Pass 2 (with graph, per chunk):** re-run the emulators on each chunk with autograd on, then call `torch.autograd.backward(S_chunk, grad=∂χ²/∂S[chunk])`. This pushes the gradient through the emulators into the node parameters, and each chunk's graph is freed before the next.

The result is the exact gradient, ∂χ²/∂θ = Σ_chunks (∂S_chunk/∂θ)ᵀ · ∂χ²/∂S_chunk. Memory stays at one chunk's graph. The cost is one extra forward pass without a graph, roughly 1.5× an iteration without the PSF.

When the whole FOV fits in a single chunk, the recomputation isn't needed. The code then keeps the graph and does one pass.

The test comparing gradients from one chunk against many chunks was written but hasn't run yet, because the command was blocked by the permission classifier. The two-pass gradient is therefore not yet verified numerically.