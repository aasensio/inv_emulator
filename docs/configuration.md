# Configuring an inversion

Every `invert_*.py` script has the same structure: configuration at the top,
no command-line arguments except in `invert_multiregion.py`. To adapt a
script, edit these blocks in order.

## 1. Forward model

```python
from modules.inversion import LineEmulator, SpectralSynthesizer

lines = [LineEmulator("weights/<checkpoint>.pt", absolute_wavelengths_of_its_output)]
synthesizer = SpectralSynthesizer(lines, psf=None, telluric=None)
```

| Argument | Values |
|----------|--------|
| `lines` | list of `LineEmulator(checkpoint, wave)`. `wave` must have as many points as the emulator outputs; the grids must not overlap |
| `psf` | `None`, `{"fwhm": x}` (Gaussian, Å), or `(dlambda, profile)` arrays; needs a uniform synthetic grid |
| `telluric` | `None`, a wavelength (prefix `"Tel_"`), a `TelluricLine(wavelength, prefix)`, or a list of them |
| `map_location` | device to load the checkpoints on (default `"cpu"`; the inverter moves them) |

For Si I + He I:

```python
from modules.hei import SiHeSynthesizer
from modules.hei_pca import HeIPCAEmulator
synthesizer = SiHeSynthesizer(photosphere, HeIPCAEmulator.load("weights/emulator_he10830_pca.pt"),
                              step=0.02, psf=None, pumping="boundary", telluric=10832.0,
                              prefix=("He1_", "He2_"), hazel_boundary_tilt=True, wave_range=None)
```

## 2. Depth grid, median stratification and bounds

```python
with h5py.File(MODELS, "r") as f:
    models = f["model"][:]
models[:, :, 4] -= models[:, 22:23, 4]                 # same preprocessing as training: v(log tau=-1) = 0
logtau = models[0, :, 0].astype(np.float64)            # (80,), 1.5 ... -7.5
columns = {"T": 1, "vz": 4, "vmic": 3, "Bx": 5, "By": 6, "Bz": 7}
mean_strat = {k: np.median(models[:, :, c], axis=0) for k, c in columns.items()}
lo_strat = {k: np.percentile(models[:, :, c], 0.1, axis=0) for k, c in columns.items()}
hi_strat = {k: np.percentile(models[:, :, c], 99.9, axis=0) for k, c in columns.items()}
lo_strat["Bx"] *= 0                                    # Bx >= 0: removes the 180-degree ambiguity
```

`training_statistics(MODELS)` returns the same information as an
`AtmosphereStatistics` object, with `.bounds(name, nodes)`.

## 3. Variables: `VariableConfig`

```python
VariableConfig(nodes, scale, bounds=None, regularization=(), fixed=False, depth=True)
slab_variable(scale, bounds=None, regularization=(), fixed=False)   # one value per pixel
```

| Field | Meaning |
|-------|---------|
| `nodes` | strictly increasing $\log\tau$ positions. The stratification is the interpolating B-spline of degree `min(3, n−1)`, constant beyond the outer nodes ([details](bsplines.md#from-nodes-to-stratification)) |
| `scale` | typical size of the variable; the optimizer works on `value / scale` |
| `bounds` | `(lower, upper)`, each a scalar or one value per node, in physical units; enforced by projection |
| `regularization` | list of `Regularizer` acting on every node map |
| `fixed` | keep the variable at its initial value |
| `depth` | `False` for per-pixel variables (use `slab_variable`) |

`InversionConfig` requires all six photospheric variables `T, vz, vmic, Bx,
By, Bz`. Slab and telluric variables are added to the same dictionary under
the names the synthesizer reads (`synthesizer.slab_names`).

Default helpers:

* `photosphere_variables(nodes, stats, scales=None, regularization=None, fixed=(), vz_bounds=(-20, 20), positive_bx=True)`
  uses the training-percentile bounds at each node height, `vmic ≥ 0`, and
  `Bx ≥ 0`;
* `he_slab_variables(prefix, fixed=("beta", "height"), field_bound=1500)`;
* `telluric_variables(TelluricLine, shift_range=0.3, fixed=())`.

### Typical scales and bounds

| Variable | `scale` | `bounds` |
|----------|--------:|----------|
| `T` | 500 K | training percentiles per node |
| `vz` | 1 km/s | (−20, 20) km/s, absolute |
| `vmic` | 0.5 km/s | (0, 3.4) km/s |
| `Bx` | 200 G | (0, training upper bound) |
| `By`, `Bz` | 200 G | training percentiles per node |
| `He*_Bx` | 100 G | (0, 1500) |
| `He*_By`, `He*_Bz` | 100 G | (−1500, 1500) |
| `He*_tau` | 0.3 | (0.01, 5) |
| `He*_v` | 1 km/s | script dependent |
| `He*_deltav` | 1 km/s | (4, 18) km/s (emulator training range) |
| `He*_a` | 0.1 | (0, 1.5) |
| `He*_beta` | 0.1 | (0, 5), usually fixed |
| `He*_height` | 0.5″ | (0.1, 10), usually fixed |
| `Tel_shift`, `Tel_width` | 0.01 Å | (−0.3, 0.3), (0.005, 0.3) Å |
| `Tel_depth`, `Tel_a` | 0.05, 0.1 | (0, 1), (0, 2) |

## 4. Regularization

```python
from modules.regularization import Regularizer, TransverseRegularizer

Regularizer("l2", weight)                                         # weight * sum |grad x|^2
Regularizer("tv", weight, {"n_iter": 50})                         # isotropic total variation
Regularizer("l1", weight)                                         # weight * sum |x|
Regularizer("wavelet_l1", weight, {"wavelet": "db2", "levels": 3})
TransverseRegularizer("Bx", "By", [Regularizer(...), ...])        # on Bt (cos 2phi, sin 2phi)
```

`weight` is a scalar or a list with one value per node. The scripts write
weights as `smooth(jump) = [Regularizer("l2", REG_STRENGTH / jump**2)]`; see
{ref}`choosing weights <choosing-weights>`.
For `wavelet_l1`, map sizes divisible by `2**levels` regularize every pixel.

## 5. `InversionConfig`

```python
config = InversionConfig(
    variables=variables,
    transverse_regularization=[TransverseRegularizer("Bx", "By", smooth(100.0))],
    stokes_weights={"I": 1.0, "Q": 1.0, "U": 1.0, "V": 1.0},
    noise={"I": 7.2e-3, "Q": 1.6e-3, "U": 1.5e-3, "V": 2.9e-3},
    optimizer="adam", lr=0.1, lr_final_fraction=0.05, n_iter=500,
    chunk_size=65536, log_every=5,
)
```

| Field | Default | Meaning |
|-------|---------|---------|
| `variables` | (required) | name → `VariableConfig` |
| `stokes_weights` | all 1 | $w_s$ in $\chi^2$ |
| `noise` | all 10⁻³ | $\sigma_s$ in $\chi^2$, in the units of the observations |
| `optimizer` | `"adam"` | `"adam"`, `"adamw"` or `"sgd"` |
| `lr` | 0.02 | initial learning rate (in units of `scale`) |
| `lr_final_fraction` | 0.05 | cosine decay to `lr · lr_final_fraction` |
| `n_iter` | 500 | iterations |
| `chunk_size` | 65536 | pixels per forward/backward pass |
| `reference_depth` | −1.0 | $\log\tau$ at which the emulators have $v_z=0$ |
| `device` | `"cuda"` if available | |
| `log_every` | 25 | log interval |
| `transverse_regularization` | `()` | list of `TransverseRegularizer` |

## 6. Observations

Every script ends up with

```python
observations    # (nx, ny, 4, nl)  I, Q, U, V in the emulators' intensity units; NaN = missing
wave_obs        # (nl,)            absolute wavelengths, A
mu              # scalar or (nx, ny)
```

The intensity units must match the training: the quiet-Sun disk-centre
continuum for Si I/He I and Fe I, and $I_\mathrm{norm}$ for Ca II. The
scripts read different formats (HDF5, FITS, text wavelength files), so
adapt this block to your data.

## 7. Initial guess

```python
from modules.config import WeakFieldConfig, WeakFieldLine
from modules.weak_field import lande_factor, effective_lande, second_order_lande

g = lande_factor(J=2, L=1, S=1)                       # Si I 10827
wfa = WeakFieldConfig(lines=[WeakFieldLine(lambda0=10827.089, geff=effective_lande(g, g, 2, 2),
                                           Geff=second_order_lande(g, g, 2, 2),
                                           wave_range=(10826.589, 10827.589))],
                      linear_polarization_sign=-1.0)
init = initial_guess(observations, wave_obs, config, logtau, mean_strat, wfa=wfa,
                     cog_lines=[(10827.089, 0.25)])
init |= he_slabs_initial_guess(observations, wave_obs, mu, prefixes=("He1_", "He2_"), height=3.0)
```

`init` maps every variable to an array broadcastable to `(nx, ny, n_nodes)`.
It can be edited freely before the run, for example to start from a
previous inversion.

## 8. Run and save

```python
inverter = Inverter(synthesizer, config, logtau, spatial_psf=None)
result = inverter.run(observations, wave_obs, mu, init)
save_result("inversions/result.h5", result, wave_obs, observations)
```

For several spectral regions, see
[`invert_multiregion.py`](scripts/multiregion.md).
