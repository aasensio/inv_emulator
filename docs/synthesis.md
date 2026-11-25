# Synthesis

This page explains how to compute Stokes profiles with the emulators,
following `synth_fe6301_si10827_he10830.py`. That script synthesizes one
model atmosphere:

1. **Si I 10827 + He I 10830.** The Si I photosphere with a He I slab on top,
   for three bulk velocities (0, 1, 2 km/s at $\log\tau=-1$).
2. **Fe I 6301 + 6302.** Photosphere only.

Each case is compared with a Hazel2 synthesis of the same atmosphere.

## Three levels of access

| Level | Object | Input | Output |
|-------|--------|-------|--------|
| single emulator | `StokesEmulator` (`load_emulator`) | stratification with $v_z(-1)=0$, μ | Stokes on the emulator's own grid |
| one or more photospheric lines | `SpectralSynthesizer` | the same + $v_\mathrm{ref}$, observed wavelengths, telluric parameters | Stokes at the requested wavelengths, Doppler-shifted, optionally PSF-convolved |
| photosphere + He I slabs | `SiHeSynthesizer` | the same + slab parameters | Stokes of the full Si I + He I window |

`SpectralSynthesizer` and `SiHeSynthesizer` have the interface used by the
inverters, so whatever you synthesize with them is exactly what the inversion
fits.

## Step 1: prepare the stratification

The training models are stored as `(80, 8)` arrays with columns
`logtau, T, Pe, vmic, v, Bx, By, Bz`. The 80 depths run from the deepest
point ($\log\tau=1.5$) upwards to $-7.5$. The emulators take the six
channels in the order `T, vz, vmic, Bx, By, Bz`:

```python
import h5py, numpy as np, torch

with h5py.File(MODELS, "r") as f:
    model = f["model"][123].astype(np.float64)        # (80, 8)

model[:, 4] += 1.0 - model[22, 4]                     # e.g. impose 1 km/s at log tau = -1

v_ref = model[22, 4]                                  # velocity at log tau = -1 (index 22)
physical = model[:, [1, 4, 3, 5, 6, 7]].T.copy()      # (6, 80): T, vz, vmic, Bx, By, Bz
physical[1] -= v_ref                                  # emulators expect vz(log tau = -1) = 0
```

Units: $T$ in K, velocities in km/s (positive = redshift/downflow),
magnetic field components in G. $B_z$ is along the line of sight and
$B_x, B_y$ are perpendicular to it.

(velocities)=
### Velocities

The emulators were trained with $v_z(\log\tau=-1)=0$. The stratification
is therefore split into a reference velocity $v_\mathrm{ref}$ plus a depth
variation $\delta v(\tau)$ that vanishes at $\log\tau=-1$:

* $\delta v(\tau)$ goes into the emulator and produces the asymmetries;
* $v_\mathrm{ref}$ is passed separately and applied as a relativistic
  Doppler shift of the whole synthetic spectrum,
  $S_\mathrm{obs}(\lambda) = S\!\left(\lambda/(1+v_\mathrm{ref}/c)\right)$.

Doppler shifts can therefore be arbitrary (within the ±20 km/s bounds used
in the inversion), even though the training set had none.

## Step 2a: Si I 10827 + He I 10830

```python
from modules.hei import SiHeSynthesizer
from modules.hei_pca import HeIPCAEmulator
from modules.inversion import LineEmulator, SpectralSynthesizer

synth = SiHeSynthesizer(
    SpectralSynthesizer([LineEmulator("weights/emulator_si10827_pca_resmlp512_cosine_deriv.pt",
                                      np.linspace(10824.0, 10833.0, 350))]),
    HeIPCAEmulator.load("weights/emulator_he10830_pca.pt"),
    step=0.02)                         # synthesis grid step for the slab (A)

# He I slab: field in G in Hazel's vertical frame, velocities in km/s, height in arcsec
he = dict(He_Bx=200.0, He_By=-100.0, He_Bz=300.0, He_tau=1.0, He_v=2.0,
          He_deltav=8.0, He_a=0.3, He_beta=1.0, He_height=3.0)

WAVE = np.linspace(10824.3, 10832.7, 421)    # requested (observed) wavelengths, A
MU = 1.0

with torch.no_grad():
    stokes, valid = synth(
        torch.tensor(physical[None], dtype=torch.float32),                       # (B, 6, 80)
        torch.tensor([MU]),                                                      # (B,)
        torch.tensor([v_ref], dtype=torch.float32),                              # (B,)
        torch.tensor(synth.relative_wavelengths(WAVE), dtype=torch.float32),     # (nl,)
        {k: torch.tensor([v], dtype=torch.float32) for k, v in he.items()})      # name -> (B,)
stokes = stokes[0].numpy()        # (4, 421): I, Q, U, V in units of the quiet-Sun continuum
```

Notes:

* **`relative_wavelengths`.** The synthesizers work internally with
  wavelengths relative to `wave_ref` (the centre of the window) to avoid
  float32 precision loss. Always convert absolute wavelengths with
  `synth.relative_wavelengths(wave)`, which computes in float64.
* **`valid`** (`(B, nl)` bool) marks the requested wavelengths that are
  covered by the shifted synthetic grid. Points outside are extrapolated and
  should be ignored, as the inversion does.
* **Slab parameters** are passed as a dictionary of `(B,)` tensors. With the
  default `prefix="He_"` the keys are `He_Bx, He_By, He_Bz, He_tau, He_v,
  He_deltav, He_a, He_beta, He_height`. For two slabs in series, use
  `SiHeSynthesizer(..., prefix=("He1_", "He2_"))` and supply both sets.
* **Options** of `SiHeSynthesizer`: `psf` (instrumental profile, see below),
  `pumping="boundary" | "quiet"`, `hazel_boundary_tilt=True`,
  `telluric=10832.0` (then `Tel_shift, Tel_width, Tel_depth, Tel_a` must also
  be in the dictionary), and `wave_range` (default: the Si I emulator range).

## Step 2b: Fe I 6301 + 6302

```python
LAMBDA_6301, LAMBDA_6302 = 6301.5012, 6302.4936
wave_fe = np.linspace(6300.8921, 6303.2671, 150)

fe_synth = SpectralSynthesizer(
    [LineEmulator("weights/emulator_fe6301_pca_resmlp512_cosine_deriv.pt", wave_fe[wave_fe - LAMBDA_6301 < 0.5]),
     LineEmulator("weights/emulator_fe6302_pca_resmlp512_cosine_deriv.pt", wave_fe[wave_fe - LAMBDA_6302 > -0.5])],
    psf=None,            # e.g. {"fwhm": 0.025} (A) or (dlambda, profile) arrays
    telluric=None,       # e.g. [TelluricLine(6302.0005, "O2a_"), TelluricLine(6302.7629, "O2b_")]
)

with torch.no_grad():
    stokes_fe, valid = fe_synth(
        torch.tensor(physical[None], dtype=torch.float32), torch.tensor([MU]),
        torch.tensor([v_ref], dtype=torch.float32),
        torch.tensor(fe_synth.relative_wavelengths(wave_fe), dtype=torch.float32),
        None)                                         # no per-pixel slab/telluric variables
```

`SpectralSynthesizer.forward` runs the following steps:

1. evaluate every emulator and concatenate the outputs on their absolute
   grids (`synthesize(physical, mu)`, shape `(B, 4, 150)`);
2. multiply Stokes I by the telluric transmission, if any (observer's
   frame);
3. convolve with the instrumental profile, if any (`conv1d` with replicate
   padding; requires a uniform synthetic grid);
4. Doppler-shift by $v_\mathrm{ref}$ and linearly interpolate to the
   requested wavelengths.

### Instrumental profile

`psf` accepts

* `None`;
* `{"fwhm": x}`: a Gaussian of FWHM `x` Å, sampled out to 4σ on the synthetic
  step;
* `(dlambda, profile)`: arbitrary arrays (Å offsets and values), interpolated
  to the synthetic step and normalized to unit sum. `invert_caii8542.py`, for
  example, passes the CRISP transmission profile this way.

## Using a single emulator directly

When only the emulator's own grid is needed, with no Doppler shift, PSF or
interpolation:

```python
from modules.emulator import load_emulator
em, meta = load_emulator("weights/emulator_si10827_pca_resmlp512_cosine_deriv.pt")
em.eval()
with torch.no_grad():
    s = em(torch.tensor(physical[None], dtype=torch.float32), torch.tensor([MU]))   # (1, 4, 350)
```

## Batches and gradients

Every input has a leading batch dimension. Synthesizing a map is the same
call with `B = nx·ny`, split into chunks that fit in memory. Without
`torch.no_grad()` the output is differentiable with respect to `physical`,
`v_ref` and the slab parameters, so a Jacobian (response function) can be
computed with autograd:

```python
x = torch.tensor(physical[None], dtype=torch.float32, requires_grad=True)
wave_rel = torch.tensor(fe_synth.relative_wavelengths(wave_fe), dtype=torch.float32)
s, _ = fe_synth(x, torch.tensor([MU]), torch.tensor([v_ref], dtype=torch.float32), wave_rel, None)
s[0, 3].sum().backward()          # d(sum of V)/d(stratification)
response_V = x.grad[0]             # (6, 80)
```

## Comparison with Hazel2

The script runs Hazel2 on the same atmosphere and overlays the profiles. The
title of each panel shows the maximum absolute difference. The Hazel setup
mirrors the emulator configuration:

```python
mod = hazel.Model(working_mode="synthesis", verbose=False)
mod.add_spectral({"Name": "spec1", "Wavelength": [WAVE[0], WAVE[-1], len(WAVE)], "topology": "ph1->ch1",
                  "LOS": [np.degrees(np.arccos(MU)), 0.0, 90.0], "Boundary condition": [1.0, 0.0, 0.0, 0.0]})
mod.add_photosphere({"Name": "ph1", "Spectral region": "spec1", "Spectral lines": [300],
                     "Wavelength": [WAVE[0], WAVE[-1]]})
mod.add_chromosphere({"Name": "ch1", "Spectral region": "spec1", "Height": he["He_height"], "Line": "10830",
                      "Wavelength": [WAVE[0], WAVE[-1]], "Reference frame": "vertical"})
mod.setup()
mod.atmospheres["ph1"].set_parameters(model, 1.0, 0.0)          # the full (80, 8) model, absolute v
mod.atmospheres["ch1"].set_parameters([he[k] for k in ("He_Bx", "He_By", "He_Bz", "He_tau", "He_v",
                                                        "He_deltav", "He_beta", "He_a")], 1.0)
mod.synthesize()
stokes_hazel = mod.spectrum["spec1"].stokes
```

For Fe I, use topology `"ph1"`, spectral lines `[200, 201]`, and no
chromosphere. Hazel receives the model with its **absolute** velocity. The
emulator path receives $\delta v$ and $v_\mathrm{ref}$ separately and
should agree with it.
