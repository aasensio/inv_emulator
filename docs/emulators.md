# The Si I, Fe I and He I emulators

All emulators were trained on Hazel2 syntheses. The photospheric ones
(Si I, Fe I) map a stratification to Stokes profiles. The He I one maps the
parameters of a constant-property chromospheric slab to its radiative
transfer coefficients.

## Summary

| | Si I 10827 | Fe I 6301 | Fe I 6302 | He I 10830 |
|-|-----------|-----------|-----------|-----------|
| checkpoint | `emulator_si10827_pca_resmlp512_cosine_deriv.pt` | `emulator_fe6301_pca_resmlp512_cosine_deriv.pt` | `emulator_fe6302_pca_resmlp512_cosine_deriv.pt` | `emulator_he10830_pca.pt` |
| training script | `si10827.py` | `fe6301.py` | `fe6302.py` | `he10830.py` |
| rest wavelength | 10827.089 Å | 6301.5012 Å | 6302.4936 Å | 10829.09, 10830.25, 10830.34 Å |
| inputs | 6×80 stratification + μ | 6×80 + μ | 6×80 + μ | Bx, By, Bz, Δv, a, height, θ |
| depth basis | 10 cubic B-splines per variable + derivative splines | same | same | n/a |
| output grid | 350 λ, `linspace(10824, 10833, 350)` | 70 λ | 80 λ | 256 λ on [10823, 10834] Å |
| output basis | PCA 18/49/49/44 | PCA 22/39/40/36 | PCA 20/34/34/32 | PCA 30 × 11 channels |
| transform | symlog(100·X/I) | symlog(30·X/I) | symlog(30·X/I) | divide by max(η_I) |
| network | ResMLP 512 × 6 | ResMLP 512 × 6 | ResMLP 512 × 6 | MLP 4 × 512 |

## Si I 10827

Si I 10827.089 Å (3s² 3p 4s ³P₂ – 3s² 3p 4p ³P₂, SIR line 300) is a
photospheric line in the same spectral window as He I 10830. Both levels have
$g=1.5$, so $\bar g = 1.5$ and $\bar G = 2.25$.

* **Training data:** model atmospheres from the `models_training.h5`
  database, with $v_z$ at $\log\tau=-1$ removed, synthesized with
  Hazel2 at random μ.
* **Wavelengths:** 350 points on $[10824, 10833]$ Å (step ≈ 0.026 Å),
  relative to $\lambda_0$ during training. Use
  `LineEmulator(checkpoint, np.linspace(10824.0, 10833.0, 350))`.
* **Use:** on its own through `SpectralSynthesizer`, or as the photospheric
  boundary condition of the He I slab in `SiHeSynthesizer`.

## Fe I 6301 and Fe I 6302

The Hinode SP doublet was synthesized together on the Hazel grid
`linspace(6300.8921, 6303.2671, 150)`, then split at the continuum point
between the lines. Each line has its own emulator:

```python
wave_syn = np.linspace(6300.8921, 6303.2671, 150)
LineEmulator("weights/emulator_fe6301_pca_resmlp512_cosine_deriv.pt", wave_syn[wave_syn - 6301.5012 < 0.5])   # 70 points
LineEmulator("weights/emulator_fe6302_pca_resmlp512_cosine_deriv.pt", wave_syn[wave_syn - 6302.4936 > -0.5])  # 80 points
```

`SpectralSynthesizer` sorts the emulators by wavelength, concatenates their
outputs into one 150-point spectrum, and checks that the grids do not overlap.

| Line | Transition | $\bar g$ |
|------|-----------|-------------|
| Fe I 6301.5012 | ⁵P₂ – ⁵D₂ | 1.667 |
| Fe I 6302.4936 | ⁵P₁ – ⁵D₀ | 2.5 |

```{note}
The Fe I emulators predict Stokes I in units of the Hazel continuum. In
`invert_fe6301_6302.py` the Hinode data are divided by the median continuum
of a quiet-Sun patch for this reason.
```

## He I 10830

The He I triplet forms in the chromosphere. Following Hazel2, it is modelled
as a **constant-property slab** illuminated from below by the photospheric
spectrum (Si I emulator) and solved exactly. The emulator (`modules/hei_pca.py`,
class `HeIPCAEmulator`) replaces only the expensive part: the atomic
calculation of the slab's

* emission vector $\boldsymbol\epsilon = (\epsilon_I,\epsilon_Q,\epsilon_U,\epsilon_V)$,
* absorption coefficients $\boldsymbol\eta = (\eta_I,\eta_Q,\eta_U,\eta_V)$,
* magneto-optical coefficients $\boldsymbol\rho = (\rho_Q,\rho_U,\rho_V)$,

as functions of the 7 inputs $(B_x, B_y, B_z, \Delta v_D, a, h, \theta)$.
The field is in G in Hazel's *vertical* (local solar) frame, $\Delta v_D$
is the Doppler width in km/s, $a$ the damping, $h$ the height in
arcsec, and $\theta=\arccos\mu$ in degrees.

### PCA targets

The training database (1 M Hazel2 models, 10 k for validation) was computed
for $v=0$, $\tau=1$, $\beta=1$ and quiet-Sun illumination. The
slab solution only depends on the coefficients relative to $\max\eta_I$,
so every profile is divided by $\max\eta_I$ of its model, and
$\epsilon$ is also divided by the Allen continuum at 10828.5 Å. The 11
normalized profiles on the 256-point grid are smooth, and a plain linear PCA
represents them very compactly:

```python
pca_cfg = {c: ChannelPCAConfig(num_components=30) for c in HEI_CHANNELS}   # 11 channels x 30
input_scaler_cfg = ScalerConfig(kind="standard", per_coeff=True)           # each input standardized
output_scaler_cfg = {c: ScalerConfig(kind="standard", per_coeff=False) for c in HEI_CHANNELS}
```

The PCA is fit on the first 200 k training models. The network is an MLP
with 4 hidden layers of 512 units. Unlike the per-wavelength neural field it
replaces, it runs **once per pixel**. The profiles are rebuilt on the
training grid and interpolated with a cubic spline to the requested
(Doppler-shifted) wavelengths.

### Slab formal solution

For each slab with optical depth $\tau$ (measured at the maximum of
$\eta_I$), set $\mathrm{d}s=\tau/\max\eta_I$. With the propagation
matrix $K(\eta,\rho)$ and the incoming Stokes vector $\mathbf I_\mathrm{in}$,

$$
\mathbf I_\mathrm{out} = e^{-K\,\mathrm{d}s}\,\mathbf I_\mathrm{in}
 + K^{-1}\left(\mathbb 1-e^{-K\,\mathrm{d}s}\right)\beta\,\boldsymbol\epsilon .
$$

Both operators are computed analytically (`slab_operators` in
`modules/hei.py`, Landi Degl'Innocenti & Landolfi 2004, Eq. A5.20). The code
does not invert $K$, which would be ill-conditioned in float32 in the far
wings. Taylor series handle the unpolarized limit. In float32 the result
agrees with a matrix exponential to ~2·10⁻⁷.

### Combining photosphere and slabs: `SiHeSynthesizer`

```python
from modules.hei import SiHeSynthesizer
from modules.hei_pca import HeIPCAEmulator

photosphere = SpectralSynthesizer([LineEmulator("weights/emulator_si10827_pca_resmlp512_cosine_deriv.pt",
                                                np.linspace(10824.0, 10833.0, 350))])
chromosphere = HeIPCAEmulator.load("weights/emulator_he10830_pca.pt")
synth = SiHeSynthesizer(photosphere, chromosphere, step=0.02, psf=None,
                        pumping="boundary", telluric=10832.0, prefix=("He1_", "He2_"))
```

1. The Si I spectrum is computed and Doppler-shifted by $v_\mathrm{ref}$.
   It is then interpolated with a **cubic spline** to a uniform grid of step
   `step` (0.02 Å). Linear interpolation would flatten the Q/U/V peaks by
   2–10 %.
2. The coefficients of all slabs are computed in one emulator call. Each slab
   is Doppler-shifted by its own `<prefix>v`.
3. The slabs are traversed **in series** from the photosphere outwards (Hazel
   topology `ph1->ch1->ch2`). Each one uses the emergent spectrum of the
   previous layer as its boundary condition.
4. **Pumping** (`pumping="boundary"`, the default, as in Hazel2): the source
   function of each slab is scaled by the ratio of its boundary intensity
   (first wavelength, which must be continuum) to the Allen quiet-Sun value
   at the same μ, so a dark umbral photosphere illuminates the slab less.
   `pumping="quiet"` keeps the training illumination.
5. `hazel_boundary_tilt=True` applies Hazel's small Allen continuum slope
   (±0.1 %) at every boundary.
6. Optional telluric lines multiply Stokes I. The instrumental profile `psf`
   is then applied, and the result is interpolated to the observed
   wavelengths.

Output intensities are in units of the quiet-Sun disk-centre continuum.

#### Slab variables

With prefix `p` (default `"He_"`) the per-pixel variables are:

| Variable | Meaning | Units | Default bounds (`he_slab_variables`) |
|----------|---------|-------|---------------------------|
| `pBx`, `pBy`, `pBz` | magnetic field, vertical frame | G | ±1500 |
| `ptau` | optical depth at the red-component peak | | 0.01–5 |
| `pv` | LOS velocity (+ = redshift) | km/s | ±20 |
| `pdeltav` | Doppler width | km/s | 4–18 (training range) |
| `pa` | damping | | 0–1.5 (training range) |
| `pbeta` | source-function enhancement | | 0–5, fixed by default |
| `pheight` | height above the surface | arcsec | 0.1–10, fixed by default |

#### Telluric lines

`TelluricLine(wavelength, prefix="Tel_")` (`modules/telluric.py`) multiplies
Stokes I by $1-d\,H\!\left(a,\frac{\lambda-\lambda_0}{\Delta\lambda}\right)$
in the observer's frame, before the instrumental profile. $H$ is the Voigt
function. The per-pixel variables are `<prefix>shift` ($\lambda_0$ minus the
reference wavelength, Å), `<prefix>width` (Å), `<prefix>depth` and `<prefix>a`.

## Ca II 8542 (additional)

`emulator_caii8542_pca_resmlp512_cosine_deriv.pt` (trained with
`caii8542.py`) has the same architecture as the Fe I and Si I emulators. It
uses 150 wavelengths from 8539.8357 Å with a 0.03 Å step and a PCA of
38/55/55/51 components. It works in absolute intensity units, normalized by
$I_\mathrm{norm} = 2.59\cdot10^{-5}$ erg s⁻¹ cm⁻² Hz⁻¹ sr⁻¹. See
[`invert_caii8542.py`](scripts/caii8542.md).
