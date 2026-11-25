# Installation

## Requirements

| Package | Needed for |
|---------|-----------|
| `numpy`, `scipy` | everything (B-spline matrices, interpolation, eigen-decompositions) |
| `torch` ≥ 2.0 | emulators, inversion (GPU optional but recommended) |
| `h5py` | training models, observations and inversion results |
| `matplotlib`, `tqdm` | plots and progress bars |
| `astropy` *(optional)* | FITS spatial PSFs (`load_psf`), SST/CRISP cubes (`invert_caii8542.py`) |
| `PyWavelets` *(optional)* | the `"wavelet_l1"` regularizer for Daubechies, Symlet and Coiflet wavelets (Haar works without it) |
| `hazel` *(optional)* | the Hazel2 comparison in `synth_fe6301_si10827_he10830.py` |
| `pyqtgraph`, `PyQt6` *(optional)* | `view_inversion.py` |

```bash
git clone <repository-url>
cd inv_emulator
pip install numpy scipy torch h5py matplotlib tqdm
pip install astropy PyWavelets pyqtgraph PyQt6   # optional
```

The code lives in the local `modules/` package. Run the scripts from the
repository root so that `from modules... import ...` resolves.

## Emulator weights

The checkpoints are found in [here](https://cloud.iac.es/index.php/s/sJJE7DSXMTwnntA). 
Place them in `weights/`:

```
weights/
├── emulator_si10827_pca_resmlp512_cosine_deriv.pt
├── emulator_fe6301_pca_resmlp512_cosine_deriv.pt
├── emulator_fe6302_pca_resmlp512_cosine_deriv.pt
├── emulator_he10830_pca.pt
└── emulator_caii8542_pca_resmlp512_cosine_deriv.pt
```

Each photospheric checkpoint is self-contained. It stores the network weights
together with the B-spline projection matrices, the PCA eigenvectors and mean
profiles, the scalers, and a `meta` dictionary describing the configuration.
See [Emulator architecture](architecture.md#the-checkpoint).

## Models

The validation models from which some statistics are extracted
are also found in [here](https://cloud.iac.es/index.php/s/sJJE7DSXMTwnntA). 
Place them in `models/`:

## Training-model database

The inversion scripts read the training model atmospheres, `MODELS =
.../models_validation.h5` (dataset `model`, shape `(N, 80, 8)`, with columns
`logtau, T, Pe, vmic, v, Bx, By, Bz`). They use them to obtain:

* the depth grid $\log\tau$ (80 points from 1.5 down to −7.5),
* the median stratification (initial $T$ and $v_\mathrm{mic}$),
* the 0.1–99.9 percentile range at each height (the box bounds of every
  node).

Edit the `MODELS` path at the top of each script to point to your copy.

## Device

`InversionConfig.device` defaults to `"cuda"` when a GPU is visible and to
`"cpu"` otherwise. The emulators run in float32. Wavelengths are handled
*relative* to a reference wavelength inside each window, because an absolute
float32 wavelength of ~6300 Å has a resolution of ~5·10⁻⁴ Å (≈ 0.02 km/s),
which would quantize Doppler shifts.
