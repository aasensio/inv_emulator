"""Joint inversion of several spectral regions with one model atmosphere:
Fe I 6301.5 + 6302.5 (Hinode SP-like, with two telluric O2 lines) and
Si I 10827 + He I 10830 (with a telluric line), observed on the same pixels.

    python invert_multiregion.py [region ...] [--output file.h5] [--iterations N]

With no arguments every region below is inverted jointly; naming some
(e.g. ``fe630``) inverts only those, with the same configuration.

How it fits together
--------------------
1. Each REGION bundles its observations (HDF5 with stokes (nx, ny, 4, nl),
   wavelength (nl,) in A, mu), the synthesizer that models them (emulators,
   instrumental profile, telluric lines, He I slab, ...), its noise and its
   weight. All regions share the pixel grid.
2. VARIABLES are shared by name. Every region uses the photospheric nodes
   T, vz, vmic, Bx, By, Bz (each on its own emulators' depth grid); extra
   components bring their own variables, named with a prefix (He I slab:
   He_*, telluric lines: O2a_*, O2b_*, Tel_*). Use the same prefix in two
   regions to share a component, different ones to keep them apart.
3. The objective is sum over pixels of sum_regions weight * chi2 / (number
   of points), so each region counts by its weight whatever its number of
   wavelengths, plus the spatial regularization.

To add a region (e.g. Ca II 8542 once its emulator is trained), add a
Region to REGIONS, any variables its synthesizer needs, and their initial
values. If an emulator was trained with absolute velocities (no vz = 0 at
log tau = -1), give its Region reference_depth=None.

The inputs here are synthetic (make_synthetic_si10827_he10830.py,
make_synthetic_fe6301_6302.py; true atmospheres in true/), with noise added.
"""
import argparse

import numpy as np

from modules.config import WeakFieldConfig, WeakFieldLine
from modules.hei import SiHeSynthesizer, he_slab_initial_guess, he_slab_variables
from modules.hei_pca import HeIPCAEmulator
from modules.inversion import (
    InversionConfig, LineEmulator, MultiRegionInverter, Region, SpectralSynthesizer, initial_guess,
    photosphere_variables, save_multiregion_result, training_statistics,
)
from modules.regularization import Regularizer
from modules.telluric import TelluricLine, telluric_initial_guess, telluric_variables
from modules.weak_field import effective_lande, lande_factor, second_order_lande

MODELS = 'models/models_validation.h5'
NOISE = 1e-3   # added to the (noiseless) synthetic observations; None for real data

LAMBDA_6301, LAMBDA_6302 = 6301.5012, 6302.4936
LAMBDA_SI = 10827.089

# --------------------------------------------------------------------------- #
# Regions: observations + synthesizer + noise (+ weight, wave_range, ...)
# --------------------------------------------------------------------------- #
# Fe I: the two PCA emulators on the Hazel grid they were trained on, split
# as in fe6301.py / fe6302.py, and the telluric O2 lines next to 6302.5.
wave_fe = np.linspace(6300.8921, 6303.2671, 150)
o2 = [TelluricLine(6302.0005, prefix="O2a_"), TelluricLine(6302.7629, prefix="O2b_")]
fe_synth = SpectralSynthesizer(
    [LineEmulator("emulator_fe6301_pca.pt", wave_fe[wave_fe - LAMBDA_6301 < 0.5]),
     LineEmulator("emulator_fe6302_pca.pt", wave_fe[wave_fe - LAMBDA_6302 > -0.5])],
    psf=None,            # e.g. {"fwhm": 0.025} (A) or (dlambda, profile) arrays
    telluric=o2,
)

# Si I + He I: Si I emulator for the photosphere, He I slab on top (PCA
# emulator + exact formal solution), telluric line to the red of He I.
tel_10832 = TelluricLine(10832.0, prefix="Tel_")
sihe_synth = SiHeSynthesizer(
    SpectralSynthesizer([LineEmulator("emulator_si10827_pca.pt", np.linspace(10824.0, 10833.0, 350))]),
    HeIPCAEmulator.load("emulator_he10830_pca.pt"),
    step=0.02, psf=None, pumping="boundary", telluric=tel_10832, prefix="He_",
)

REGIONS = {
    "fe630": Region.from_file("fe630", "synthetic_fe6301_6302.h5", fe_synth, noise=1e-3, weight=1.0),
    "sihe10830": Region.from_file("sihe10830", "synthetic_si10827_he10830.h5", sihe_synth, noise=1e-3, weight=1.0),
}

# --------------------------------------------------------------------------- #
# Variables. Photosphere: nodes in log tau per variable, bounded by the
# training models at each node's height (training_statistics). He I slab and
# telluric lines: defaults with the emulator's training ranges; pass
# fixed=(...) to keep some at their initial value. Regularization, e.g.
# regularization={"T": [Regularizer("l2", 1.0 / 100.0 ** 2)]}: weights are
# relative to a chi2 per point (each region's chi2 is divided by its number
# of points).
# --------------------------------------------------------------------------- #
stats = training_statistics(MODELS)
nodes = {
    "T": [-4.5, -3.0, -1.5, 0.0],
    "vz": [-3.0, -1.0, 0.5],
    "vmic": [-1.0],
    "Bx": [-2.5, -0.5],
    "By": [-2.5, -0.5],
    "Bz": [-2.5, -0.5],
}
variables = (photosphere_variables(nodes, stats)
             | he_slab_variables("He_", fixed=("beta", "height"))
             | telluric_variables(tel_10832)
             | telluric_variables(o2[0], shift_range=0.05, fixed=("a",))
             | telluric_variables(o2[1], shift_range=0.05, fixed=("a",)))

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("regions", nargs="*", help=f"regions to invert, among {list(REGIONS)} (default: all)")
parser.add_argument("--output", default=None)
parser.add_argument("--iterations", type=int, default=1000)
args = parser.parse_args()
args.regions = args.regions or list(REGIONS)
unknown = [r for r in args.regions if r not in REGIONS]
if unknown:
    parser.error(f"unknown regions {unknown}, choose among {list(REGIONS)}")
regions = [REGIONS[r] for r in args.regions]
output = args.output or f"inversion_multiregion_{'_'.join(args.regions)}.h5"

# keep only the variables the selected regions use
used = {n for r in regions for n in list(r.synthesizer.phys_names) + list(r.synthesizer.slab_names)}
variables = {k: v for k, v in variables.items() if k in used}

config = InversionConfig(
    variables=variables,
    optimizer="adam",
    lr=0.1,
    lr_final_fraction=0.05,
    n_iter=args.iterations,
    chunk_size=1024,
    log_every=50,
)

if NOISE is not None:
    rng = np.random.default_rng(0)
    for r in regions:
        r.stokes = r.stokes + rng.normal(0.0, NOISE, r.stokes.shape)

# --------------------------------------------------------------------------- #
# Initial guess. Photosphere from the first selected region with a
# weak-field configuration below (centre-of-gravity velocity, weak-field B,
# median T and vmic); He I slab and telluric lines from their own regions.
# --------------------------------------------------------------------------- #
g_l, g_u = lande_factor(J=2, L=1, S=2), lande_factor(J=2, L=2, S=2)
fe6301 = dict(lambda0=LAMBDA_6301, geff=effective_lande(g_u, g_l, 2, 2), Geff=second_order_lande(g_u, g_l, 2, 2))
g_l, g_u = lande_factor(J=1, L=1, S=2), lande_factor(J=0, L=2, S=2)
fe6302 = dict(lambda0=LAMBDA_6302, geff=effective_lande(g_u, g_l, 0, 1), Geff=second_order_lande(g_u, g_l, 0, 1))
g_si = lande_factor(J=2, L=1, S=1)
photosphere_guess = {
    "fe630": dict(wfa=WeakFieldConfig(lines=[WeakFieldLine(**fe6301, wave_range=(LAMBDA_6301 - 0.3, LAMBDA_6301 + 0.3)),
                                             WeakFieldLine(**fe6302, wave_range=(LAMBDA_6302 - 0.3, LAMBDA_6302 + 0.25))],
                                      linear_polarization_sign=-1.0),
                  cog_lines=[(LAMBDA_6301, 0.25), (LAMBDA_6302, 0.2)]),
    "sihe10830": dict(wfa=WeakFieldConfig(lines=[WeakFieldLine(lambda0=LAMBDA_SI, geff=effective_lande(g_si, g_si, 2, 2),
                                                               Geff=second_order_lande(g_si, g_si, 2, 2),
                                                               wave_range=(LAMBDA_SI - 0.5, LAMBDA_SI + 0.5))],
                                          linear_polarization_sign=-1.0),
                      cog_lines=[(LAMBDA_SI, 0.25)]),
}
first = next(r for r in regions if r.name in photosphere_guess)
init = initial_guess(first.stokes, first.wavelength, config, stats.logtau, stats.median, **photosphere_guess[first.name])
if "sihe10830" in args.regions:
    r = REGIONS["sihe10830"]
    init |= he_slab_initial_guess(r.stokes, r.wavelength, r.mu, prefix="He_", height=3.0)
    init |= telluric_initial_guess(r.stokes, r.wavelength, tel_10832, continuum=(10831.3, 10832.5))
if "fe630" in args.regions:
    r = REGIONS["fe630"]
    for line in o2:
        init |= telluric_initial_guess(r.stokes, r.wavelength, line, search=0.05, continuum=(6302.15, 6302.30),
                                       width=0.03, a=0.1)

# --------------------------------------------------------------------------- #
# Inversion
# --------------------------------------------------------------------------- #
for r in regions:
    print(f"{r.name}: observations {r.stokes.shape}, {len(r.wavelength)} wavelengths "
          f"[{r.wavelength.min():.3f}, {r.wavelength.max():.3f}] A")
result = MultiRegionInverter(regions, config, stats.logtau).run(init)
save_multiregion_result(output, result)
print(f"saved {output}; median chi2 per point " +
      ", ".join(f"{k} {np.median(v.chi2):.3e}" for k, v in result.regions.items()))
