"""Invert Ca II 8542 Stokes observations from SST/CRISP with the PCA emulator
(emulator_caii8542_pca_resmlp512_cosine_deriv.pt: residual MLP with
depth-derivative inputs, trained with caii8542.py).

Observations are one scan of a CRISP cube (numpy axes y, x, Stokes, scan,
wavelength; intensities in W m^-2 Hz^-1 sr^-1), with 15 wavelengths and the
wavelengths and coordinates in the WCS-TAB extension. The result is written
to another HDF5 file (see mlp_bspline.inversion.save_result). Edit the
configuration below.
"""
import h5py
import numpy as np
from astropy.io import fits

from modules.config import WeakFieldConfig, WeakFieldLine
from modules.inversion import (
    InversionConfig, Inverter, LineEmulator, SpectralSynthesizer, VariableConfig, initial_guess,
    interpolate_in_depth, save_result,
)
from modules.regularization import Regularizer, TransverseRegularizer
from modules.weak_field import effective_lande, lande_factor, second_order_lande
import pyFPI

LAMBDA_8542 = 8542.09
MODELS = 'models/models_validation.h5'

OBSERVATIONS = "/scratch1/datasets/sst/nb_8542_2023-06-29T09:05:41_09:05:41=0-21_stokes_corrected_export2026-06-11T09:30:36_im.fits"
# /scratch1/crisp/nb_8542_2020-07-27T08:35:09_scans=0-10_stokes_corrected_sp.fits'
SCAN = 3
# Field of view, in pixels of the cube. This is the region of
# /scratch1/crisp/README.txt, also inverted there with STiC and Hazel. Use
# (0, 926) and (0, 945) for the whole field of view.
# X_RANGE = (500, 1100)
# Y_RANGE = (500, 1100)
X_RANGE = (500, 1500)
Y_RANGE = (250, 1600)
RSUN_ARCSEC = 945.0     # apparent solar radius on 2020-07-27

# The emulator works in units of I_NORM = 2.59e-5 erg s^-1 cm^-2 Hz^-1 sr^-1
# (caii8542.py), i.e. absolute intensities, not a fraction of the continuum.
# The cube is calibrated in absolute units too.
I_NORM_SI = 2.59e-8     # W m^-2 Hz^-1 sr^-1

# --------------------------------------------------------------------------- #
# Emulator and its wavelength grid: the first 150 wavelengths of the database
# (see caii8542.py).
# --------------------------------------------------------------------------- #
wave_syn = 8539.8357 + 0.03 * np.arange(150)
lines = [LineEmulator("weights/emulator_caii8542_pca_resmlp512_cosine_deriv.pt", wave_syn)]
# Instrumental profile: None, {"fwhm": 0.03} (Gaussian, A) or (dlambda, profile) arrays

fpi_8542 = pyFPI.CRISP(8542.0, version=1, nrays_hr=11, nrays_lr=11, verbose=False)
tr_8542 = fpi_8542.dual_fpi_full_complex(wave_syn - 8542.0, ech=-0.0 / 787e4 * 8542.0, erh=-0.008, normalize_tr=True)

psf = (wave_syn - LAMBDA_8542, tr_8542)

synthesizer = SpectralSynthesizer(lines, psf=psf)



# --------------------------------------------------------------------------- #
# Depth grid (from the deepest point upwards, as in the models), median
# stratification and bounds from the training models (same
# preprocessing as in training: velocity at log(tau) = -1 removed).
# --------------------------------------------------------------------------- #
with h5py.File(MODELS, 'r') as f:
    models = f['model'][:]
models[:, :, 4] -= models[:, 22:23, 4]
logtau = models[0, :, 0].astype(np.float64)
columns = {"T": 1, "vz": 4, "vmic": 3, "Bx": 5, "By": 6, "Bz": 7}
mean_strat = {k: np.median(models[:, :, c], axis=0) for k, c in columns.items()}
lo_strat = {k: np.percentile(models[:, :, c], 0.1, axis=0) for k, c in columns.items()}
hi_strat = {k: np.percentile(models[:, :, c], 99.9, axis=0) for k, c in columns.items()}

# Avoid 180 deg ambiguity
lo_strat['Bx'] *= 0


def training_bounds(name, nodes):
    """Per-node bounds: the 0.1-99.9 percentile range of the training
    models at each node's height, so the emulators never extrapolate."""
    return interpolate_in_depth(nodes, logtau, lo_strat[name]), interpolate_in_depth(nodes, logtau, hi_strat[name])

# --------------------------------------------------------------------------- #
# Regularization. Regularizer("l2", w) adds to the objective
#
#     w * sum_pixels |grad x|^2      (forward differences, physical units)
#
# for every node map x of a variable, next to chi2 = sum_pixels sum_points
# ((synthetic - observed) / noise)^2. A natural way to choose w is
#
#     w = REG_STRENGTH / jump^2
#
# where `jump` is the difference between neighbouring pixels that you
# consider typical (in the units of the variable): such a jump then costs
# REG_STRENGTH units of chi2, while larger ones are penalized
# quadratically. With 60 points per pixel (4 Stokes x 15 wavelengths), chi2
# fluctuates by ~sqrt(2 * 60) ~ 11 per pixel from noise alone. With the
# jumps below and these few points, REG_STRENGTH = 5 (the Fe I value) wipes
# out the penumbral filaments (median chi2 per point 2.1), 0.5 keeps them
# (1.1) and 0.05 starts fitting the noise in B (0.93). Raise
# REG_STRENGTH to smooth everything more, or reduce the `jump` of a single
# variable to smooth that one more. A list of weights (one per node, 0 to
# switch a node off) regularizes the heights of a depth variable differently.
# --------------------------------------------------------------------------- #
REG_STRENGTH = 0.5


def smooth(jump):
    """L2 smoothness regularization tolerating neighbour differences of
    ~`jump` (scalar, or one value per node), in physical units."""
    return [Regularizer("l2", REG_STRENGTH / np.asarray(jump, dtype=np.float64) ** 2)]

# --------------------------------------------------------------------------- #
# Nodes, scales, bounds and spatial regularization per variable. Weights are
# in the physical units of each variable and are relative to chi2 (i.e. to
# the noise levels below). Available: Regularizer("l2" | "tv" | "l1" |
# "wavelet_l1", weight, options); e.g.
#   Regularizer("tv", 1e-3), Regularizer("l2", 1e-6),
#   Regularizer("wavelet_l1", 1e-3, {"wavelet": "db2", "levels": 3})
# Ca II 8542 forms from the photosphere (wings) to log(tau) ~ -5.5 (core), so
# T, vz and vmic have nodes in the chromosphere too.
# --------------------------------------------------------------------------- #
nodes = {
    "T": [-7.5, -5.5, -4.0, -2.5, -1.0, 0.0, 1.5],
    "vz": [-7.5, -4.0, 1.5],
    "vmic": [-7.5, 1.5],
    "Bx": [-7.5, 1.5],
    "By": [-7.5, 1.5],
    "Bz": [-7.5, 1.5],
}
variables = {
    "T": VariableConfig(nodes["T"], scale=500.0, bounds=training_bounds("T", nodes["T"]), regularization=smooth(100.)),
    # absolute velocities (the Doppler shift of the whole line is included)
    "vz": VariableConfig(nodes["vz"], scale=1.0, bounds=(-20.0, 20.0)),
    "vmic": VariableConfig(nodes["vmic"], scale=0.5, bounds=(0.0, 3.4)),
    "Bx": VariableConfig(nodes["Bx"], scale=200.0, bounds=training_bounds("Bx", nodes["Bx"])),
    "By": VariableConfig(nodes["By"], scale=200.0, bounds=training_bounds("By", nodes["By"])),
    "Bz": VariableConfig(nodes["Bz"], scale=200.0, bounds=training_bounds("Bz", nodes["Bz"]), regularization=smooth(200.)),
}

# Transverse field: with Bx >= 0 the solution jumps from (0, Bt) to (0, -Bt)
# where the azimuth crosses +-90 deg. Smoothing Bx and By separately turns
# that jump into a spurious Bt ~ 0 line, so (Bx, By) are smoothed through
# the ambiguity-invariant Bt (cos 2phi, sin 2phi) instead.
transverse_regularization = [TransverseRegularizer("Bx", "By", smooth(100.0))]

config = InversionConfig(
    variables=variables,
    transverse_regularization=transverse_regularization,
    stokes_weights={"I": 1.0, "Q": 1.0, "U": 1.0, "V": 1.0},
    # Q, U, V: standard deviation at the outermost wavelengths in a quiet
    # region, in units of I_NORM_SI. I: of the order of the emulator error.
    noise={"I": 1e-2, "Q": 2.5e-3, "U": 2e-3, "V": 2e-3},
    optimizer="adam",
    lr=0.1,
    lr_final_fraction=0.05,
    n_iter=500,
    chunk_size=65536*4,
    log_every=5,
)

# --------------------------------------------------------------------------- #
# Observations, initial guess (weak-field B, centre-of-gravity velocity, mean
# T and vmic) and inversion
# --------------------------------------------------------------------------- #
(x0, x1), (y0, y1) = X_RANGE, Y_RANGE
with fits.open(OBSERVATIONS, memmap=True) as hdu:
    observations = np.array(hdu[0].data[SCAN, :, :, y0:y1, x0:x1], dtype=np.float64)    
    n_y, n_x = hdu[0].data.shape[-2:]
    # WCS-TAB coordinates (scan, tuning, y corner, x corner, [hpln, hplt, wave, time])
    wcs = hdu['WCS-TAB'].data[0][0]

observations = np.transpose(observations, (2, 3, 0, 1)) / I_NORM_SI

wave_obs = wcs[SCAN, :, 0, 0, 2] * 10.0
hpln = np.linspace(wcs[SCAN, 0, 0, 0, 0], wcs[SCAN, 0, 0, 1, 0], n_x)[x0:x1]
hplt = np.linspace(wcs[SCAN, 0, 0, 0, 1], wcs[SCAN, 0, 1, 0, 1], n_y)[y0:y1]
mu = np.sqrt(1.0 - (hpln[:, None] ** 2 + hplt[None, :] ** 2) / RSUN_ARCSEC ** 2)
mu = mu.T

print(f"observations {observations.shape}, {len(wave_obs)} wavelengths "
      f"[{wave_obs.min():.3f}, {wave_obs.max():.3f}] A, mu [{mu.min():.3f}, {mu.max():.3f}]")

# 3d 2D5/2 - 4p 2P3/2
g_l, g_u = lande_factor(J=2.5, L=2, S=0.5), lande_factor(J=1.5, L=1, S=0.5)
line_8542 = dict(lambda0=LAMBDA_8542, geff=effective_lande(g_u, g_l, 1.5, 2.5),
                 Geff=second_order_lande(g_u, g_l, 1.5, 2.5))
wfa = WeakFieldConfig(lines=[WeakFieldLine(**line_8542, wave_range=(LAMBDA_8542 - 0.6, LAMBDA_8542 + 0.6))],
                      linear_polarization_sign=-1.0)
init = initial_guess(observations, wave_obs, config, logtau, mean_strat, wfa=wfa,
                     cog_lines=[(LAMBDA_8542, 0.6)])

output = 'inversions/inversion_result_caii8542.h5'

inverter = Inverter(synthesizer, config, logtau)
result = inverter.run(observations, wave_obs, mu, init)
save_result(output, result, wave_obs, observations)
print(f"saved {output}; median chi2 per point {np.median(result.chi2):.3e}")
