"""Invert Fe I 6301.5 + 6302.5 Stokes observations with the two PCA
emulators (emulator_fe6301_pca_resmlp512_cosine_deriv.pt,
emulator_fe6302_pca_resmlp512_cosine_deriv.pt: residual MLP with depth-derivative
inputs, trained with fe6301.py / fe6302.py).

Observations are read from an HDF5 file with datasets

    stokes      (nx, ny, 4, nl)   I, Q, U, V in the same units as the Hazel synthesis
    wavelength  (nl,)             absolute wavelengths in A
    mu          scalar or (nx, ny)

and the result is written to another HDF5 file (see
mlp_bspline.inversion.save_result). Edit the configuration below.
"""
import argparse

import h5py
import numpy as np

from modules.config import WeakFieldConfig, WeakFieldLine
from modules.inversion import (
    InversionConfig, Inverter, LineEmulator, SpectralSynthesizer, VariableConfig, initial_guess,
    interpolate_in_depth, save_result,
)
from modules.regularization import Regularizer, TransverseRegularizer
from modules.spatial_psf import SpatialPSF, load_psf
from modules.weak_field import effective_lande, lande_factor, second_order_lande

LAMBDA_6301 = 6301.5012
LAMBDA_6302 = 6302.4936
MODELS = 'models/models_validation.h5'


# --------------------------------------------------------------------------- #
# Emulators and their wavelength grids: the Hazel synthesis grid
# linspace(6300.8921, 6303.2671, 150), split as in fe6301.py / fe6302.py.
# --------------------------------------------------------------------------- #
wave_syn = np.linspace(6300.8921, 6303.2671, 150)
lines = [
    LineEmulator("weights/emulator_fe6301_pca_resmlp512_cosine_deriv.pt", wave_syn[wave_syn - LAMBDA_6301 < 0.5]),
    LineEmulator("weights/emulator_fe6302_pca_resmlp512_cosine_deriv.pt", wave_syn[wave_syn - LAMBDA_6302 > -0.5]),
]
# Instrumental profile: None, {"fwhm": 0.03} (Gaussian, A) or (dlambda, profile) arrays
psf = None
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
# quadratically. With ~1700 points per pixel (4 Stokes x 421 wavelengths),
# chi2 fluctuates by ~sqrt(2 * 1700) ~ 60 per pixel from noise alone, so
# REG_STRENGTH ~ 10 is a mild regularization and ~100 a strong one. Raise
# REG_STRENGTH to smooth everything more, or reduce the `jump` of a single
# variable to smooth that one more. A list of weights (one per node, 0 to
# switch a node off) regularizes the heights of a depth variable differently.
# Regularization of fixed variables (He_beta, He_height) is ignored.
# --------------------------------------------------------------------------- #
REG_STRENGTH = 5.0


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
# --------------------------------------------------------------------------- #
nodes = {
    "T": [-7.5, -4.0, -2.5, -1.0, 0.0, 1.5],
    "vz": [-7.5, 1.5],
    "vmic": [-1.0],
    "Bx": [-7.5, 1.5],
    "By": [-7.5, 1.5],
    "Bz": [-7.5, 1.5],
}
variables = {
    "T": VariableConfig(nodes["T"], scale=500.0, bounds=training_bounds("T", nodes["T"]), regularization=smooth(20.0)),
    # absolute velocities (the Doppler shift of the whole line is included)
    "vz": VariableConfig(nodes["vz"], scale=1.0, bounds=(-20.0, 20.0), regularization=smooth(0.1)),
    "vmic": VariableConfig(nodes["vmic"], scale=0.5, bounds=(0.0, 3.4), regularization=smooth(0.1)),
    "Bx": VariableConfig(nodes["Bx"], scale=200.0, bounds=training_bounds("Bx", nodes["Bx"])),
    "By": VariableConfig(nodes["By"], scale=200.0, bounds=training_bounds("By", nodes["By"])),
    "Bz": VariableConfig(nodes["Bz"], scale=200.0, bounds=training_bounds("Bz", nodes["Bz"]), regularization=smooth(100.)),
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
    noise={"I": 7.2e-3, "Q": 1.6e-3, "U": 1.5e-3, "V": 2.9e-3},
    optimizer="adam",
    lr=0.1,
    lr_final_fraction=0.05,
    n_iter=500,
    chunk_size=65536*3,
    log_every=5,
)

# --------------------------------------------------------------------------- #
# Observations, initial guess (weak-field B, centre-of-gravity velocity, mean
# T and vmic) and inversion
# --------------------------------------------------------------------------- #
# with h5py.File(args.observations, 'r') as f:
    # observations = f['stokes'][:]
    # wave_obs = f['wavelength'][:].astype(np.float64)
    # mu = f['mu'][()]

with h5py.File('/scratch1/hinode/ar11967/ar11967.h5', 'r') as f:
    observations = f['stokes'][:]

contQS = np.median(observations[0, 0:100, 300:400, 0])

print(f"contQS {contQS:.3e} (median of I in QS region)")

observations = np.transpose(observations, (1, 2, 0, 3))[0:400, 0:400, :, :] / contQS


wave_obs = np.loadtxt('wavelength_Hinode.txt', dtype=np.float64)
mu = np.ones(observations.shape[0])

print(f"observations {observations.shape}, {len(wave_obs)} wavelengths "
      f"[{wave_obs.min():.3f}, {wave_obs.max():.3f}] A")

g_l, g_u = lande_factor(J=2, L=1, S=2), lande_factor(J=2, L=2, S=2)
line_6301 = dict(lambda0=LAMBDA_6301, geff=effective_lande(g_u, g_l, 2, 2), Geff=second_order_lande(g_u, g_l, 2, 2))
g_l, g_u = lande_factor(J=1, L=1, S=2), lande_factor(J=0, L=2, S=2)
line_6302 = dict(lambda0=LAMBDA_6302, geff=effective_lande(g_u, g_l, 0, 1), Geff=second_order_lande(g_u, g_l, 0, 1))
split = 0.5 * (LAMBDA_6301 + LAMBDA_6302)
wfa = WeakFieldConfig(lines=[WeakFieldLine(**line_6301, wave_range=(0.0, split)),
                             WeakFieldLine(**line_6302, wave_range=(split + 1e-4, 1e5))],
                      linear_polarization_sign=-1.0)
init = initial_guess(observations, wave_obs, config, logtau, mean_strat, wfa=wfa,
                     cog_lines=[(LAMBDA_6301, 0.25), (LAMBDA_6302, 0.25)])

# Spatial PSF of the telescope (on the Hinode pixel grid, peak in the central
# pixel): the synthetic maps are convolved with it before the comparison with
# the observations. Set to None to invert pixel by pixel without it. Cropped
# to PSF_SIZE pixels (enclosed energy: 31 -> 95%, 63 -> 98%, 257 -> 100%).
PSF_FILE = 'psfs/hinode_psf_size_256_def_-0.32_hinodepix.fits'
PSF_FILE = None
# PSF_SIZE = 63
spatial_psf = None if PSF_FILE is None else SpatialPSF(load_psf(PSF_FILE), size=PSF_SIZE)

if PSF_FILE is None:
    output = 'inversions/inversion_result_fe6301_6302_nopsf.h5'
else:
    output = f'inversions/inversion_result_fe6301_6302_psf.h5'

inverter = Inverter(synthesizer, config, logtau, spatial_psf=spatial_psf)
result = inverter.run(observations, wave_obs, mu, init)
save_result(output, result, wave_obs, observations)
print(f"saved {output}; median chi2 per point {np.median(result.chi2):.3e}")
