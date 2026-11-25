"""Invert Si I 10827 + He I 10830 Stokes observations: the Si I PCA emulator
(emulator_si10827_pca.pt) for the photosphere and a constant-property He I
slab (the PCA emulator emulator_he10830_pca.pt, trained with he10830.py, +
the exact slab formal solution, as in Hazel2) for the chromosphere, with
the Si I spectrum as the slab's boundary condition, and a telluric Voigt
absorption line to the red of He I in Stokes I. All photospheric node values, He I slab
parameters and telluric line parameters are inverted jointly, with an L2
(Tikhonov) spatial smoothness regularization of the maps of every inverted
quantity (see "Regularization" below).

Observations are read from an HDF5 file with datasets

    stokes      (nx, ny, 4, nl)   I, Q, U, V in units of the quiet-Sun disk-centre continuum
    wavelength  (nl,)             absolute wavelengths in A
    mu          scalar or (nx, ny)

(e.g. the output of make_synthetic_si10827_he10830.py) and the result is
written to another HDF5 file (see mlp_bspline.inversion.save_result; the He
I parameters are in nodes/He_*, the telluric ones in nodes/Tel_*, with
lambda0 = TELLURIC + Tel_shift). Edit the configuration below.
"""
import h5py
import numpy as np

from modules.config import WeakFieldConfig, WeakFieldLine
from modules.hei import SiHeSynthesizer, he_slabs_initial_guess
from modules.hei_pca import HeIPCAEmulator
from modules.inversion import (
    InversionConfig, Inverter, LineEmulator, SpectralSynthesizer, VariableConfig,
    initial_guess, interpolate_in_depth, save_result, slab_variable
)
from modules.regularization import Regularizer, TransverseRegularizer
from modules.weak_field import effective_lande, lande_factor, second_order_lande

LAMBDA_SI = 10827.089
LAMBDA_HE_RED = 10830.30  # centre of the (blended) red components of the triplet
MODELS = '/scratch/Dropbox/GIT/DeepLearning/universal_stokes_flow/database/models/models_validation.h5'

observations_file = 'observations/stokesall_15Jul24ARM1_009_norm_xtalk.h5'
output = 'inversions/inversion_result_si10827_he10830_l1.h5'
noise_level = 1e-3      # added to the (noiseless) synthetic observations; set to None for real data
HE_HEIGHT = 3.0         # arcsec, fixed
# He I slabs, from the Si I photosphere outwards (variables <prefix>Bx, ...):
# ("He_",) for one slab, e.g. ("He1_", "He2_") for two in series
HE_PREFIXES = ("He1_", "He2_")
TELLURIC = 10832.0      # A, reference wavelength of the telluric line (None: no telluric line)

# --------------------------------------------------------------------------- #
# Forward model. The Si I emulator was trained on the Hazel grid
# linspace(10824, 10833, 150); the He I slab is added on a uniform 0.02 A
# grid over the same range, with the Si I spectrum (Doppler shifted) as
# boundary condition. The slab's eps, eta, rho come from the PCA emulator
# (one MLP call per pixel, profiles on Hazel's 256-point grid interpolated to
# the Doppler-shifted synthesis grid; the old per-wavelength neural field
# HeIEmulator("hei/2025-10-23-17_21_32.best.pth", "hei/input_normalization.npz")
# is a drop-in alternative). The pumping radiation of the slab follows the
# photospheric continuum, as in Hazel2 (pumping="boundary").
# --------------------------------------------------------------------------- #
wave_si = np.linspace(10824.0, 10833.0, 350)
photosphere = SpectralSynthesizer([LineEmulator("weights/emulator_si10827_pca_resmlp512_cosine_deriv.pt", wave_si)])
chromosphere = HeIPCAEmulator.load("weights/emulator_he10830_pca.pt")
# Instrumental profile: None, {"fwhm": 0.03} (Gaussian, A) or (dlambda, profile) arrays
psf = None
synthesizer = SiHeSynthesizer(photosphere, chromosphere, step=0.02, psf=psf, pumping="boundary", telluric=TELLURIC,
                              prefix=HE_PREFIXES)

# --------------------------------------------------------------------------- #
# Depth grid, median stratification and bounds from the training models (as
# in invert_fe6301_6302.py).
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
# Photospheric nodes and He I slab parameters. The slab field is in Hazel's
# vertical (local solar) frame; bounds keep the slab inside the emulator's
# training ranges (deltav 4-18 km/s, a 0-1.5, B components ~ N(0, 500 G)).
# He_beta (source-function factor) and He_height are fixed; make them free
# with fixed=False. Regularization as in invert_fe6301_6302.py, e.g.
# regularization=[Regularizer("tv", 1e-3)].
# --------------------------------------------------------------------------- #
nodes = {
    "T": [-7.5, -4.0, -2.5, -1.0, 0.0, 1.5],
    "vz": [-7.5, 1.5],
    "vmic": [-1.0],
    "Bx": [-1.0],
    "By": [-1.0],
    "Bz": [-7.5, 1.5],
}

variables = {
    "T": VariableConfig(nodes["T"], scale=500.0, bounds=training_bounds("T", nodes["T"]),
                        regularization=smooth(100.0)),                       # K
    # absolute velocities (the Doppler shift of the whole line is included)
    "vz": VariableConfig(nodes["vz"], scale=1.0, bounds=(-20.0, 20.0),
                         regularization=smooth(0.5)),                        # km/s
    "vmic": VariableConfig(nodes["vmic"], scale=0.5, bounds=(0.0, 3.4),
                           regularization=smooth(0.2)),                      # km/s
    # Bx, By: regularized jointly below (transverse_regularization)
    "Bx": VariableConfig(nodes["Bx"], scale=200.0, bounds=training_bounds("Bx", nodes["Bx"])),  # G
    "By": VariableConfig(nodes["By"], scale=200.0, bounds=training_bounds("By", nodes["By"])),  # G
    # one jump per node: the upper node (log tau = -2) is smoothed more
    "Bz": VariableConfig(nodes["Bz"], scale=200.0, bounds=training_bounds("Bz", nodes["Bz"]),
                         regularization=smooth(50.0)),               # G
}
for p in HE_PREFIXES:
    if p == "He1_":
        vleft = -40.0
        vright = 40.0
        regularization_tau = []
    elif p == "He2_":
        vleft = 15.0
        vright = 40.0
        regularization_tau = [Regularizer("l1", 1.0 / 0.03 ** 2)]  # L1 regularization to avoid He2 tau = 0

    variables |= {
        p + "Bx": slab_variable(scale=100.0, bounds=(0.0, 1500.0)),   # G
        p + "By": slab_variable(scale=100.0, bounds=(-1500.0, 1500.0)),   # G
        p + "Bz": slab_variable(scale=100.0, bounds=(-1500.0, 1500.0), regularization=smooth(50.0)),   # G
        p + "tau": slab_variable(scale=0.3, bounds=(0.01, 5.0), regularization=regularization_tau),
        p + "v": slab_variable(scale=1.0, bounds=(vleft, vright), regularization=[]),          # km/s
        p + "deltav": slab_variable(scale=1.0, bounds=(4.0, 10.0), regularization=[]),       # km/s
        p + "a": slab_variable(scale=0.1, bounds=(0.0, 0.2), regularization=[]),
        p + "beta": slab_variable(scale=0.1, bounds=(0.0, 5.0), fixed=True),
        p + "height": slab_variable(scale=0.5, bounds=(0.1, 10.0), fixed=True),
    }
if TELLURIC is not None:
    variables |= {
        "Tel_shift": slab_variable(scale=0.01, bounds=(-0.3, 0.3), regularization=[]),   # A
        "Tel_width": slab_variable(scale=0.01, bounds=(0.005, 0.3), regularization=[]),  # A
        "Tel_depth": slab_variable(scale=0.05, bounds=(0.0, 1.0), regularization=[]),
        "Tel_a": slab_variable(scale=0.1, bounds=(0.0, 2.0), regularization=[]),
    }

# Transverse field: with Bx >= 0 the solution jumps from (0, Bt) to (0, -Bt)
# where the azimuth crosses +-90 deg. Smoothing Bx and By separately turns
# that jump into a spurious Bt ~ 0 line, so (Bx, By) are smoothed through
# the ambiguity-invariant Bt (cos 2phi, sin 2phi) instead.
transverse_regularization = [TransverseRegularizer(p + "Bx", p + "By", smooth(20.0)) for p in ("",) + tuple(HE_PREFIXES)]

config = InversionConfig(
    variables=variables,
    transverse_regularization=transverse_regularization,
    stokes_weights={"I": 1.0, "Q": 2.0, "U": 2.0, "V": 1.0},
    noise={"I": 7.2e-3, "Q": 1.6e-3, "U": 1.5e-3, "V": 2.9e-3},
    optimizer="adam",
    lr=0.1,
    lr_final_fraction=0.05,
    n_iter=500,
    chunk_size=1024,
    log_every=5,
)

# --------------------------------------------------------------------------- #
# Observations
# --------------------------------------------------------------------------- #
with h5py.File(observations_file, 'r') as f:
    observations = f['stokes'][:]

observations = observations.reshape((447, 170, -1, 4))
observations = np.transpose(observations, (0, 1, 3, 2))  # (nx, ny, 4, nl)

observations = observations[:, :, :, :]


wave_obs = np.loadtxt('observations/stokesall_15Jul24ARM1_009_norm.wavelength')
mu_map = np.ones(observations.shape[:2])


nx, ny = observations.shape[:2]

print(f"observations {observations.shape}, {len(wave_obs)} wavelengths "
      f"[{wave_obs.min():.3f}, {wave_obs.max():.3f}] A, mu [{mu_map.min():.2f}, {mu_map.max():.2f}]")

# --------------------------------------------------------------------------- #
# Initial guess
# Photosphere: weak-field B and centre-of-gravity velocity of Si I, mean T and
# vmic. Si I 10827.089 A is 3P2 - 3P2 (SIR line 300): gbar = Gbar^0.5 = 1.5.
# --------------------------------------------------------------------------- #
g_si = lande_factor(J=2, L=1, S=1)
wfa_si = WeakFieldConfig(lines=[WeakFieldLine(lambda0=LAMBDA_SI, geff=effective_lande(g_si, g_si, 2, 2),
                                              Geff=second_order_lande(g_si, g_si, 2, 2),
                                              wave_range=(LAMBDA_SI - 0.5, LAMBDA_SI + 0.5))],
                         linear_polarization_sign=-1.0)
init = initial_guess(observations, wave_obs, config, logtau, mean_strat, wfa=wfa_si,
                     cog_lines=[(LAMBDA_SI, 0.25)])

# init = initial_guess(observations, wave_obs, config, logtau, mean_strat, wfa=None,
                    #  cog_lines=[(LAMBDA_SI, 0.25)])

# He I: velocity from the centre of gravity of the red component, tau from
# its depth relative to the nearby continuum, field from the weak-field
# approximation of the red component (he_slab_initial_guess). Unlike the
# photosphere, the He I Q/U follow the textbook sign convention (+1), and the
# effective Lande factor of the red blend (1.13) was calibrated on Hazel2
# syntheses. The WFA field is in the LOS frame, rotated to the vertical frame
# with mu as Hazel does. With several slabs, tau is divided among them and
# their velocities are spread by +-2 km/s (velocity_offsets=...) so they
# do not start identical.
init |= he_slabs_initial_guess(observations, wave_obs, mu_map, prefixes=HE_PREFIXES, height=HE_HEIGHT)


flat = observations.reshape(nx * ny, 4, -1)
cont = (wave_obs > 10831.3) & (wave_obs < 10832.5)

# Telluric line: position and depth of the I minimum within 0.3 A of TELLURIC
# relative to the nearby continuum (high percentile, so the line itself does
# not lower it); typical width and damping.
if TELLURIC is not None:
    win = np.abs(wave_obs - TELLURIC) <= 0.3
    i_win = np.nan_to_num(flat[:, 0, win], nan=np.inf)
    cont_tel = np.nanpercentile(flat[:, 0, cont], 90, axis=1)
    init["Tel_shift"][:] = (wave_obs[win][i_win.argmin(1)] - TELLURIC).reshape(nx, ny, 1)
    init["Tel_depth"][:] = np.clip(1.0 - i_win.min(1) / cont_tel, 0.02, 0.9).reshape(nx, ny, 1)
    init["Tel_width"][:] = 0.04
    init["Tel_a"][:] = 0.5

# --------------------------------------------------------------------------- #
# Inversion
# --------------------------------------------------------------------------- #
inverter = Inverter(synthesizer, config, logtau)
result = inverter.run(observations, wave_obs, mu_map, init)
save_result(output, result, wave_obs, observations)
print(f"saved {output}; median chi2 per point {np.median(result.chi2):.3e}")
