import numpy as np
import h5py
from modules.config import ChannelPCAConfig, ChannelSplineConfig, ScalerConfig, StokesTransformConfig, WeakFieldConfig, WeakFieldLine, DepthFeatureConfig
from modules.bspline import clamped_knots_from_positions
from modules.train import TrainConfig, train_emulator
from modules.weak_field import effective_lande, lande_factor, second_order_lande

# What the model predicts for the Stokes profiles: "bspline" coefficients or
# "pca" coefficients (principal components fit on the training profiles).
STOKES_BASIS = "pca"

# Rest wavelengths of the doublet (SIR LINEAS, lines 200 and 201)
LAMBDA_6301 = 6301.5012
LAMBDA_6302 = 6302.4936

print("Reading models...")
f = h5py.File('/scratch/Dropbox/GIT/DeepLearning/universal_stokes_flow/database/models/models_training.h5', 'r')
models = f['model'][:]
f.close()

# Remove velocity at log(tau)=-1
print("Removing velocity at log(tau)=-1")
models[:, :, 4] -= models[:, 22:23, 4]

ind = [1, 4, 3, 5, 6, 7]
depth = models[0, :, 0]
models = np.transpose(models, (0, 2, 1))[:, ind, :]
n_models = models.shape[0]

# The Fe 6301 database stores every model at the same 5 heliocentric angles:
# stokes is [n_mu, N, 4, L]. To keep one sample per model (as in the Si
# 10827 database, where each model has its own random mu), pick one of the
# 5 mu values at random for each model. Each mu slice is read whole (~1.8 GB)
# rather than fancy-indexed, which is much faster with h5py.
print("Reading Stokes...")
f = h5py.File('/scratch/Dropbox/GIT/DeepLearning/universal_stokes_flow/database/Fe_6301/stokes_training_randommu.h5', 'r')
mu = f['mu'][:]
wave = f['wavelength'][:].astype(np.float64) - LAMBDA_6301
stokes = f['stokes'][:]
f.close()

print("Restricting range to 6301 A line")
ind = np.where(wave < 0.5)[0]
wave = wave[ind]
stokes = stokes[:, :, ind]

# physical: [N, 6, D] in channel order T, vz, vmic, Bx, By, Bz
# stokes:   [N, 4, L] in channel order I, Q, U, V
# mu:       [N]
# depth, wave: [D], [L] grids

phys_cfg = {
    "T":  ChannelSplineConfig(num_coeffs=10, degree=3),
    "vz": ChannelSplineConfig(num_coeffs=10, degree=3),
    "vmic": ChannelSplineConfig(num_coeffs=10, degree=3),
    "Bx": ChannelSplineConfig(num_coeffs=10, degree=3),
    "By": ChannelSplineConfig(num_coeffs=10, degree=3),
    "Bz": ChannelSplineConfig(num_coeffs=10, degree=3),
}

# Sampling positions for the interior knots of the Stokes B-splines. The
# wavelength axis is relative to 6301.5012 A, so the two line centers are at
# 0 and +0.992 A. The grid (~0.0215 A step, -0.61 to +1.77 A) is already
# fairly coarse, so knots go on every other grid point within +-0.35 A of
# each line center and are spaced more coarsely in the wings and in the
# continuum between the two lines.
degree = 3

# Coarse knots in the far wings
left = np.linspace(wave.min() + 0.05, -0.22, 5)
right = np.linspace(0.22, wave.max() - 0.05, 5)

# Use the original sampling of the training data in the line core between -0.2 and +0.2 A
idx1 = np.argmin(np.abs(wave + 0.2))
idx2 = np.argmin(np.abs(wave - 0.2))
center = wave[idx1:idx2+1]

positions = np.concatenate([left, center, right])  # interior knots, dense near line center

knots = clamped_knots_from_positions(wave.min(), wave.max(), positions, degree)

stokes_cfg = {
    "I": ChannelSplineConfig(num_coeffs=len(positions) + degree + 1, degree=degree, knots=knots),
    "Q": ChannelSplineConfig(num_coeffs=len(positions) + degree + 1, degree=degree, knots=knots),
    "U": ChannelSplineConfig(num_coeffs=len(positions) + degree + 1, degree=degree, knots=knots),
    "V": ChannelSplineConfig(num_coeffs=len(positions) + degree + 1, degree=degree, knots=knots),
}
print(f"{len(wave)} wavelength points, {len(positions) + degree + 1} B-spline coefficients per Stokes parameter")

# PCA basis, used when STOKES_BASIS == "pca". It is fit on the training
# split of what is actually fit (WFA residual + symlog transform below), per
# Stokes parameter. On the validation set, 99.999% of the variance takes
# ~20 (I), 27 (Q), 29 (U) and 32 (V) components, which reconstructs the
# held-out physical profiles to ~1e-4 rms, similar to the B-splines above
# with fewer outputs (~108 vs 172).
stokes_pca_cfg = {
    "I": ChannelPCAConfig(variance_fraction=0.99999),
    "Q": ChannelPCAConfig(variance_fraction=0.99999),
    "U": ChannelPCAConfig(variance_fraction=0.99999),
    "V": ChannelPCAConfig(variance_fraction=0.99999),
}

# Fit I directly, but Q/I, U/I, V/I with a symlog compression (see si10827.py).
stokes_transform = StokesTransformConfig(amplitude=30.0)

# Fit Q, U, V as residuals with respect to the weak-field approximation
# built from Stokes I and the depth-averaged Bx, By, Bz (I is still fit
# directly). The two lines have different Lande factors, so each one gets
# its own WFA, applied only in its own wavelength window:
#
# Fe I 6301.5012 A is 5P2 - 5D2 (SIR line 200): g_l = 1.833, g_u = 1.5,
#   gbar = 1.667, Gbar = 2.517.
# Fe I 6302.4936 A is 5P1 - 5D0 (SIR line 201): J_u = 0, a normal Zeeman
#   triplet, gbar = 2.5, Gbar = gbar^2 = 6.25.
g_l, g_u = lande_factor(J=2, L=1, S=2), lande_factor(J=2, L=2, S=2)
geff_6301 = effective_lande(g_u, g_l, J_u=2, J_l=2)
Geff_6301 = second_order_lande(g_u, g_l, J_u=2, J_l=2)

g_l, g_u = lande_factor(J=1, L=1, S=2), lande_factor(J=0, L=2, S=2)
geff_6302 = effective_lande(g_u, g_l, J_u=0, J_l=1)
Geff_6302 = second_order_lande(g_u, g_l, J_u=0, J_l=1)
print(f"Fe I 6301: gbar = {geff_6301:.4f}, Gbar = {Geff_6301:.4f}")
#print(f"Fe I 6302: gbar = {geff_6302:.4f}, Gbar = {Geff_6302:.4f}")

# The windows split the grid in the continuum between the lines, at +0.5 A
# (6302.0 A), where dI/dl is ~0.4% of its peak, so the baseline of each line
# is essentially zero when it is cut off. The boundary lies between two
# grid points (0.482 and 0.504 A); the second window starts just after it
# since windows are inclusive and may not overlap.
split = 0.5

# Bx, By, Bz are in the line-of-sight frame and the Q/U sign follows SIR's
# convention, as in si10827.py (the same Hazel photosphere synthesized both).
weak_field = WeakFieldConfig(
    lambda0=LAMBDA_6301,
    geff=geff_6301,
    Geff=Geff_6301,
    linear_polarization_sign=-1.0,
    depth_range=None,  # average over all depth points
)

weak_field = None


# Q/U/V are symlog-ratio values (not physical amplitudes), so let
# per-coefficient standard scaling adapt to whatever range the transform
# produces.
#
# For PCA, a single scale per channel (per_coeff=False) keeps the loss
# proportional to the profile-space MSE (the basis is orthonormal), instead
# of giving the trailing, low-variance components the same weight as the
# leading ones.
per_coeff = STOKES_BASIS != "pca"
stokes_scaler_cfg = {
    "I": ScalerConfig(kind="standard", per_coeff=per_coeff),
    "Q": ScalerConfig(kind="standard", per_coeff=per_coeff),
    "U": ScalerConfig(kind="standard", per_coeff=per_coeff),
    "V": ScalerConfig(kind="standard", per_coeff=per_coeff),
}

phys_scaler_cfg = {
    "T": ScalerConfig(kind="fixed", per_coeff=False, scale=5000.0, offset=0.0),
    "vz": ScalerConfig(kind="fixed", per_coeff=False, scale=2.0, offset=0.0),
    "vmic": ScalerConfig(kind="fixed", per_coeff=False, scale=2.0, offset=0.0),
    "Bx": ScalerConfig(kind="fixed", per_coeff=False, scale=5e2, offset=0.0),
    "By": ScalerConfig(kind="fixed", per_coeff=False, scale=5e2, offset=0.0),
    "Bz": ScalerConfig(kind="fixed", per_coeff=False, scale=5e2, offset=0.0),
}

# Add the depth derivatives of every physical channel as extra inputs (as in
# si10827.py): the line formation depends on the gradients of the
# stratification, which the MLP otherwise has to infer from the coefficients.
depth_features = DepthFeatureConfig(derivatives=["T", "vz", "vmic", "Bx", "By", "Bz"], mode="diff")

result = train_emulator(
    models,
    stokes,
    mu,
    depth,
    wave,
    phys_cfg,
    stokes_cfg,
    phys_scaler_configs=phys_scaler_cfg,
    stokes_scaler_configs=stokes_scaler_cfg,
    stokes_transform=stokes_transform,
    weak_field=weak_field,
    train_cfg=TrainConfig(epochs=300,
                          batch_size=1024,
                          hidden_dims=(256, 256, 256, 256),
                          lr=1e-3,
                          model_type='resmlp',
                          model_kwargs=dict(width=512, num_blocks=6, expansion=2),
                          optimizer='adamw',
                          weight_decay=1e-5,
                          scheduler='cosine',
                          patience=None,
                          log_every=1,
                          plot_every=0,
                          stokes_loss_weights={"I": 1.0, "Q": 1.0, "U": 1.0, "V": 1.0}),
    checkpoint_path="weights/emulator_fe6301.pt" if STOKES_BASIS == "bspline" else "weights/emulator_fe6301_pca_resmlp512_cosine_deriv.pt",
    stokes_basis=STOKES_BASIS,
    stokes_pca_configs=stokes_pca_cfg,
    depth_features=depth_features,
)
