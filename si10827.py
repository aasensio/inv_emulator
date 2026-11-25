import numpy as np
import h5py
from modules.config import ChannelPCAConfig, ChannelSplineConfig, ScalerConfig, StokesTransformConfig, WeakFieldConfig, DepthFeatureConfig
from modules.bspline import clamped_knots_from_positions
from modules.train import TrainConfig, train_emulator
from modules.weak_field import effective_lande, lande_factor, second_order_lande

# What the model predicts for the Stokes profiles: "bspline" coefficients or
# "pca" coefficients (principal components fit on the training profiles).
STOKES_BASIS = "pca"

# Rest wavelengths of the doublet (SIR LINEAS, lines 200 and 201)
LAMBDA_10827 = 10827.089

print("Reading models...")
f = h5py.File('/scratch/Dropbox/GIT/DeepLearning/universal_stokes_flow/database/models/models_training.h5', 'r')
models = f['model'][:]
f.close()

# Remove velocity at log(tau)=-1
print("Removing velocity at log(tau)=1")
models[:, :, 4] -= models[:, 22:23, 4]

ind = [1, 4, 3, 5, 6, 7]
depth = models[0, :, 0]
models = np.transpose(models, (0, 2, 1))[:, ind, :]
n_models = models.shape[0]

print("Reading Stokes...")
f = h5py.File('/scratch/Dropbox/GIT/DeepLearning/universal_stokes_flow/database/Si_10827/stokes_training.h5', 'r')
stokes = f['stokes'][:]
mu = f['mu'][:]
wave = f['wavelength'][:] - LAMBDA_10827
f.close()

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
# line center is at 0, so we want a dense knot spacing there to capture the
# sharp features of the Stokes profiles, and a coarser spacing in the wings.
degree = 3

# Coarse knots in the far wings
left = np.linspace(wave.min() + 0.05, -1.2, 5)
right = np.linspace(1.2, wave.max() - 0.05, 5)

# Use the original sampling of the training data in the line core between -1 and +1 A
idx1 = np.argmin(np.abs(wave + 1.0))
idx2 = np.argmin(np.abs(wave - 1.0))
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

# Fit I directly, but Q/I, U/I, V/I (rather than Q, U, V) with a symlog
# compression: symlog(amplitude * ratio) = sign(.) * log1p(|amplitude *
# ratio|). This keeps the profiles linear (and exactly zero where there's no
# polarization signal) near zero, while compressing the long tail from
# strong-field pixels. amplitude=100 is a starting point carried over from
# the previous fixed Q/U/V scale (~1e-2): it puts a ratio of that size just
# past the linear-to-log transition (100 * 1e-2 = 1) -- retune once you see
# the actual Q/I, U/I, V/I distribution for this line.
stokes_transform = StokesTransformConfig(amplitude=100.0)

# Fit Q, U, V as residuals with respect to the weak-field approximation
# built from Stokes I and the depth-averaged Bx, By, Bz (I is still fit
# directly). The residual is taken first, so together with the transform
# above the model fits symlog(a * (Q - Q_WFA) / I) etc.
#
# Si I 10827.089 A is 3s2 3p 4s 3P2 - 3s2 3p 4p 3P2 (SIR line 300): both
# levels have g = 1.5, so gbar = 1.5 and, since g_u = g_l, Delta = 0 and
# Gbar = gbar^2 = 2.25.
g_l = lande_factor(J=2, L=1, S=1)
g_u = lande_factor(J=2, L=1, S=1)
geff = effective_lande(g_u, g_l, J_u=2, J_l=2)
Geff = second_order_lande(g_u, g_l, J_u=2, J_l=2)
print(f"Si I 10827: gbar = {geff:.4f}, Gbar = {Geff:.4f}")

# Bx, By, Bz are in the line-of-sight frame (SIR takes Bz along the LOS), and
# SIR's reference direction for positive Q is rotated by 90 deg w.r.t. the
# textbook one, hence linear_polarization_sign=-1 (both checked against this
# database: V/V_WFA ~ 1.02 with no mu dependence, and the Q/U azimuth is
# chi_B + 90 deg).
# weak_field = WeakFieldConfig(
#     lambda0=LAMBDA_10827,
#     geff=geff,
#     Geff=Geff,
#     linear_polarization_sign=-1.0,
#     depth_range=None,  # average over all depth points
# )

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
                          plot_every=1,
                          stokes_loss_weights={"I": 1.0, "Q": 1.0, "U": 1.0, "V": 1.0}),
    checkpoint_path="weights/emulator_si10827.pt" if STOKES_BASIS == "bspline" else "weights/emulator_si10827_pca.pt",
    stokes_basis=STOKES_BASIS,
    stokes_pca_configs=stokes_pca_cfg,
    depth_features=depth_features
)