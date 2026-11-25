import numpy as np
import h5py
from astropy.io import fits
from modules.config import ChannelPCAConfig, ChannelSplineConfig, ScalerConfig, StokesTransformConfig, DepthFeatureConfig
from modules.bspline import clamped_knots_from_positions
from modules.train import TrainConfig, train_emulator

# What the model predicts for the Stokes profiles: "bspline" coefficients or
# "pca" coefficients (principal components fit on the training profiles).
STOKES_BASIS = "pca"

# Rest wavelength of Ca II 8542 (air)
LAMBDA_8542 = 8542.09

# Ca II 8542 is on the first 150 of the 401 wavelengths of the database (the
# remaining ones are Ca II K)
N_WAVE = 150

# Stokes profiles are in absolute units: normalize them to the mean intensity
# at the blue end of the grid at mu=1 in the training set.
I_NORM = 2.59e-5

# How vz(log tau=-1) is handled. Unlike Fe I 6301/6302, whose database was
# synthesized after setting vz(log tau=-1) = 0, the Ca II database was
# synthesized with the models as they are, so the profiles carry that
# velocity as a Doppler shift (subtracting it only from the models, as in
# fe6301.py, gives errors ~5 times larger). "rest" removes it from both models
# and profiles, moving every profile to the rest frame of log(tau)=-1, so the
# emulator follows the same convention as the Fe I ones in modules.inversion.
# "absolute" keeps it in both, for inversions with reference_depth=None.
VELOCITY_FRAME = "rest"

# Number of models to read. None reads all of them.
N_MODELS = None

C_KMS = 299792.458
LOGTAU_REF_INDEX = 22
CHUNK = 50000


def read_stokes(path, n_models=None):
    """Stokes [n_mu, N, 4, N_WAVE] (float32), wavelengths [N_WAVE] and mu
    [n_mu] of a Ca II FITS file. The file is ~32 GB for the training set and
    only 150/401 of it is needed, so it is read in chunks."""
    with fits.open(path, memmap=True) as hdu:
        data = hdu[0].data
        wave = np.array(hdu[1].data[:N_WAVE], dtype=np.float64)
        mu = np.array(hdu[2].data, dtype=np.float64)
        n_mu, n = data.shape[:2]
        n = n if n_models is None else min(n, n_models)
        stokes = np.empty((n_mu, n, 4, N_WAVE), dtype=np.float32)
        for m in range(n_mu):
            for start in range(0, n, CHUNK):
                stop = min(start + CHUNK, n)
                stokes[m, start:stop] = data[m, start:stop, :, :N_WAVE]
    if not np.allclose(np.diff(wave), wave[1] - wave[0]) or not wave[0] < LAMBDA_8542 < wave[-1]:
        raise ValueError(f"unexpected wavelength axis: {wave[0]} - {wave[-1]}")
    return stokes, wave, mu


def to_rest_frame(stokes, v, wave):
    """Removes the Doppler shift ``v`` (km/s, [N], positive = redshift) from
    ``stokes`` [n_mu, N, 4, L], in place: S_rest(l) = S(l (1 + v / c)), the
    inverse of what modules.inversion applies. Cubic (Catmull-Rom)
    interpolation; beyond the ends of the grid the end values are repeated."""
    n_wave = len(wave)
    grid = np.arange(n_wave)
    pixels = LAMBDA_8542 * v / C_KMS / (wave[1] - wave[0])
    for start in range(0, len(v), CHUNK):
        pos = grid[None, :] + pixels[start:start + CHUNK, None]
        pos = np.clip(pos, 0, n_wave - 1)
        i0 = np.minimum(np.floor(pos).astype(np.int64), n_wave - 2)
        t = (pos - i0)[:, None, :]
        w = (0.5 * t * ((2 - t) * t - 1), 0.5 * (t * t * (3 * t - 5) + 2),
             0.5 * t * ((4 - 3 * t) * t + 1), 0.5 * t * t * (t - 1))
        idx = [np.clip(i0 + k, 0, n_wave - 1)[:, None, :] for k in (-1, 0, 1, 2)]
        for cube in stokes:
            part = cube[start:start + CHUNK]
            part[:] = sum(wk * np.take_along_axis(part, ik, axis=2) for wk, ik in zip(w, idx))


print("Reading models...")
f = h5py.File('/scratch/Dropbox/GIT/DeepLearning/universal_stokes_flow/database/models/models_training.h5', 'r')
models = f['model'][:N_MODELS]
f.close()

# stokes: [n_mu, N, 4, L] in channel order I, Q, U, V. Every model is available
# at the same 5 heliocentric angles and train_emulator picks them at random
# when building the batches (see modules.data.MultiMuDataset).
print("Reading Stokes...")
stokes, wave, mu = read_stokes('/scratch1/datasets/caii_universal/synthetic_allmus_final_CaII_training.fits', N_MODELS)
wave -= LAMBDA_8542

# Failed syntheses are flagged with 1e13 (model 62411 at mu=0.2). Remove the
# model at all angles, since a [n_mu, N, 4, L] cube needs every angle.
good = (np.abs(stokes) < 1.0).all(axis=(0, 2, 3))
print(f"Removing {np.sum(~good)} models with failed syntheses")
if not good.all():
    models = models[good]
    stokes = stokes[:, good]
stokes /= I_NORM

if VELOCITY_FRAME == "rest":
    print("Moving profiles to the rest frame of log(tau)=-1")
    v_ref = models[:, LOGTAU_REF_INDEX, 4].astype(np.float64)
    to_rest_frame(stokes, v_ref, wave)
    models[:, :, 4] -= models[:, LOGTAU_REF_INDEX:LOGTAU_REF_INDEX + 1, 4]

ind = [1, 4, 3, 5, 6, 7]
depth = models[0, :, 0]
models = np.transpose(models, (0, 2, 1))[:, ind, :]
n_models = models.shape[0]

# physical: [N, 6, D] in channel order T, vz, vmic, Bx, By, Bz
# depth, wave: [D], [L] grids

phys_cfg = {
    "T":  ChannelSplineConfig(num_coeffs=10, degree=3),
    "vz": ChannelSplineConfig(num_coeffs=10, degree=3),
    "vmic": ChannelSplineConfig(num_coeffs=10, degree=3),
    "Bx": ChannelSplineConfig(num_coeffs=10, degree=3),
    "By": ChannelSplineConfig(num_coeffs=10, degree=3),
    "Bz": ChannelSplineConfig(num_coeffs=10, degree=3),
}

# Sampling positions for the interior knots of the Stokes B-splines, used
# when STOKES_BASIS == "bspline". The wavelength axis is relative to
# 8542.09 A, with a 0.03 A step between -2.25 and +2.22 A: original sampling
# in the line core and coarser knots in the wings.
degree = 3

# Coarse knots in the far wings
left = np.linspace(wave.min() + 0.05, -0.55, 5)
right = np.linspace(0.55, wave.max() - 0.05, 5)

# Use the original sampling of the training data in the line core between -0.5 and +0.5 A
idx1 = np.argmin(np.abs(wave + 0.5))
idx2 = np.argmin(np.abs(wave - 0.5))
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
# split of what is actually fit (symlog transform below), per Stokes
# parameter, with all the angles together.
stokes_pca_cfg = {
    "I": ChannelPCAConfig(variance_fraction=0.99999),
    "Q": ChannelPCAConfig(variance_fraction=0.99999),
    "U": ChannelPCAConfig(variance_fraction=0.99999),
    "V": ChannelPCAConfig(variance_fraction=0.99999),
}

# Fit I directly, but Q/I, U/I, V/I with a symlog compression (see si10827.py).
stokes_transform = StokesTransformConfig(amplitude=30.0)

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
    checkpoint_path="weights/emulator_caii8542.pt" if STOKES_BASIS == "bspline" else "weights/emulator_caii8542_pca_resmlp512_cosine_deriv.pt",
    stokes_basis=STOKES_BASIS,
    stokes_pca_configs=stokes_pca_cfg,
    depth_features=depth_features,
)
