"""Train the PCA emulator of the He I 10830 slab (mlp_bspline/hei_pca.py):
(Bx, By, Bz, deltav, a, height, theta) -> PCA coefficients of eps (IQUV),
eta (IQUV), rho (QUV), normalized by max(eta_I). Replaces the neural field
of hei/ in invert_si10827_he10830.py."""
from modules.config import ChannelPCAConfig, ScalerConfig
from modules.hei_pca import HEI_CHANNELS, train_hei_emulator
from modules.train import TrainConfig

# Hazel2 syntheses (hazel_neural/database/synth_all.py): 1M training and
# 10k validation models on linspace(10823, 10834, 256)
ROOT = '/scratch/Dropbox/GIT/DeepLearning/hazel_neural/database'

# Linear PCA of the normalized profiles, fit on the first 200k training
# models. On the validation set 30 components reconstruct every channel to
# ~1e-6 rms (99th percentile of the max error per profile ~3e-6), far below
# what the MLP reaches, for 330 outputs in total.
pca_cfg = {c: ChannelPCAConfig(num_components=30) for c in HEI_CHANNELS}

# Inputs standardized one by one; for the outputs a single scale per channel
# keeps the loss proportional to the profile-space MSE of each channel (the
# basis is orthonormal).
input_scaler_cfg = ScalerConfig(kind="standard", per_coeff=True)
output_scaler_cfg = {c: ScalerConfig(kind="standard", per_coeff=False) for c in HEI_CHANNELS}

emulator = train_hei_emulator(
    ROOT,
    pca_cfg,
    train_cfg=TrainConfig(epochs=300,
                          batch_size=1024,
                          hidden_dims=(512, 512, 512, 512),
                          lr=1e-3,
                          patience=30,
                          log_every=1,
                          plot_every=10),
    input_scaler_config=input_scaler_cfg,
    output_scaler_configs=output_scaler_cfg,
    checkpoint_path="weights/emulator_he10830_pca.pt",
    n_pca_fit=200_000,
)
