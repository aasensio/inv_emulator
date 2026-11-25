"""PCA emulator of the He I 10830 slab coefficients, a fast replacement for
the per-wavelength neural field :class:`~mlp_bspline.hei.HeIEmulator`.

As for the photospheric lines (``si10827.py``), an MLP maps the model
parameters to principal-component coefficients of the output profiles, so
it is evaluated once per pixel instead of once per wavelength. The slab has
no depth stratification, so there are no B-splines: the inputs are the 7
parameters ``(Bx, By, Bz, deltav, a, height, theta)`` (field in G in
Hazel's vertical frame, Doppler width in km/s, damping, height in arcsec,
LOS angle in degrees) and the outputs the 11 profiles ``eps (I, Q, U, V)``,
``eta (I, Q, U, V)``, ``rho (Q, U, V)`` on the 256-point Hazel grid of
[10823, 10834] A of the training set
(``hazel_neural/database/synth_all.py``: ``v = 0``, ``tau = 1``, ``beta =
1``, quiet-Sun illumination).

Targets
-------
The slab formal solution (:func:`~mlp_bspline.hei.slab_formal_solution`
with ``ds = tau / max(eta_I)``) only depends on ``eta / max(eta_I)``,
``rho / max(eta_I)`` and ``eps / max(eta_I)``, so every profile is divided
by the maximum of ``eta_I`` of its model (and ``eps`` also by the
continuum ``I0_allen(10828.5 A, 1)``, to make it of order unity). These
normalized profiles are smooth and linear PCA represents them very
compactly: 30 components per channel reconstruct the validation set to
~1e-6 rms (no symlog transform is needed, unlike for the old neural
field).

Interface
---------
:meth:`HeIPCAEmulator.forward` has the signature and units of
:meth:`HeIEmulator.forward` (``eps`` multiplied back by the continuum
constant), so it plugs into :class:`~mlp_bspline.hei.SiHeSynthesizer`
unchanged; the profiles are rebuilt on the training grid and interpolated
to the requested (Doppler-shifted) wavelengths with a cubic spline.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Sequence

import h5py
# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .config import ChannelPCAConfig, ScalerConfig
from .hei import HEI_WAVE_RANGE, i0_allen
from .inversion import cubic_spline_interpolate, spline_second_derivative_matrix
from .model import MLP
from .pca import PCAProjector
from .scaling import ChannelScaler, MultiChannelScaler
from .train import TrainConfig

HEI_INPUTS = ("Bx", "By", "Bz", "deltav", "a", "height", "theta")
HEI_CHANNELS = ("eps_I", "eps_Q", "eps_U", "eps_V", "eta_I", "eta_Q", "eta_U", "eta_V", "rho_Q", "rho_U", "rho_V")
# continuum used to make eps of order unity (any constant works: it is
# multiplied back in forward())
HEI_EPS_UNIT = float(i0_allen(0.5 * (HEI_WAVE_RANGE[0] + HEI_WAVE_RANGE[1]), 1.0))


def normalize_coefficients(eps: np.ndarray, eta: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """``eps (N, 4, L)``, ``eta (N, 4, L)``, ``rho (N, 3, L)`` (Hazel units)
    -> the emulator's targets ``(N, 11, L)`` (float32): everything divided
    by ``max(eta_I)`` of each model, ``eps`` also by :data:`HEI_EPS_UNIT`."""
    norm = eta[:, 0].max(axis=-1)[:, None, None].astype(np.float64)
    out = np.concatenate([eps / (norm * HEI_EPS_UNIT), eta / norm, rho / norm], axis=1)
    return out.astype(np.float32)


def load_hei_data(root: str, label: str, start: int = 0, stop: Optional[int] = None):
    """Models ``start:stop`` of the ``label`` (training/validation/testing)
    set in ``root``: inputs ``(n, 7)`` (:data:`HEI_INPUTS`), normalized
    targets ``(n, 11, L)`` and the wavelength grid ``(L,)`` (A)."""
    with h5py.File(os.path.join(root, f"models_{label}.h5"), "r") as f:
        pars = np.concatenate([f["model_he"][start:stop], f["geom_he"][start:stop]], axis=1)
    with h5py.File(os.path.join(root, f"emissivity_absorption_{label}.h5"), "r") as f:
        targets = normalize_coefficients(f["eps"][start:stop], f["eta"][start:stop], f["rho"][start:stop])
        wave = f["wvl"][:].astype(np.float64)
    return pars.astype(np.float32), targets, wave


def _n_models(root: str, label: str) -> int:
    with h5py.File(os.path.join(root, f"models_{label}.h5"), "r") as f:
        return f["model_he"].shape[0]


class HeIPCAEmulator(nn.Module):
    """``(Bx, By, Bz, deltav, a, height, theta)`` -> MLP -> per-channel PCA
    coefficients -> normalized ``eps, eta, rho`` profiles. All the
    precomputed numbers (input scaling, PCA bases and means, coefficient
    scaling, the spline matrix of the grid) are frozen buffers."""

    def __init__(self, wave: np.ndarray, coeff_sizes: Dict[str, int], hidden_dims: Sequence[int] = (512,) * 4,
                 dropout: float = 0.0, layer_norm: bool = False):
        super().__init__()
        self.channels = list(HEI_CHANNELS)
        self.coeff_sizes = {c: int(coeff_sizes[c]) for c in self.channels}
        self.hidden_dims = list(hidden_dims)
        self.dropout, self.layer_norm = dropout, layer_norm
        # theta goes in the MLP's "mu" slot (the geometry), the rest are
        # one-coefficient "physical" channels
        self.model = MLP(list(HEI_INPUTS[:-1]), {n: 1 for n in HEI_INPUTS[:-1]}, self.channels, self.coeff_sizes,
                         hidden_dims=hidden_dims, dropout=dropout, layer_norm=layer_norm)
        wave = np.asarray(wave, dtype=np.float64)
        n = len(wave)
        self.wave64 = wave
        self.wave_center = 0.5 * (HEI_WAVE_RANGE[0] + HEI_WAVE_RANGE[1])
        self.wave_half_width = 0.5 * (HEI_WAVE_RANGE[1] - HEI_WAVE_RANGE[0])
        self.i0 = HEI_EPS_UNIT
        x = (wave - self.wave_center) / self.wave_half_width
        self.register_buffer("grid_x", torch.as_tensor(x, dtype=torch.float32))
        self.register_buffer("d2", torch.as_tensor(spline_second_derivative_matrix(x), dtype=torch.float32))
        self.register_buffer("in_offset", torch.zeros(len(HEI_INPUTS)))
        self.register_buffer("in_scale", torch.ones(len(HEI_INPUTS)))
        for c in self.channels:
            self.register_buffer(f"B_{c}", torch.zeros(n, self.coeff_sizes[c]))
            self.register_buffer(f"mean_{c}", torch.zeros(n))
            self.register_buffer(f"offset_{c}", torch.zeros(1))
            self.register_buffer(f"scale_{c}", torch.ones(1))

    @property
    def meta(self) -> dict:
        return {"wave": self.wave64.tolist(), "coeff_sizes": self.coeff_sizes, "hidden_dims": self.hidden_dims,
                "dropout": self.dropout, "layer_norm": self.layer_norm}

    def set_preprocessing(self, in_scaler: ChannelScaler, projectors: Dict[str, PCAProjector],
                          out_scaler: MultiChannelScaler):
        with torch.no_grad():
            self.in_offset.copy_(torch.as_tensor(in_scaler.offset, dtype=torch.float32).reshape(-1))
            self.in_scale.copy_(torch.as_tensor(in_scaler.scale, dtype=torch.float32).reshape(-1))
            for c in self.channels:
                getattr(self, f"B_{c}").copy_(torch.as_tensor(projectors[c].B, dtype=torch.float32))
                getattr(self, f"mean_{c}").copy_(torch.as_tensor(projectors[c].mean, dtype=torch.float32))
                s = out_scaler.scalers[c]
                getattr(self, f"offset_{c}").copy_(torch.as_tensor(s.offset, dtype=torch.float32).reshape(-1))
                getattr(self, f"scale_{c}").copy_(torch.as_tensor(s.scale, dtype=torch.float32).reshape(-1))

    def coefficients(self, pars: torch.Tensor) -> Dict[str, torch.Tensor]:
        """``pars (B, 7)`` -> scaled PCA coefficients (what the MLP is trained on)."""
        x = (pars - self.in_offset) / self.in_scale
        inputs = {n: x[:, i:i + 1] for i, n in enumerate(HEI_INPUTS[:-1])}
        return self.model(inputs, x[:, -1:])

    def decode(self, coeffs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Scaled PCA coefficients -> normalized profiles ``(B, 11, L)`` on the training grid."""
        out = []
        for c in self.channels:
            k = coeffs[c] * getattr(self, f"scale_{c}") + getattr(self, f"offset_{c}")
            out.append(k @ getattr(self, f"B_{c}").T + getattr(self, f"mean_{c}"))
        return torch.stack(out, dim=1)

    def profiles(self, pars: torch.Tensor) -> torch.Tensor:
        """``pars (B, 7)`` -> normalized profiles ``(B, 11, L)`` on the training grid."""
        return self.decode(self.coefficients(pars))

    def forward(self, pars: torch.Tensor, x: torch.Tensor):
        """As :meth:`HeIEmulator.forward`: ``pars (B, 7)``, ``x (B, L)``
        (rest-frame wavelengths mapped to ``(lambda - 10828.5) / 5.5``) ->
        ``eps (B, 4, L)``, ``eta (B, 4, L)``, ``rho (B, 3, L)``, normalized
        to ``max(eta_I) = 1`` on the training grid (``eps`` in units of
        :attr:`i0`, as the absolute ``eps`` of the neural field)."""
        prof = self.profiles(pars)
        xq = x.clamp(float(self.grid_x[0]), float(self.grid_x[-1]))
        out, _ = cubic_spline_interpolate(self.grid_x, prof, self.d2, xq)
        return out[:, 0:4] * self.i0, out[:, 4:8], out[:, 8:11]

    # -- persistence ----------------------------------------------------------
    def save(self, path: str):
        torch.save({"state_dict": self.state_dict(), "meta": self.meta}, path)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "HeIPCAEmulator":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        meta = ckpt["meta"]
        em = cls(np.asarray(meta["wave"]), meta["coeff_sizes"], meta["hidden_dims"],
                 dropout=meta.get("dropout", 0.0), layer_norm=meta.get("layer_norm", False))
        em.load_state_dict(ckpt["state_dict"])
        for p in em.parameters():
            p.requires_grad_(False)
        return em.eval()


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #

def _project(projectors: Dict[str, PCAProjector], targets: np.ndarray) -> Dict[str, np.ndarray]:
    return {c: projectors[c].project(targets[:, i].astype(np.float64)).astype(np.float32)
            for i, c in enumerate(HEI_CHANNELS)}


def _save_comparison_plot(path: str, emulator: HeIPCAEmulator, pars: np.ndarray, target: np.ndarray,
                          device: torch.device):
    with torch.no_grad():
        pred = emulator.profiles(torch.as_tensor(pars[None], device=device))[0].cpu().numpy()
    wave = emulator.wave64
    fig, axes = plt.subplots(3, 4, figsize=(12, 7), sharex=True)
    for ax in axes.flat:
        ax.set_visible(False)
    for i, c in enumerate(HEI_CHANNELS):
        ax = axes.flat[i if i < 8 else i + 1]
        ax.set_visible(True)
        ax.plot(wave, target[i], color="black", label="Hazel")
        ax.plot(wave, pred[i], color="tab:red", linestyle="--", label="emulator")
        ax.set_title(c, fontsize=9)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Bx=%.0f By=%.0f Bz=%.0f dv=%.1f a=%.2f h=%.1f theta=%.0f" % tuple(pars), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train_hei_emulator(
    root: str,
    pca_configs: Dict[str, ChannelPCAConfig],
    train_cfg: TrainConfig = TrainConfig(),
    input_scaler_config: ScalerConfig = ScalerConfig(kind="standard", per_coeff=True),
    output_scaler_configs: Optional[Dict[str, ScalerConfig]] = None,
    loss_weights: Optional[Dict[str, float]] = None,
    checkpoint_path: Optional[str] = None,
    n_pca_fit: int = 200_000,
    n_train: Optional[int] = None,
    chunk_size: int = 50_000,
) -> HeIPCAEmulator:
    """Fits the PCA bases on the first ``n_pca_fit`` training models,
    projects the training set (``n_train`` models, default all; read in
    chunks of ``chunk_size``) and the validation set of ``root`` onto them,
    and trains the MLP on the scaled coefficients as
    :func:`~mlp_bspline.train.train_emulator` does (Adam, ReduceLROnPlateau,
    early stopping, weighted MSE, best-validation checkpoint). The
    ``stokes_loss_weights``/``model_type`` fields of ``train_cfg`` are
    ignored (use ``loss_weights``, keyed by channel). With one scale per
    channel (``per_coeff=False``, the default) the loss is proportional to
    the profile-space MSE of each channel, as for the PCA Stokes basis."""
    torch.manual_seed(train_cfg.seed)
    if output_scaler_configs is None:
        output_scaler_configs = {c: ScalerConfig(kind="standard", per_coeff=False) for c in HEI_CHANNELS}
    loss_weights = {c: 1.0 for c in HEI_CHANNELS} | dict(loss_weights or {})

    n_total = _n_models(root, "training")
    n_train = n_total if n_train is None else min(n_train, n_total)

    print(f"Fitting PCA on {min(n_pca_fit, n_train)} training models...")
    _, targets, wave = load_hei_data(root, "training", 0, min(n_pca_fit, n_train))
    projectors = {}
    for i, c in enumerate(HEI_CHANNELS):
        projectors[c] = PCAProjector(wave, pca_configs[c]).fit(targets[:, i])
        print(f"PCA {c}: {projectors[c].n} components, explained variance "
              f"{projectors[c].explained_variance_fraction:.10f}")
    del targets

    print(f"Projecting {n_train} training models...")
    pars_train, coeffs_train = [], {c: [] for c in HEI_CHANNELS}
    for start in tqdm(range(0, n_train, chunk_size)):
        p, t, _ = load_hei_data(root, "training", start, min(start + chunk_size, n_train))
        pars_train.append(p)
        for c, k in _project(projectors, t).items():
            coeffs_train[c].append(k)
    pars_train = np.concatenate(pars_train)
    coeffs_train = {c: np.concatenate(v) for c, v in coeffs_train.items()}

    pars_val, targets_val, _ = load_hei_data(root, "validation")
    coeffs_val = _project(projectors, targets_val)

    in_scaler = ChannelScaler(input_scaler_config).fit(pars_train.astype(np.float64))
    out_scaler = MultiChannelScaler(output_scaler_configs).fit(coeffs_train)

    emulator = HeIPCAEmulator(wave, {c: projectors[c].n for c in HEI_CHANNELS}, train_cfg.hidden_dims,
                              dropout=train_cfg.dropout, layer_norm=train_cfg.layer_norm)
    emulator.set_preprocessing(in_scaler, projectors, out_scaler)
    device = torch.device(train_cfg.device)
    emulator.to(device)
    model = emulator.model
    print(f"model has {sum(p.numel() for p in model.parameters()):,} trainable parameters")

    def to_device(pars, coeffs):
        y = out_scaler.transform(coeffs)
        return (torch.as_tensor(pars, device=device),
                {c: torch.as_tensor(y[c], dtype=torch.float32, device=device) for c in HEI_CHANNELS})

    x_tr, y_tr = to_device(pars_train, coeffs_train)
    x_va, y_va = to_device(pars_val, coeffs_val)
    del coeffs_train
    n_tr, n_va = x_tr.shape[0], x_va.shape[0]

    def weighted_mse(pred, target):
        total, n = 0.0, 0
        for c, t in target.items():
            total = total + loss_weights[c] * (pred[c] - t).pow(2).sum()
            n += t.numel()
        return total / n

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=max(1, train_cfg.patience // 3) if train_cfg.patience else 10)

    if checkpoint_path:
        ckpt_stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
        os.makedirs("figs", exist_ok=True)

    best_val, best_state, since_improve = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": []}
    bs = train_cfg.batch_size
    for epoch in range(train_cfg.epochs):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        running = 0.0
        for start in tqdm(range(0, n_tr, bs), desc=f"epoch {epoch:4d}", leave=False):
            idx = perm[start:start + bs]
            optimizer.zero_grad()
            loss = weighted_mse(emulator.coefficients(x_tr[idx]), {c: v[idx] for c, v in y_tr.items()})
            loss.backward()
            optimizer.step()
            running += loss.item() * idx.shape[0]
        train_loss = running / n_tr

        model.eval()
        running = 0.0
        with torch.no_grad():
            for start in range(0, n_va, 8192):
                sl = slice(start, start + 8192)
                loss = weighted_mse(emulator.coefficients(x_va[sl]), {c: v[sl] for c, v in y_va.items()})
                running += loss.item() * x_va[sl].shape[0]
        val_loss = running / n_va

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if val_loss < best_val:
            best_val, since_improve = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if checkpoint_path:
                emulator.save(checkpoint_path)
        else:
            since_improve += 1

        if train_cfg.log_every and epoch % train_cfg.log_every == 0:
            print(f"epoch {epoch:4d}  train {train_loss:.3e}  val {val_loss:.3e}  "
                  f"lr {optimizer.param_groups[0]['lr']:.1e}")
        if checkpoint_path and train_cfg.plot_every and (epoch + 1) % train_cfg.plot_every == 0:
            model.eval()
            _save_comparison_plot(f"figs/{ckpt_stem}_comparison_epoch{epoch + 1:04d}.png", emulator,
                                  pars_val[0], targets_val[0], device)
        if train_cfg.patience and since_improve >= train_cfg.patience:
            print(f"early stopping at epoch {epoch}, best val {best_val:.3e}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    emulator.to("cpu").eval()
    if checkpoint_path:
        emulator.save(checkpoint_path)
    emulator.history = history
    return emulator
