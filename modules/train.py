"""Training loop for the physical-stratification -> Stokes-profile emulator.

Everything happens in B-spline coefficient space: the physical cubes and
Stokes cubes are projected once (up front) onto their per-channel B-spline
coefficients, scaled per-channel, and only then does the coefficient-space
model ever see them. That model is selected via ``TrainConfig.model_type``:
``"mlp"`` (the original flat MLP), ``"cnn"``, or ``"transformer"`` (both
encoder-decoder architectures aware of each channel's coefficient ordering;
see :mod:`mlp_bspline.seq_models`). Reconstruction back to full profiles
(via the precomputed B matrices) is deferred to the
:class:`~mlp_bspline.emulator.StokesEmulator` used at inference time,
keeping training itself cheap (it never touches the full [L]-dimensional
profiles). Alternatively (``stokes_basis="pca"``) the Stokes profiles are
represented by principal-component coefficients fit on the training data
(see :mod:`mlp_bspline.pca`). ``stokes_transform`` optionally reparameterizes what gets
projected/fit -- e.g. I and symlog(a * Q/I)/symlog(a * U/I)/symlog(a * V/I)
instead of I/Q/U/V directly (see :mod:`mlp_bspline.transforms`) -- with the
emulator inverting it back to physical units automatically at decode time.
``weak_field`` optionally makes Q/U/V residuals with respect to the
weak-field approximation built from I and the depth-averaged field (see
:mod:`mlp_bspline.weak_field`), which the emulator likewise adds back.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

try:
    from nvitop import Device as _NvitopDevice
except ImportError:
    _NvitopDevice = None

from .bspline import BSplineGroup
from .config import (
    ChannelPCAConfig, ChannelSplineConfig, DepthFeatureConfig, ScalerConfig, StokesTransformConfig,
    WeakFieldConfig,
    PHYSICAL_CHANNELS, STOKES_CHANNELS,
)
from .data import MultiMuDataset, SplineDataset, prepare_dataset
from .emulator import StokesEmulator, build_emulator, save_emulator
from .features import build_depth_features
from .pca import PCAGroup
from .seq_models import build_model
from .scaling import ChannelScaler, MultiChannelScaler
from .transforms import apply_stokes_transform
from .weak_field import subtract_weak_field


def _gpu_devices():
    """Best-effort list of nvitop ``Device`` handles; empty if nvitop or an
    NVIDIA driver isn't available."""
    if _NvitopDevice is None:
        return []
    try:
        return _NvitopDevice.all()
    except Exception:
        return []


def _gpu_postfix(devices) -> Dict[str, str]:
    """One tqdm postfix entry per GPU: utilization % and memory used."""
    postfix = {}
    for d in devices:
        try:
            postfix[f"gpu{d.index}"] = f"{d.gpu_utilization()}%"
            postfix[f"mem{d.index}"] = d.memory_used_human()
        except Exception:
            pass
    return postfix


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    model_type: str = "mlp"  # "mlp", "resmlp", "cnn", or "transformer" (see mlp_bspline.seq_models)
    hidden_dims: Sequence[int] = (256, 256, 256)  # only used when model_type == "mlp"
    dropout: float = 0.0  # only used when model_type == "mlp"
    layer_norm: bool = False  # only used when model_type == "mlp"
    model_kwargs: Dict = field(default_factory=dict)  # extra kwargs for "resmlp"/"cnn"/"transformer"
    optimizer: str = "adam"  # "adam" or "adamw" (decoupled weight decay)
    # "plateau": halve the lr when the val loss stalls. "cosine": linear
    # warmup over warmup_epochs, then per-step cosine decay to 0 at `epochs`.
    scheduler: str = "plateau"
    warmup_epochs: int = 5
    val_fraction: float = 0.1
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    stokes_loss_weights: Dict[str, float] = field(
        default_factory=lambda: {"I": 1.0, "Q": 1.0, "U": 1.0, "V": 1.0}
    )
    patience: Optional[int] = 30
    log_every: int = 10
    plot_every: int = 10
    # With several angles per model, the PCA basis is fit on this many training
    # models (at all the angles), since all of them would not fit in memory.
    pca_fit_models: int = 200_000


@dataclass
class TrainResult:
    emulator: StokesEmulator
    meta: dict
    history: Dict[str, list]


def _save_comparison_plot(
    path_prefix: str,
    emulator: StokesEmulator,
    physical_sample: np.ndarray,
    stokes_sample: np.ndarray,
    mu_sample: np.ndarray,
    stokes_grid: np.ndarray,
    device: torch.device,
):
    """Runs one stratification through the emulator and saves a ground-truth
    vs. predicted Stokes-profile comparison as ``{path_prefix}.png`` /
    ``.npy`` (the latter holding a ``(2, C_stokes, L)`` array: [ground
    truth, prediction])."""
    was_training = emulator.training
    emulator.eval()
    with torch.no_grad():
        phys_t = torch.as_tensor(physical_sample[None], dtype=torch.float32, device=device)
        mu_t = torch.as_tensor(mu_sample[None], dtype=torch.float32, device=device)
        pred = emulator(phys_t, mu_t)[0].cpu().numpy()
    if was_training:
        emulator.train()

    np.save(f"figs/{path_prefix}.npy", np.stack([stokes_sample, pred], axis=0))

    names = emulator.stokes_names
    fig, axes = plt.subplots(len(names), 1, figsize=(6, 2.2 * len(names)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, name, gt_c, pred_c in zip(axes, names, stokes_sample, pred):
        ax.plot(stokes_grid, gt_c, color="black", label="ground truth")
        ax.plot(stokes_grid, pred_c, color="tab:red", linestyle="--", label="emulator")
        ax.set_ylabel(name)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("wavelength")
    fig.tight_layout()
    fig.savefig(f"figs/{path_prefix}.png", dpi=150)
    plt.close(fig)


def train_emulator(
    physical: np.ndarray,
    stokes: np.ndarray,
    mu: np.ndarray,
    phys_grid: np.ndarray,
    stokes_grid: np.ndarray,
    phys_spline_configs: Dict[str, ChannelSplineConfig],
    stokes_spline_configs: Optional[Dict[str, ChannelSplineConfig]],
    phys_scaler_configs: Optional[Dict[str, ScalerConfig]] = None,
    stokes_scaler_configs: Optional[Dict[str, ScalerConfig]] = None,
    mu_scaler_config: Optional[ScalerConfig] = None,
    phys_names: Sequence[str] = PHYSICAL_CHANNELS,
    stokes_names: Sequence[str] = STOKES_CHANNELS,
    stokes_transform: Optional[StokesTransformConfig] = None,
    weak_field: Optional[WeakFieldConfig] = None,
    train_cfg: TrainConfig = TrainConfig(),
    checkpoint_path: Optional[str] = None,
    stokes_basis: str = "bspline",
    stokes_pca_configs: Optional[Dict[str, ChannelPCAConfig]] = None,
    depth_features: Optional[DepthFeatureConfig] = None,
) -> TrainResult:
    """``stokes_basis`` selects what the model predicts for each Stokes
    channel: ``"bspline"`` (default) the coefficients of the B-spline basis
    given by ``stokes_spline_configs``, or ``"pca"`` the coefficients of a
    principal-component basis given by ``stokes_pca_configs`` (one
    :class:`~mlp_bspline.config.ChannelPCAConfig` per channel;
    ``stokes_spline_configs`` is then ignored and may be ``None``). The PCA
    basis is fit on the training split of the profiles actually being fit
    (after ``weak_field``/``stokes_transform``), see :mod:`mlp_bspline.pca`.
    The physical stratifications always use B-splines.

    ``stokes_transform``, if given, reparameterizes what's actually fit:
    e.g. with ``StokesTransformConfig(amplitude=a)`` the model is fit
    against I and ``symlog(a * Q/I)``/``symlog(a * U/I)``/``symlog(a * V/I)``
    instead of I/Q/U/V directly (see :mod:`mlp_bspline.transforms`). This
    only changes what the B-spline coefficients (and hence the model)
    represent internally -- ``stokes`` here, ``emulator(...)`` at inference,
    and the saved comparison plots all remain in physical units, since
    :meth:`~mlp_bspline.emulator.StokesEmulator.decode` inverts the
    transform back automatically.

    ``weak_field``, if given, replaces Q/U/V by their residual with respect
    to the weak-field approximation computed from the true I and the
    depth-averaged field of ``physical`` (I itself is still fit directly).
    It is applied before ``stokes_transform``, so with both the model fits
    e.g. ``symlog(a * (Q - Q_WFA) / I)``. Again only the internal
    representation changes: the emulator rebuilds the WFA from its
    predicted I and adds it back.

    ``depth_features``, if given, adds derivatives with respect to the depth
    grid and cross-coupling products (e.g. ``dvz*dT``) of the physical
    channels as extra model inputs, computed from their B-spline
    coefficients (see :mod:`mlp_bspline.features`). The emulator computes
    them itself, so ``emulator(physical, mu)`` is unchanged.

    ``stokes`` is normally ``(N, 4, L)`` with one angle per model in ``mu``
    ``(N,)``. If every model was synthesized at the same angles, pass
    ``stokes`` as ``(n_mu, N, 4, L)`` and ``mu`` as ``(n_mu,)``: the
    train/validation split is then done by model, so that no stratification
    is in both, and the batches mix the angles (see
    :class:`~mlp_bspline.data.MultiMuDataset`).
    """
    rng = np.random.default_rng(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)

    multi_mu = stokes.ndim == 4
    n = physical.shape[0]
    perm = rng.permutation(n)
    n_val = max(1, int(n * train_cfg.val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    phys_group = BSplineGroup(phys_names, phys_grid, phys_spline_configs)
    features = build_depth_features(depth_features, phys_group) if depth_features is not None else None

    def fit_targets(idx):
        """What is actually fit, for the profiles ``idx``. Both the weak-field
        residual and the transform act on each profile independently, so
        the splits are processed separately, in place on their own copy:
        with 1e6 profiles every float64 copy of the whole cube is ~10 GB.
        With several angles per model the cube is n_mu times larger, so
        this is a generator yielding one angle at a time."""
        if multi_mu:
            return (fit_targets_cube(cube, idx) for cube in stokes)
        return fit_targets_cube(stokes, idx)

    def fit_targets_cube(cube, idx):
        if weak_field is None and stokes_transform is None:
            return cube[idx]
        s = cube[idx].astype(np.float64)  # fancy indexing copies: never the caller's array
        if weak_field is not None:
            s = subtract_weak_field(s, physical[idx], stokes_names, phys_names, stokes_grid, phys_grid, weak_field)
        if stokes_transform is not None:
            s = apply_stokes_transform(s, stokes_names, stokes_transform, inplace=True)
        return s

    stokes_train, stokes_val = fit_targets(train_idx), fit_targets(val_idx)  # lazy with multi_mu

    if stokes_basis == "bspline":
        stokes_group = BSplineGroup(stokes_names, stokes_grid, stokes_spline_configs)
    elif stokes_basis == "pca":
        if stokes_pca_configs is None:
            raise ValueError("stokes_basis='pca' requires stokes_pca_configs")
        stokes_group = PCAGroup(stokes_names, stokes_grid, stokes_pca_configs)
        if multi_mu:
            stokes_group.fit(np.concatenate(list(fit_targets(train_idx[:train_cfg.pca_fit_models]))))
        else:
            stokes_group.fit(stokes_train)
        for name in stokes_group.names:
            proj = stokes_group.projectors[name]
            print(f"PCA {name}: {proj.n} components, "
                  f"explained variance {proj.explained_variance_fraction:.8f}")
    else:
        raise ValueError(f"stokes_basis must be 'bspline' or 'pca', got {stokes_basis!r}")

    if phys_scaler_configs is None:
        phys_scaler_configs = {n_: ScalerConfig(kind="standard") for n_ in phys_names}
    if stokes_scaler_configs is None:
        stokes_scaler_configs = {n_: ScalerConfig(kind="standard") for n_ in stokes_names}
    if mu_scaler_config is None:
        mu_scaler_config = ScalerConfig(kind="minmax", per_coeff=False)

    phys_scaler = MultiChannelScaler(phys_scaler_configs)
    stokes_scaler = MultiChannelScaler(stokes_scaler_configs)
    mu_scaler = ChannelScaler(mu_scaler_config)

    train_data = prepare_dataset(
        physical[train_idx], stokes_train, mu if multi_mu else mu[train_idx],
        phys_group, stokes_group,
        phys_scaler=phys_scaler, stokes_scaler=stokes_scaler, mu_scaler=mu_scaler,
        fit_scalers=True, depth_features=features,
    )
    val_data = prepare_dataset(
        physical[val_idx], stokes_val, mu if multi_mu else mu[val_idx],
        phys_group, stokes_group,
        phys_scaler=phys_scaler, stokes_scaler=stokes_scaler, mu_scaler=mu_scaler,
        fit_scalers=False, depth_features=features,
    )

    # The samplers yield whole index batches, which SplineDataset slices in
    # one go: much faster than collating batch_size single samples.
    if multi_mu:
        train_set = MultiMuDataset(train_data.x, train_data.mu, train_data.y, random_mu=True)
        val_set = MultiMuDataset(val_data.x, val_data.mu, val_data.y, random_mu=False)
    else:
        train_set = SplineDataset(train_data.x, train_data.mu, train_data.y)
        val_set = SplineDataset(val_data.x, val_data.mu, val_data.y)
    train_loader = DataLoader(
        train_set, batch_size=None,
        sampler=BatchSampler(RandomSampler(train_set), train_cfg.batch_size, drop_last=False),
    )
    val_loader = DataLoader(
        val_set, batch_size=None,
        sampler=BatchSampler(SequentialSampler(val_set), train_cfg.batch_size, drop_last=False),
    )

    # Depth features are just more input "channels" to the model.
    model_in_names = list(phys_group.names) + (features.feature_names if features is not None else [])
    model_in_sizes = dict(phys_group.sizes, **(features.feature_sizes if features is not None else {}))
    base_model_kwargs = dict(
        phys_names=model_in_names, phys_sizes=model_in_sizes,
        stokes_names=list(stokes_group.names), stokes_sizes=dict(stokes_group.sizes),
    )
    if train_cfg.model_type == "mlp":
        model_kwargs = dict(
            base_model_kwargs, hidden_dims=list(train_cfg.hidden_dims),
            dropout=train_cfg.dropout, layer_norm=train_cfg.layer_norm,
        )
    elif train_cfg.model_type in ("resmlp", "cnn", "transformer"):
        model_kwargs = dict(base_model_kwargs, **train_cfg.model_kwargs)
    else:
        raise ValueError(f"model_type must be one of 'mlp', 'resmlp', 'cnn', 'transformer', "
                         f"got {train_cfg.model_type!r}")
    model = build_model(train_cfg.model_type, model_kwargs)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model has {n_params:,} trainable parameters")

    # Built once, up front: `emulator.model` is the very `model` object being
    # trained below, so its weights (and thus predictions) always reflect
    # the current training state without rebuilding anything.
    emulator, meta = build_emulator(
        phys_group, stokes_group, phys_scaler, stokes_scaler, mu_scaler,
        model, train_cfg.model_type, model_kwargs,
        stokes_transform=stokes_transform,
        weak_field=weak_field,
        stokes_basis=stokes_basis,
        depth_features=features,
    )
    device = torch.device(train_cfg.device)
    emulator.to(device)

    gpu_devices = _gpu_devices()

    ckpt_dir = ckpt_stem = None
    if checkpoint_path:
        ckpt_dir = os.path.dirname(checkpoint_path) or "."
        ckpt_stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
        os.makedirs(ckpt_dir, exist_ok=True)
        plot_idx = val_idx[0]

    stokes_loss_weights = train_cfg.stokes_loss_weights

    if train_cfg.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    elif train_cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    else:
        raise ValueError(f"optimizer must be 'adam' or 'adamw', got {train_cfg.optimizer!r}")
    if train_cfg.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=max(1, train_cfg.patience // 3) if train_cfg.patience else 10)
    elif train_cfg.scheduler == "cosine":
        total_steps = train_cfg.epochs * len(train_loader)
        warmup_steps = min(train_cfg.warmup_epochs * len(train_loader), total_steps - 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: (step + 1) / (warmup_steps + 1) if step < warmup_steps else
            0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps))),
        )
    else:
        raise ValueError(f"scheduler must be 'plateau' or 'cosine', got {train_cfg.scheduler!r}")

    def weighted_mse(pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Matches a single weighted mean over every (batch, coefficient)
        # entry across all channels -- computed channel-by-channel since
        # channels are never concatenated into one tensor.
        total = 0.0
        total_elems = 0
        for name, t in target.items():
            w = stokes_loss_weights.get(name, 1.0)
            total = total + w * (pred[name] - t).pow(2).sum()
            total_elems += t.numel()
        return total / total_elems

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    epochs_since_improve = 0

    for epoch in range(train_cfg.epochs):
        model.train()
        running = 0.0
        n_seen = 0
        batch_bar = tqdm(train_loader, desc=f"epoch {epoch:4d}", leave=False)
        for xb, mub, yb in batch_bar:
            xb = {k: v.to(device) for k, v in xb.items()}
            mub = mub.to(device)
            yb = {k: v.to(device) for k, v in yb.items()}
            optimizer.zero_grad()
            pred = model(xb, mub)
            loss = weighted_mse(pred, yb)
            loss.backward()
            optimizer.step()
            if train_cfg.scheduler == "cosine":
                scheduler.step()
            running += loss.item() * mub.shape[0]
            n_seen += mub.shape[0]
            batch_bar.set_postfix(loss=f"{loss.item():.6f}", **_gpu_postfix(gpu_devices))
        train_loss = running / n_seen

        model.eval()
        running = 0.0
        n_seen = 0
        with torch.no_grad():
            for xb, mub, yb in val_loader:
                xb = {k: v.to(device) for k, v in xb.items()}
                mub = mub.to(device)
                yb = {k: v.to(device) for k, v in yb.items()}
                pred = model(xb, mub)
                loss = weighted_mse(pred, yb)
                running += loss.item() * mub.shape[0]
                n_seen += mub.shape[0]
        val_loss = running / n_seen

        if train_cfg.scheduler == "plateau":
            scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
            if checkpoint_path:
                save_emulator(checkpoint_path, emulator, meta)
        else:
            epochs_since_improve += 1

        if train_cfg.log_every and epoch % train_cfg.log_every == 0:
            print(f"epoch {epoch:4d}  train {train_loss:.6f}  val {val_loss:.6f}")

        if checkpoint_path and train_cfg.plot_every and (epoch + 1) % train_cfg.plot_every == 0:
            _save_comparison_plot(
                os.path.join(ckpt_dir, f"{ckpt_stem}_comparison_epoch{epoch + 1:04d}"),
                emulator,
                physical[plot_idx],
                stokes[0, plot_idx] if multi_mu else stokes[plot_idx],
                mu[0] if multi_mu else mu[plot_idx],
                stokes_grid, device,
            )

        if train_cfg.patience and epochs_since_improve >= train_cfg.patience:
            print(f"early stopping at epoch {epoch}, best val {best_val:.6f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    emulator.to("cpu")

    if checkpoint_path:
        save_emulator(checkpoint_path, emulator, meta)

    return TrainResult(emulator=emulator, meta=meta, history=history)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _load_channel_configs(path: Optional[str], names: Sequence[str], default_num_coeffs: int, default_degree: int) -> Dict[str, ChannelSplineConfig]:
    overrides = {}
    if path:
        with open(path) as f:
            overrides = json.load(f)
    cfgs = {}
    for name in names:
        o = overrides.get(name, {})
        cfgs[name] = ChannelSplineConfig(
            num_coeffs=o.get("num_coeffs", default_num_coeffs),
            degree=o.get("degree", default_degree),
        )
    return cfgs


def _load_scaler_configs(path: Optional[str], names: Sequence[str]) -> Dict[str, ScalerConfig]:
    overrides = {}
    if path:
        with open(path) as f:
            overrides = json.load(f)
    cfgs = {}
    for name in names:
        o = overrides.get(name, {})
        cfgs[name] = ScalerConfig(
            kind=o.get("kind", "standard"),
            per_coeff=o.get("per_coeff", True),
            scale=o.get("scale"),
            offset=o.get("offset"),
        )
    return cfgs


def main():
    p = argparse.ArgumentParser(description="Train a B-spline + MLP Stokes profile emulator")
    p.add_argument("--physical", required=True, help=".npy file, shape [N, 6, D]")
    p.add_argument("--stokes", required=True, help=".npy file, shape [N, 4, L]")
    p.add_argument("--mu", required=True, help=".npy file, shape [N]")
    p.add_argument("--phys-grid", required=True, help=".npy file, shape [D], e.g. log(tau)")
    p.add_argument("--stokes-grid", required=True, help=".npy file, shape [L], wavelength")
    p.add_argument("--num-knots-phys", type=int, default=12)
    p.add_argument("--num-knots-stokes", type=int, default=24)
    p.add_argument("--degree", type=int, default=3)
    p.add_argument("--spline-config-phys", default=None, help="JSON per-channel override: {chan: {num_coeffs, degree}}")
    p.add_argument("--spline-config-stokes", default=None)
    p.add_argument("--scaler-config-phys", default=None, help="JSON per-channel override: {chan: {kind, per_coeff, scale, offset}}")
    p.add_argument("--scaler-config-stokes", default=None)
    p.add_argument("--stokes-basis", choices=["bspline", "pca"], default="bspline",
                    help="predict B-spline or PCA coefficients of the Stokes profiles")
    p.add_argument("--num-pca-stokes", type=int, default=None,
                    help="PCA components per Stokes channel (with --stokes-basis pca)")
    p.add_argument("--pca-variance-stokes", type=float, default=None,
                    help="instead of --num-pca-stokes: keep components up to this explained variance fraction")
    p.add_argument("--pca-config-stokes", default=None,
                    help="JSON per-channel override: {chan: {num_components | variance_fraction, center}}")
    p.add_argument("--depth-features", default=None,
                    help='JSON DepthFeatureConfig, e.g. \'{"derivatives":["T","vz","Bz"],'
                         '"products":["dvz*dT","dvz*dB","dvz*B"]}\'')
    p.add_argument("--stokes-loss-weights", default=None, help='JSON, e.g. {"I":1.0,"Q":2.0,"U":2.0,"V":3.0}')
    p.add_argument("--fit-stokes-ratio", action="store_true",
                    help="fit I and symlog(a * Q/I), symlog(a * U/I), symlog(a * V/I) instead of I/Q/U/V "
                         "directly (see mlp_bspline.transforms); a is set by --symlog-amplitude")
    p.add_argument("--symlog-amplitude", type=float, default=1.0,
                    help="the 'a' in symlog(a * x); only used with --fit-stokes-ratio")
    p.add_argument("--intensity-channel", default="I", help="normalizing channel; only used with --fit-stokes-ratio")
    p.add_argument("--wfa-lambda0", type=float, default=None,
                    help="fit Q/U/V as residuals w.r.t. the weak-field approximation for a line at this "
                         "rest wavelength (A); requires --wfa-geff and --wfa-Geff")
    p.add_argument("--wfa-geff", type=float, default=None, help="effective Lande factor (Stokes V)")
    p.add_argument("--wfa-Geff", type=float, default=None, help="second-order effective Lande factor (Stokes Q/U)")
    p.add_argument("--wfa-sign", type=float, default=1.0, choices=[1.0, -1.0],
                    help="+1: textbook Q/U sign; -1: reference direction rotated by 90 deg (SIR/Hazel)")
    p.add_argument("--wfa-depth-range", type=float, nargs=2, default=None,
                    help="depth window (min max) over which the field is averaged; default: all depths")
    p.add_argument("--wfa-wave-range", type=float, nargs=2, default=None,
                    help="wavelength window (min max, units of --stokes-grid) where the WFA is applied; "
                         "default: whole grid")
    p.add_argument("--wfa-lines", default=None,
                    help='JSON list of lines, each with its own window, instead of --wfa-lambda0/geff/Geff, '
                         'e.g. \'[{"lambda0":6301.5012,"geff":1.667,"Geff":2.517,"wave_range":[-1.2,0.5]}, '
                         '{"lambda0":6302.4936,"geff":2.5,"Geff":6.25,"wave_range":[0.5,1.8]}]\'')
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--model-type", choices=["mlp", "resmlp", "cnn", "transformer"], default="mlp",
                    help="coefficient-space architecture; cnn/transformer are encoder-decoders "
                         "that respect each channel's coefficient ordering (see mlp_bspline.seq_models)")
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256, 256], help="mlp only")
    p.add_argument("--dropout", type=float, default=0.0, help="mlp only")
    p.add_argument("--model-kwargs", default=None,
                    help='JSON kwargs for --model-type resmlp/cnn/transformer, e.g. '
                         '\'{"width":256,"num_blocks":4}\', \'{"channels":64,"num_layers":3}\' or '
                         '\'{"d_model":128,"nhead":8}\'')
    p.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--scheduler", choices=["plateau", "cosine"], default="plateau")
    p.add_argument("--warmup-epochs", type=int, default=5, help="cosine scheduler only")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--checkpoint", required=True, help="output .pt path")
    args = p.parse_args()

    physical = np.load(args.physical)
    stokes = np.load(args.stokes)
    mu = np.load(args.mu)
    phys_grid = np.load(args.phys_grid)
    stokes_grid = np.load(args.stokes_grid)

    phys_spline_cfg = _load_channel_configs(args.spline_config_phys, PHYSICAL_CHANNELS, args.num_knots_phys, args.degree)
    stokes_spline_cfg = _load_channel_configs(args.spline_config_stokes, STOKES_CHANNELS, args.num_knots_stokes, args.degree)
    phys_scaler_cfg = _load_scaler_configs(args.scaler_config_phys, PHYSICAL_CHANNELS)
    stokes_scaler_cfg = _load_scaler_configs(args.scaler_config_stokes, STOKES_CHANNELS)

    stokes_pca_cfg = None
    if args.stokes_basis == "pca":
        overrides = {}
        if args.pca_config_stokes:
            with open(args.pca_config_stokes) as f:
                overrides = json.load(f)
        stokes_pca_cfg = {}
        for name in STOKES_CHANNELS:
            o = overrides.get(name, {})
            if "num_components" not in o and "variance_fraction" not in o:
                if args.num_pca_stokes is None and args.pca_variance_stokes is None:
                    p.error(f"--stokes-basis pca needs --num-pca-stokes, --pca-variance-stokes or a "
                            f"--pca-config-stokes entry for {name}")
                o = dict(o, num_components=args.num_pca_stokes, variance_fraction=args.pca_variance_stokes)
            stokes_pca_cfg[name] = ChannelPCAConfig(**o)

    stokes_loss_weights = {"I": 1.0, "Q": 1.0, "U": 1.0, "V": 1.0}
    if args.stokes_loss_weights:
        stokes_loss_weights.update(json.loads(args.stokes_loss_weights))

    model_kwargs = json.loads(args.model_kwargs) if args.model_kwargs else {}

    stokes_transform = (
        StokesTransformConfig(amplitude=args.symlog_amplitude, intensity_channel=args.intensity_channel)
        if args.fit_stokes_ratio else None
    )

    weak_field = None
    if args.wfa_lines is not None:
        if args.wfa_lambda0 is not None or args.wfa_wave_range is not None:
            p.error("--wfa-lines excludes --wfa-lambda0/--wfa-wave-range")
        weak_field = WeakFieldConfig(
            lines=json.loads(args.wfa_lines),
            linear_polarization_sign=args.wfa_sign, depth_range=args.wfa_depth_range,
        )
    elif args.wfa_lambda0 is not None:
        if args.wfa_geff is None or args.wfa_Geff is None:
            p.error("--wfa-lambda0 requires --wfa-geff and --wfa-Geff")
        weak_field = WeakFieldConfig(
            lambda0=args.wfa_lambda0, geff=args.wfa_geff, Geff=args.wfa_Geff,
            linear_polarization_sign=args.wfa_sign, depth_range=args.wfa_depth_range,
            wave_range=args.wfa_wave_range,
        )

    train_cfg = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        model_type=args.model_type,
        hidden_dims=tuple(args.hidden_dims), dropout=args.dropout,
        model_kwargs=model_kwargs,
        optimizer=args.optimizer, weight_decay=args.weight_decay,
        scheduler=args.scheduler, warmup_epochs=args.warmup_epochs,
        val_fraction=args.val_fraction, patience=args.patience,
        stokes_loss_weights=stokes_loss_weights,
    )

    result = train_emulator(
        physical, stokes, mu, phys_grid, stokes_grid,
        phys_spline_cfg, stokes_spline_cfg,
        phys_scaler_configs=phys_scaler_cfg, stokes_scaler_configs=stokes_scaler_cfg,
        stokes_transform=stokes_transform, weak_field=weak_field,
        train_cfg=train_cfg, checkpoint_path=args.checkpoint,
        stokes_basis=args.stokes_basis, stokes_pca_configs=stokes_pca_cfg,
        depth_features=DepthFeatureConfig(**json.loads(args.depth_features)) if args.depth_features else None,
    )
    print(f"saved emulator to {args.checkpoint}")
    print(f"final train loss {result.history['train_loss'][-1]:.6f}  "
          f"val loss {result.history['val_loss'][-1]:.6f}")


if __name__ == "__main__":
    main()
