"""The deployable end-to-end module: stratification -> B-spline projection
-> coefficient-space model -> B-spline (or PCA) reconstruction -> Stokes
profiles.

Everything needed at inference time (the projection matrices ``P`` for each
physical channel, the design matrices ``B`` for each Stokes channel, and the
per-channel scaler affine coefficients) is stored as frozen (``requires_grad
= False``) parameters inside the module, so the whole pipeline is a single
``nn.Module`` that can be saved/loaded/moved to GPU as one unit, and whose
only *trainable* parameters are the coefficient-space model's (an MLP, CNN
encoder-decoder, or Transformer encoder-decoder; see
:mod:`mlp_bspline.seq_models`). When trained with a
:class:`~mlp_bspline.config.WeakFieldConfig`, the wavelength grid and the
depth-averaging weights needed to rebuild the weak-field baseline are
stored the same way, and so are the mean profiles of a PCA Stokes basis
(``stokes_basis="pca"``, see :mod:`mlp_bspline.pca`), whose eigenvectors
simply take the place of the B-spline design matrices in ``B_stokes``.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .bspline import BSplineGroup
from .config import DepthFeatureConfig, StokesTransformConfig, WeakFieldConfig
from .features import DepthFeatures
from .scaling import ChannelScaler, MultiChannelScaler
from .seq_models import build_model
from .transforms import invert_stokes_transform
import numpy as np

from .weak_field import add_weak_field, average_field, depth_weights, wfa_coefficients

def _scaler_shape(scaler: ChannelScaler, n_coeffs: int) -> int:
    return n_coeffs if scaler.cfg.per_coeff else 1


class StokesEmulator(nn.Module):
    def __init__(
        self,
        phys_names: Sequence[str],
        phys_grid_sizes: Dict[str, int],
        phys_coeff_sizes: Dict[str, int],
        phys_scale_sizes: Dict[str, int],
        stokes_names: Sequence[str],
        stokes_grid_sizes: Dict[str, int],
        stokes_coeff_sizes: Dict[str, int],
        stokes_scale_sizes: Dict[str, int],
        model: nn.Module,
        stokes_transform_cfg: Optional[StokesTransformConfig] = None,
        weak_field_cfg: Optional[WeakFieldConfig] = None,
        stokes_grid_size: Optional[int] = None,
        phys_grid_size: Optional[int] = None,
        stokes_basis: str = "bspline",
        depth_features: Optional[DepthFeatures] = None,
    ):
        super().__init__()
        self.stokes_basis = stokes_basis
        self.phys_names = list(phys_names)
        self.stokes_names = list(stokes_names)
        self.phys_coeff_sizes = dict(phys_coeff_sizes)
        self.stokes_coeff_sizes = dict(stokes_coeff_sizes)
        self.model = model
        self.stokes_transform_cfg = stokes_transform_cfg
        self.weak_field_cfg = weak_field_cfg
        # Only registered when used, so checkpoints trained without depth
        # features keep loading unchanged.
        self.depth_features = depth_features

        self.P_phys = nn.ParameterDict({
            name: nn.Parameter(
                torch.zeros(phys_coeff_sizes[name], phys_grid_sizes[name]), requires_grad=False
            )
            for name in self.phys_names
        })
        self.B_stokes = nn.ParameterDict({
            name: nn.Parameter(
                torch.zeros(stokes_grid_sizes[name], stokes_coeff_sizes[name]), requires_grad=False
            )
            for name in self.stokes_names
        })
        self.phys_offset = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(phys_scale_sizes[name]), requires_grad=False)
            for name in self.phys_names
        })
        self.phys_scale = nn.ParameterDict({
            name: nn.Parameter(torch.ones(phys_scale_sizes[name]), requires_grad=False)
            for name in self.phys_names
        })
        self.stokes_offset = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(stokes_scale_sizes[name]), requires_grad=False)
            for name in self.stokes_names
        })
        self.stokes_scale = nn.ParameterDict({
            name: nn.Parameter(torch.ones(stokes_scale_sizes[name]), requires_grad=False)
            for name in self.stokes_names
        })
        if stokes_basis == "pca":
            # Mean profile added back on reconstruction. Only created for the
            # PCA basis, so B-spline checkpoints keep loading unchanged.
            self.stokes_mean = nn.ParameterDict({
                name: nn.Parameter(torch.zeros(stokes_grid_sizes[name]), requires_grad=False)
                for name in self.stokes_names
            })
        elif stokes_basis != "bspline":
            raise ValueError(f"stokes_basis must be 'bspline' or 'pca', got {stokes_basis!r}")
        self.mu_offset = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.mu_scale = nn.Parameter(torch.ones(1), requires_grad=False)
        if weak_field_cfg is not None:
            # Only created when needed, so checkpoints trained without the
            # weak-field baseline keep loading unchanged.
            self.wfa_wave = nn.Parameter(torch.zeros(stokes_grid_size), requires_grad=False)
            self.wfa_depth_weights = nn.Parameter(torch.zeros(phys_grid_size), requires_grad=False)

    # -- filling in precomputed numerical content -------------------------------
    def set_projection_matrices(self, phys_group: BSplineGroup, stokes_group: BSplineGroup):
        with torch.no_grad():
            for name in self.phys_names:
                self.P_phys[name].copy_(torch.as_tensor(phys_group.projectors[name].P, dtype=torch.float32))
            for name in self.stokes_names:
                self.B_stokes[name].copy_(torch.as_tensor(stokes_group.projectors[name].B, dtype=torch.float32))
                if self.stokes_basis == "pca":
                    self.stokes_mean[name].copy_(
                        torch.as_tensor(stokes_group.projectors[name].mean, dtype=torch.float32)
                    )

    def set_scalers(self, phys_scaler: MultiChannelScaler, stokes_scaler: MultiChannelScaler, mu_scaler: ChannelScaler):
        with torch.no_grad():
            for name in self.phys_names:
                s = phys_scaler.scalers[name]
                self.phys_offset[name].copy_(torch.as_tensor(s.offset, dtype=torch.float32).reshape(-1))
                self.phys_scale[name].copy_(torch.as_tensor(s.scale, dtype=torch.float32).reshape(-1))
            for name in self.stokes_names:
                s = stokes_scaler.scalers[name]
                self.stokes_offset[name].copy_(torch.as_tensor(s.offset, dtype=torch.float32).reshape(-1))
                self.stokes_scale[name].copy_(torch.as_tensor(s.scale, dtype=torch.float32).reshape(-1))
            self.mu_offset.copy_(torch.as_tensor(mu_scaler.offset, dtype=torch.float32).reshape(-1))
            self.mu_scale.copy_(torch.as_tensor(mu_scaler.scale, dtype=torch.float32).reshape(-1))

    def set_weak_field(self, stokes_grid, phys_grid):
        # The line windows are applied to the float32 copy of the grid at
        # inference and to the float64 one in training: they must select
        # the same points, or train/inference baselines would differ.
        wave32 = torch.as_tensor(stokes_grid, dtype=torch.float32)
        for k64, k32 in zip(wfa_coefficients(np.asarray(stokes_grid, dtype=np.float64), self.weak_field_cfg),
                            wfa_coefficients(wave32, self.weak_field_cfg)):
            if not np.array_equal(k64 != 0, k32.numpy() != 0):
                raise ValueError("a weak-field wave_range boundary coincides with a wavelength grid point to "
                                 "float32 precision; move it away from the grid points")
        with torch.no_grad():
            self.wfa_wave.copy_(torch.as_tensor(stokes_grid, dtype=torch.float32))
            self.wfa_depth_weights.copy_(
                torch.as_tensor(depth_weights(phys_grid, self.weak_field_cfg), dtype=torch.float32)
            )

    # -- pipeline stages, usable independently for coefficient-space training ---
    def encode(self, physical: torch.Tensor, mu: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """physical: (B, C_phys, D), mu: (B,) or (B, 1) -> (per-channel
        scaled coefficients, scaled mu (B, 1)). Channels are kept separate
        (never concatenated into one tensor): each has its own coefficient
        count, so concatenating them would produce an axis that is not
        monotonic in depth. With depth features, these are added as further
        entries of the dict (see :mod:`mlp_bspline.features`)."""
        if mu.dim() == 1:
            mu = mu.unsqueeze(-1)
        coeffs = {}
        raw = {}
        for i, name in enumerate(self.phys_names):
            f = physical[:, i, :]
            raw[name] = f @ self.P_phys[name].t()
            coeffs[name] = (raw[name] - self.phys_offset[name]) / self.phys_scale[name]
        if self.depth_features is not None:
            coeffs.update(self.depth_features(raw))
        mu_s = (mu - self.mu_offset) / self.mu_scale
        return coeffs, mu_s

    def decode(self, y_scaled: Dict[str, torch.Tensor], physical: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Per-channel scaled predicted Stokes coefficients -> physical
        Stokes profiles (B, C_stokes, L).

        If the emulator was trained with a ``stokes_transform_cfg`` (fit
        against e.g. symlog(a * Q/I) instead of Q), the B-spline
        reconstruction below yields that fitting-space profile first, and
        this then inverts it back to physical units -- so the public
        contract (physical Stokes profiles in, physical Stokes profiles
        out) is unaffected by whether/how the model was reparameterized
        internally.

        If it was trained with a ``weak_field_cfg``, Q/U/V are residuals
        with respect to the weak-field approximation: the WFA is rebuilt
        from the predicted I and the depth-averaged field of ``physical``
        (B, C_phys, D) -- which is then required -- and added back.
        """
        outs = []
        for name in self.stokes_names:
            c = y_scaled[name] * self.stokes_scale[name] + self.stokes_offset[name]
            f = c @ self.B_stokes[name].t()
            if self.stokes_basis == "pca":
                f = f + self.stokes_mean[name]
            outs.append(f)
        stacked = torch.stack(outs, dim=1)
        if self.stokes_transform_cfg is not None:
            stacked = invert_stokes_transform(stacked, self.stokes_names, self.stokes_transform_cfg)
        if self.weak_field_cfg is not None:
            if physical is None:
                raise ValueError("this emulator fits residuals w.r.t. the weak-field approximation; "
                                 "decode() needs the physical stratification")
            B = average_field(physical, self.phys_names, self.wfa_depth_weights, self.weak_field_cfg)
            stacked = add_weak_field(stacked, B, self.stokes_names, self.wfa_wave, self.weak_field_cfg)
        return stacked

    def forward(self, physical: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        coeffs, mu_s = self.encode(physical, mu)
        y = self.model(coeffs, mu_s)
        return self.decode(y, physical)


def build_emulator(
    phys_group: BSplineGroup,
    stokes_group: BSplineGroup,
    phys_scaler: MultiChannelScaler,
    stokes_scaler: MultiChannelScaler,
    mu_scaler: ChannelScaler,
    model: nn.Module,
    model_type: str,
    model_kwargs: dict,
    stokes_transform: Optional[StokesTransformConfig] = None,
    weak_field: Optional[WeakFieldConfig] = None,
    stokes_basis: str = "bspline",
    depth_features: Optional[DepthFeatures] = None,
) -> Tuple[StokesEmulator, dict]:
    phys_scale_sizes = {
        n: _scaler_shape(phys_scaler.scalers[n], phys_group.sizes[n]) for n in phys_group.names
    }
    stokes_scale_sizes = {
        n: _scaler_shape(stokes_scaler.scalers[n], stokes_group.sizes[n]) for n in stokes_group.names
    }
    phys_grid_sizes = {n: phys_group.projectors[n].n_grid for n in phys_group.names}
    stokes_grid_sizes = {n: stokes_group.projectors[n].n_grid for n in stokes_group.names}

    meta = {
        "phys_names": list(phys_group.names),
        "phys_grid_sizes": phys_grid_sizes,
        "phys_coeff_sizes": dict(phys_group.sizes),
        "phys_scale_sizes": phys_scale_sizes,
        "stokes_names": list(stokes_group.names),
        "stokes_grid_sizes": stokes_grid_sizes,
        "stokes_coeff_sizes": dict(stokes_group.sizes),
        "stokes_scale_sizes": stokes_scale_sizes,
        "in_dim": sum(phys_group.sizes.values()) + 1
                  + (sum(depth_features.feature_sizes.values()) if depth_features is not None else 0),
        "out_dim": sum(stokes_group.sizes.values()),
        "model_type": model_type,
        "model_kwargs": model_kwargs,
        "stokes_transform": dataclasses.asdict(stokes_transform) if stokes_transform is not None else None,
        "weak_field": dataclasses.asdict(weak_field) if weak_field is not None else None,
        "stokes_grid_size": len(stokes_group.grid),
        "phys_grid_size": len(phys_group.grid),
        "stokes_basis": stokes_basis,
        "depth_features": dataclasses.asdict(depth_features.cfg) if depth_features is not None else None,
        "depth_feature_eval_points": depth_features.eval_points if depth_features is not None else None,
    }

    emulator = StokesEmulator(
        phys_names=meta["phys_names"],
        phys_grid_sizes=meta["phys_grid_sizes"],
        phys_coeff_sizes=meta["phys_coeff_sizes"],
        phys_scale_sizes=meta["phys_scale_sizes"],
        stokes_names=meta["stokes_names"],
        stokes_grid_sizes=meta["stokes_grid_sizes"],
        stokes_coeff_sizes=meta["stokes_coeff_sizes"],
        stokes_scale_sizes=meta["stokes_scale_sizes"],
        model=model,
        stokes_transform_cfg=stokes_transform,
        weak_field_cfg=weak_field,
        stokes_grid_size=meta["stokes_grid_size"],
        phys_grid_size=meta["phys_grid_size"],
        stokes_basis=stokes_basis,
        depth_features=depth_features,
    )
    emulator.set_projection_matrices(phys_group, stokes_group)
    emulator.set_scalers(phys_scaler, stokes_scaler, mu_scaler)
    if weak_field is not None:
        emulator.set_weak_field(stokes_group.grid, phys_group.grid)
    return emulator, meta


def save_emulator(path: str, emulator: StokesEmulator, meta: dict):
    torch.save({"state_dict": emulator.state_dict(), "meta": meta}, path)


def load_emulator(path: str, map_location: Optional[str] = None) -> Tuple[StokesEmulator, dict]:
    """Loads a saved emulator, reconstructing whichever model architecture
    (MLP / CNN / Transformer) it was trained with.

    Also loads checkpoints saved before ``model_type``/``model_kwargs``
    existed (when the only option was the MLP and its submodule was named
    ``mlp`` instead of ``model``): the legacy meta fields
    (``in_dim``/``out_dim``/``hidden_dims``/...) are translated into
    ``model_kwargs`` for ``"mlp"``, and the ``"mlp."``-prefixed state_dict
    keys are remapped to ``"model."``.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    meta = ckpt["meta"]
    state_dict = ckpt["state_dict"]

    model_type = meta.get("model_type", "mlp")
    model_kwargs = meta.get("model_kwargs")
    if model_kwargs is None:
        model_kwargs = dict(
            phys_names=meta["phys_names"], phys_sizes=meta["phys_coeff_sizes"],
            stokes_names=meta["stokes_names"], stokes_sizes=meta["stokes_coeff_sizes"],
            hidden_dims=meta["hidden_dims"],
            dropout=meta.get("dropout", 0.0), layer_norm=meta.get("layer_norm", False),
        )
        state_dict = {
            ("model." + k[len("mlp."):] if k.startswith("mlp.") else k): v
            for k, v in state_dict.items()
        }

    stokes_transform_dict = meta.get("stokes_transform")
    stokes_transform_cfg = StokesTransformConfig(**stokes_transform_dict) if stokes_transform_dict else None
    weak_field_dict = meta.get("weak_field")
    weak_field_cfg = WeakFieldConfig(**weak_field_dict) if weak_field_dict else None

    depth_features = None
    if meta.get("depth_features"):
        depth_features = DepthFeatures(DepthFeatureConfig(**meta["depth_features"]), meta["phys_names"],
                                       meta["phys_coeff_sizes"], meta["depth_feature_eval_points"])

    model = build_model(model_type, model_kwargs)
    emulator = StokesEmulator(
        phys_names=meta["phys_names"],
        phys_grid_sizes=meta["phys_grid_sizes"],
        phys_coeff_sizes=meta["phys_coeff_sizes"],
        phys_scale_sizes=meta["phys_scale_sizes"],
        stokes_names=meta["stokes_names"],
        stokes_grid_sizes=meta["stokes_grid_sizes"],
        stokes_coeff_sizes=meta["stokes_coeff_sizes"],
        stokes_scale_sizes=meta["stokes_scale_sizes"],
        model=model,
        stokes_transform_cfg=stokes_transform_cfg,
        weak_field_cfg=weak_field_cfg,
        stokes_grid_size=meta.get("stokes_grid_size"),
        phys_grid_size=meta.get("phys_grid_size"),
        stokes_basis=meta.get("stokes_basis", "bspline"),
        depth_features=depth_features,
    )
    emulator.load_state_dict(state_dict)
    return emulator, meta
