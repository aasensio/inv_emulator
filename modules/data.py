"""Turns raw [N, C, D] stratifications / [N, C, L] Stokes cubes into scaled,
per-channel B-spline coefficient tensors ready for model training.

Channels are kept separate throughout (a dict ``name -> (N, n_name)``
tensor), never concatenated into one ``(N, sum(n_name))`` tensor: each
channel has its own independent B-spline coefficient count and knot
placement, so concatenating them would produce a single axis that is not
monotonic in depth/wavelength.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .bspline import BSplineGroup
from .config import ScalerConfig
from .features import DepthFeatures
from .scaling import ChannelScaler, MultiChannelScaler


@dataclass
class PreparedData:
    x: Dict[str, torch.Tensor]  # phys channel name -> (N, n_name)
    mu: torch.Tensor  # (N, 1)
    y: Dict[str, torch.Tensor]  # stokes channel name -> (N, n_name)
    phys_scaler: MultiChannelScaler
    stokes_scaler: MultiChannelScaler
    mu_scaler: ChannelScaler


def prepare_dataset(
    physical: np.ndarray,
    stokes: np.ndarray,
    mu: np.ndarray,
    phys_group: BSplineGroup,
    stokes_group: BSplineGroup,
    phys_scaler: Optional[MultiChannelScaler] = None,
    stokes_scaler: Optional[MultiChannelScaler] = None,
    mu_scaler: Optional[ChannelScaler] = None,
    fit_scalers: bool = True,
    depth_features: Optional[DepthFeatures] = None,
) -> PreparedData:
    """Project + scale raw physical/Stokes cubes into per-channel
    coefficient tensors for the model.

    Parameters
    ----------
    physical: (N, 6, D) array (channel order must match phys_group.names).
    stokes: (N, 4, L) array (channel order must match stokes_group.names),
        or an iterable of n_mu such arrays (e.g. a generator, so that only
        one is in memory at a time) when every model is available at the
        same n_mu angles. The rows of ``y`` are then mu-major (row ``m * N +
        i`` is model ``i`` at ``mu[m]``), to be read through
        :class:`MultiMuDataset`.
    mu: (N,) or (N, 1) heliocentric angle (or cos(theta)) used as
        conditioning input; (n_mu,) if ``stokes`` is an iterable of angles.
    fit_scalers: if True, scalers are fit on this data (use only for the
        training split; reuse the fitted scalers for val/test/inference).
        This also fits the standardization of ``depth_features``.
    depth_features: if given, its features (computed from the unscaled
        physical coefficients) are added to ``x`` under their own names.
    """
    mu = np.asarray(mu, dtype=np.float64).reshape(-1, 1)

    coeffs_phys = phys_group.project(np.asarray(physical, dtype=np.float64))
    if getattr(stokes, "ndim", None) == 3:
        coeffs_stokes = stokes_group.project(np.asarray(stokes, dtype=np.float64))
    else:
        # One angle at a time, keeping float32 coefficients: float64 copies
        # of all the angles would not fit in memory.
        n_models = physical.shape[0]
        coeffs_stokes = {}
        for m, cube in enumerate(stokes):
            for name, c in stokes_group.project(np.asarray(cube, dtype=np.float64)).items():
                if name not in coeffs_stokes:
                    coeffs_stokes[name] = np.empty((mu.shape[0] * n_models, c.shape[1]), dtype=np.float32)
                coeffs_stokes[name][m * n_models:(m + 1) * n_models] = c

    if phys_scaler is None:
        phys_scaler = MultiChannelScaler({n: ScalerConfig(kind="standard") for n in phys_group.names})
    if stokes_scaler is None:
        stokes_scaler = MultiChannelScaler({n: ScalerConfig(kind="standard") for n in stokes_group.names})
    if mu_scaler is None:
        mu_scaler = ChannelScaler(ScalerConfig(kind="minmax", per_coeff=False))

    if fit_scalers:
        phys_scaler.fit(coeffs_phys)
        stokes_scaler.fit(coeffs_stokes)
        mu_scaler.fit(mu)

    scaled_phys = phys_scaler.transform(coeffs_phys)
    scaled_mu = mu_scaler.transform(mu)

    x = {n: torch.as_tensor(scaled_phys[n], dtype=torch.float32) for n in phys_group.names}
    y = {n: torch.as_tensor(stokes_scaler.scalers[n].transform(coeffs_stokes.pop(n)), dtype=torch.float32)
         for n in stokes_group.names}
    if depth_features is not None:
        raw = {n: torch.as_tensor(coeffs_phys[n]) for n in phys_group.names}
        if fit_scalers:
            depth_features.fit(raw)
        with torch.no_grad():
            x.update({n: f.float() for n, f in depth_features(raw).items()})

    return PreparedData(
        x=x,
        mu=torch.as_tensor(scaled_mu, dtype=torch.float32),
        y=y,
        phys_scaler=phys_scaler,
        stokes_scaler=stokes_scaler,
        mu_scaler=mu_scaler,
    )


class SplineDataset(Dataset):
    def __init__(self, x: Dict[str, torch.Tensor], mu: torch.Tensor, y: Dict[str, torch.Tensor]):
        n = mu.shape[0]
        for t in list(x.values()) + list(y.values()):
            assert t.shape[0] == n
        self.x = x
        self.mu = mu
        self.y = y
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        xi = {k: v[idx] for k, v in self.x.items()}
        yi = {k: v[idx] for k, v in self.y.items()}
        return xi, self.mu[idx], yi


class MultiMuDataset(Dataset):
    """Models available at the same ``n_mu`` angles: ``x`` has one row per
    model, ``mu`` one row per angle and ``y`` one row per (angle, model) pair,
    mu-major (see :func:`prepare_dataset`).

    With ``random_mu=True`` (training) an epoch visits every model once, each
    time at an angle drawn independently, so a batch mixes all the angles and
    the number of steps per epoch does not depend on ``n_mu``. With
    ``random_mu=False`` (validation) every (angle, model) pair is visited
    once, so the loss is deterministic and covers all the angles.

    ``idx`` must be a batch of indices, as yielded by a ``BatchSampler``.
    """

    def __init__(self, x: Dict[str, torch.Tensor], mu: torch.Tensor, y: Dict[str, torch.Tensor], random_mu: bool = True):
        self.n_models = next(iter(x.values())).shape[0]
        self.n_mu = mu.shape[0]
        for t in x.values():
            assert t.shape[0] == self.n_models
        for t in y.values():
            assert t.shape[0] == self.n_mu * self.n_models
        self.x = x
        self.mu = mu
        self.y = y
        self.random_mu = random_mu

    def __len__(self) -> int:
        return self.n_models if self.random_mu else self.n_mu * self.n_models

    def __getitem__(self, idx) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        idx = torch.as_tensor(idx)
        if self.random_mu:
            model, angle = idx, torch.randint(self.n_mu, idx.shape)
        else:
            model, angle = idx % self.n_models, idx // self.n_models
        row = angle * self.n_models + model
        xi = {k: v[model] for k, v in self.x.items()}
        yi = {k: v[row] for k, v in self.y.items()}
        return xi, self.mu[angle], yi
