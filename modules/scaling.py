"""Per-channel scaling of B-spline coefficients.

Different physical channels (T vs. vz vs. Bz, ...) and different Stokes
channels (I vs. Q/U/V, which are typically 2-3 orders of magnitude smaller)
live on very different scales. Each channel gets its own
:class:`ChannelScaler`, independently configured, and grouped together in a
:class:`MultiChannelScaler`.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .config import ScalerConfig


class ChannelScaler:
    """Affine scaler ``y = (x - offset) / scale`` for one channel's
    B-spline coefficients."""

    def __init__(self, cfg: ScalerConfig):
        self.cfg = cfg
        self.offset: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None
        if cfg.kind == "fixed":
            self.offset = np.asarray(cfg.offset, dtype=np.float64)
            self.scale = np.asarray(cfg.scale, dtype=np.float64)
        elif cfg.kind == "none":
            self.offset = np.asarray(0.0)
            self.scale = np.asarray(1.0)

    def fit(self, x: np.ndarray) -> "ChannelScaler":
        """x: (N, n_coeffs)."""
        if self.cfg.kind in ("fixed", "none"):
            return self  # nothing to fit
        axis = 0 if self.cfg.per_coeff else None
        if self.cfg.kind == "standard":
            offset = x.mean(axis=axis)
            scale = x.std(axis=axis)
        elif self.cfg.kind == "minmax":
            xmin = x.min(axis=axis)
            xmax = x.max(axis=axis)
            offset = xmin
            scale = xmax - xmin
        else:
            raise ValueError(self.cfg.kind)
        scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
        self.offset = np.asarray(offset, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.offset is None:
            raise RuntimeError("ChannelScaler must be fit() before use")
        return (x - self.offset) / self.scale

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.offset is None:
            raise RuntimeError("ChannelScaler must be fit() before use")
        return x * self.scale + self.offset


class MultiChannelScaler:
    """dict name -> ChannelScaler."""

    def __init__(self, configs: Dict[str, ScalerConfig]):
        self.configs = configs
        self.scalers: Dict[str, ChannelScaler] = {
            name: ChannelScaler(cfg) for name, cfg in configs.items()
        }

    def fit(self, coeffs: Dict[str, np.ndarray]) -> "MultiChannelScaler":
        for name, scaler in self.scalers.items():
            scaler.fit(coeffs[name])
        return self

    def transform(self, coeffs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {name: self.scalers[name].transform(coeffs[name]) for name in self.scalers}

    def inverse_transform(self, coeffs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {name: self.scalers[name].inverse_transform(coeffs[name]) for name in self.scalers}
