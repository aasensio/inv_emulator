"""Elementwise preprocessing applied to raw Stokes profiles before they are
fit, and its exact inverse for reconstructing physical profiles at
inference time.

Fitting Q, U, V directly is awkward: their amplitude is set by the local
magnetic field and can span several orders of magnitude across a dataset,
while also needing to stay exactly zero (and change sign) in the
non-magnetic case. Two standard remedies, composed here:

1. Normalize by the (always-positive) Stokes I profile, i.e. fit Q/I, U/I,
   V/I instead of Q, U, V -- this removes the overall intensity scale from
   the polarization channels.
2. Apply a symmetric log ("symlog") to those ratios: ``symlog(a * x) =
   sign(a * x) * log1p(|a * x|)``. This is linear (and sign-preserving)
   for ``|x| << 1/a`` and logarithmic for ``|x| >> 1/a``, so it compresses
   the long tail of strong-field profiles without distorting the
   near-zero, weak-field regime where linearity (and an exact zero) is
   physically required. ``a`` is user-selectable: it sets where the
   linear-to-logarithmic transition happens, and should be tuned to the
   typical size of the ratios in the dataset (larger ``a`` compresses
   smaller amplitudes more aggressively).
"""
from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import torch

from .config import StokesTransformConfig

ArrayOrTensor = Union[np.ndarray, torch.Tensor]


def symlog(x: ArrayOrTensor, a: float = 1.0) -> ArrayOrTensor:
    """``sign(a * x) * log1p(|a * x|)``."""
    if isinstance(x, torch.Tensor):
        z = a * x
        return torch.sign(z) * torch.log1p(torch.abs(z))
    z = a * x
    return np.sign(z) * np.log1p(np.abs(z))


def inv_symlog(y: ArrayOrTensor, a: float = 1.0) -> ArrayOrTensor:
    """Exact inverse of :func:`symlog`: recovers ``x`` from ``y =
    symlog(a * x)``."""
    if isinstance(y, torch.Tensor):
        z = torch.sign(y) * torch.expm1(torch.abs(y))
        return z / a
    z = np.sign(y) * np.expm1(np.abs(y))
    return z / a


def _channel_indices(names: Sequence[str], cfg: StokesTransformConfig):
    names = list(names)
    if cfg.intensity_channel not in names:
        raise ValueError(f"intensity_channel {cfg.intensity_channel!r} not found in channels {names}")
    ratio_channels = (
        list(cfg.ratio_channels) if cfg.ratio_channels is not None
        else [n for n in names if n != cfg.intensity_channel]
    )
    i_idx = names.index(cfg.intensity_channel)
    r_idx = [names.index(n) for n in ratio_channels]
    return i_idx, r_idx


def apply_stokes_transform(
    stokes: ArrayOrTensor, names: Sequence[str], cfg: StokesTransformConfig, inplace: bool = False
) -> ArrayOrTensor:
    """Physical Stokes profiles ``(..., C, L)`` -> model-fitting targets:
    ``intensity_channel`` unchanged, every ``ratio_channels`` entry replaced
    by ``symlog(a * channel / intensity)``. With ``inplace=True``,
    ``stokes`` itself (which must then be a float array) is overwritten and
    returned, saving a copy of a large training cube."""
    i_idx, r_idx = _channel_indices(names, cfg)
    if inplace:
        out = stokes
    else:
        out = stokes.clone() if isinstance(stokes, torch.Tensor) else stokes.copy()
    intensity = stokes[:, i_idx, :]
    for c in r_idx:
        ratio = stokes[:, c, :] / intensity
        out[:, c, :] = symlog(ratio, cfg.amplitude)
    return out


def invert_stokes_transform(
    targets: ArrayOrTensor, names: Sequence[str], cfg: StokesTransformConfig
) -> ArrayOrTensor:
    """Exact inverse of :func:`apply_stokes_transform`: model-fitting
    targets -> physical Stokes profiles."""
    i_idx, r_idx = _channel_indices(names, cfg)
    out = targets.clone() if isinstance(targets, torch.Tensor) else targets.copy()
    intensity = targets[:, i_idx, :]
    for c in r_idx:
        ratio = inv_symlog(targets[:, c, :], cfg.amplitude)
        out[:, c, :] = ratio * intensity
    return out
