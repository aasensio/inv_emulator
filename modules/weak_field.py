"""Weak-field approximation (WFA) baseline for Q, U, V, so the emulator can
fit only the residual with respect to it.

To first order in the Zeeman splitting (Landi Degl'Innocenti & Landolfi
2004, Sect. 9.6), for a magnetic field constant along the line of sight:

    V(l) = -C gbar l0^2 B_z dI/dl
    Q(l) = -s (1/4) C^2 Gbar l0^4 (B_x^2 - B_y^2) d^2I/dl^2
    U(l) = -s (1/4) C^2 Gbar l0^4 (2 B_x B_y)     d^2I/dl^2

with ``C = e / (4 pi m_e c^2) = 4.6686e-13 A^-1 G^-1`` (wavelengths in A,
fields in G), ``gbar`` the effective Lande factor and ``Gbar`` the
second-order effective Lande factor of the transverse Zeeman effect. ``B_x,
B_y, B_z`` are in the line-of-sight frame (``B_z`` along the LOS), which is
the frame Hazel/SIR take them in. ``B_x^2 - B_y^2 = B_perp^2 cos 2chi`` and
``2 B_x B_y = B_perp^2 sin 2chi``. ``s = +1`` is the textbook sign;
``s = -1`` corresponds to a reference direction for positive Q rotated by
90 deg, which is what SIR uses (see :class:`WeakFieldConfig`).

For several lines in the same spectral window (e.g. the Fe I 6301.5/6302.5
doublet) the prefactors -- i.e. l0, gbar and Gbar -- are wavelength
dependent: each line's values are used inside its own wavelength window
(:class:`~mlp_bspline.config.WeakFieldLine`) and the baseline is zero
outside all of them.

The field entering the formulae is the depth average of each component of
the stratification (optionally restricted to a depth window), and the
derivatives of I are taken by finite differences on the wavelength grid.
During training the WFA is built from the true I; at inference the emulator
builds it from its own predicted I, with exactly the same derivative
operator, so the decomposition ``Stokes = WFA(I) + residual`` is consistent
between the two.
"""
from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np
import torch

from .config import WeakFieldConfig

ArrayOrTensor = Union[np.ndarray, torch.Tensor]

# e / (4 pi m_e c^2) in A^-1 G^-1
LARMOR_CONSTANT = 4.6686e-13


def lande_factor(J: float, L: float, S: float) -> float:
    """Lande factor of an LS-coupled level."""
    if J == 0:
        return 0.0
    return 1.0 + (J * (J + 1) + S * (S + 1) - L * (L + 1)) / (2.0 * J * (J + 1))


def effective_lande(g_u: float, g_l: float, J_u: float, J_l: float) -> float:
    """First-order effective Lande factor ``gbar`` (LL04 eq. 3.44)."""
    return 0.5 * (g_u + g_l) + 0.25 * (g_u - g_l) * (J_u * (J_u + 1) - J_l * (J_l + 1))


def second_order_lande(g_u: float, g_l: float, J_u: float, J_l: float) -> float:
    """Second-order effective Lande factor ``Gbar = gbar^2 - Delta`` of the
    transverse Zeeman effect (LL04 eqs. 3.45-3.47), with

        Delta = (1/80) (g_u - g_l)^2 (16 s - 7 d^2 - 4),
        s = J_u(J_u+1) + J_l(J_l+1),  d = J_u(J_u+1) - J_l(J_l+1).
    """
    gbar = effective_lande(g_u, g_l, J_u, J_l)
    s = J_u * (J_u + 1) + J_l * (J_l + 1)
    d = J_u * (J_u + 1) - J_l * (J_l + 1)
    delta = (g_u - g_l) ** 2 * (16.0 * s - 7.0 * d ** 2 - 4.0) / 80.0
    return gbar ** 2 - delta


def wavelength_derivatives(intensity: ArrayOrTensor, wave: ArrayOrTensor) -> Tuple[ArrayOrTensor, ArrayOrTensor]:
    """First and second derivative of ``intensity (..., L)`` with respect to
    ``wave (L,)``: second-order central differences in the interior
    (valid on non-uniform grids), one-sided at the edges, and the second
    derivative as the derivative of the first -- i.e. ``np.gradient``
    applied twice, for both numpy arrays and torch tensors."""
    if isinstance(intensity, torch.Tensor):
        wave = torch.as_tensor(wave, dtype=intensity.dtype, device=intensity.device)
        d1 = torch.gradient(intensity, spacing=(wave,), dim=-1)[0]
        d2 = torch.gradient(d1, spacing=(wave,), dim=-1)[0]
        return d1, d2
    d1 = np.gradient(intensity, wave, axis=-1)
    d2 = np.gradient(d1, wave, axis=-1)
    return d1, d2


def depth_weights(depth: np.ndarray, cfg: WeakFieldConfig) -> np.ndarray:
    """Normalized averaging weights over the depth grid: uniform over all
    points, or over ``cfg.depth_range`` (inclusive) if given."""
    depth = np.asarray(depth, dtype=np.float64)
    if cfg.depth_range is None:
        mask = np.ones_like(depth, dtype=bool)
    else:
        lo, hi = sorted(cfg.depth_range)
        mask = (depth >= lo) & (depth <= hi)
        if not mask.any():
            raise ValueError(f"depth_range {cfg.depth_range} contains no points of the depth grid")
    return mask / mask.sum()


def average_field(
    physical: ArrayOrTensor, phys_names: Sequence[str], weights: ArrayOrTensor, cfg: WeakFieldConfig
) -> Tuple[ArrayOrTensor, ArrayOrTensor, ArrayOrTensor]:
    """``physical (N, C_phys, D)`` -> depth-averaged ``(Bx, By, Bz)``, each
    ``(N,)``."""
    names = list(phys_names)
    return tuple(physical[:, names.index(n), :] @ weights for n in cfg.field_channels)


def wfa_coefficients(wave: ArrayOrTensor, cfg: WeakFieldConfig) -> Tuple[ArrayOrTensor, ArrayOrTensor]:
    """Per-wavelength prefactors ``(k_V, k_QU)``, each ``(L,)``, of the WFA
    ``V = k_V B_z dI/dl`` and ``Q = k_QU (B_x^2 - B_y^2) d^2I/dl^2`` (same
    for U): each line of ``cfg`` sets them inside its ``wave_range`` (the
    whole grid if ``None``), and they are zero outside every window."""
    if isinstance(wave, torch.Tensor):
        kv, kqu = torch.zeros_like(wave), torch.zeros_like(wave)
    else:
        wave = np.asarray(wave, dtype=np.float64)
        kv, kqu = np.zeros_like(wave), np.zeros_like(wave)
    for line in cfg.all_lines:
        l0 = line.lambda0
        if line.wave_range is None:
            mask = slice(None)
        else:
            lo, hi = line.wave_range
            mask = (wave >= lo) & (wave <= hi)
        kv[mask] = -LARMOR_CONSTANT * line.geff * l0 ** 2
        kqu[mask] = -cfg.linear_polarization_sign * 0.25 * LARMOR_CONSTANT ** 2 * line.Geff * l0 ** 4
    return kv, kqu


def weak_field_stokes(
    intensity: ArrayOrTensor,
    wave: ArrayOrTensor,
    Bx: ArrayOrTensor,
    By: ArrayOrTensor,
    Bz: ArrayOrTensor,
    cfg: WeakFieldConfig,
) -> Tuple[ArrayOrTensor, ArrayOrTensor, ArrayOrTensor]:
    """WFA ``(Q, U, V)``, each ``(N, L)``, from ``intensity (N, L)`` and the
    ``(N,)`` field components (G). The derivatives of I are taken over the
    whole grid, and each line's prefactors are then applied inside its own
    wavelength window (see :func:`wfa_coefficients`)."""
    if isinstance(intensity, torch.Tensor):
        wave = torch.as_tensor(wave, dtype=intensity.dtype, device=intensity.device)
    d1, d2 = wavelength_derivatives(intensity, wave)
    kv, kqu = wfa_coefficients(wave, cfg)
    V = Bz[:, None] * (kv * d1)
    Q = (Bx ** 2 - By ** 2)[:, None] * (kqu * d2)
    U = (2.0 * Bx * By)[:, None] * (kqu * d2)
    return Q, U, V


def _stokes_indices(names: Sequence[str], cfg: WeakFieldConfig):
    names = list(names)
    return names.index(cfg.intensity_channel), [names.index(n) for n in cfg.stokes_channels]


def _add_weak_field(
    stokes: ArrayOrTensor,
    stokes_names: Sequence[str],
    wave: ArrayOrTensor,
    B: Tuple[ArrayOrTensor, ArrayOrTensor, ArrayOrTensor],
    cfg: WeakFieldConfig,
    sign: float,
) -> None:
    """In place (numpy): ``stokes[:, Q/U/V] += sign * WFA(stokes[:, I])``."""
    i_idx, quv_idx = _stokes_indices(stokes_names, cfg)
    wfa = weak_field_stokes(stokes[:, i_idx, :], wave, *B, cfg)
    for c, w in zip(quv_idx, wfa):
        stokes[:, c, :] += sign * w


def subtract_weak_field(
    stokes: np.ndarray,
    physical: np.ndarray,
    stokes_names: Sequence[str],
    phys_names: Sequence[str],
    wave: np.ndarray,
    depth: np.ndarray,
    cfg: WeakFieldConfig,
    chunk_size: int = 65536,
) -> np.ndarray:
    """Physical Stokes ``(N, C, L)`` -> fitting targets: I unchanged, Q/U/V
    replaced by their residual with respect to the WFA built from the true
    I and the depth-averaged field of ``physical (N, C_phys, D)``.

    Operates **in place** on ``stokes`` (which must be a float array; pass a
    copy if the original is still needed) and returns it, processing
    ``chunk_size`` profiles at a time to keep the temporaries small."""
    weights = depth_weights(depth, cfg)
    wave = np.asarray(wave, dtype=np.float64)
    for start in range(0, stokes.shape[0], chunk_size):
        sl = slice(start, start + chunk_size)
        B = average_field(np.asarray(physical[sl], dtype=np.float64), phys_names, weights, cfg)
        _add_weak_field(stokes[sl], stokes_names, wave, B, cfg, sign=-1.0)
    return stokes


def add_weak_field(
    residual: torch.Tensor,
    B: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    stokes_names: Sequence[str],
    wave: torch.Tensor,
    cfg: WeakFieldConfig,
) -> torch.Tensor:
    """Inverse of :func:`subtract_weak_field` (torch, out of place and
    autograd-safe): I plus Q/U/V residuals -> physical Stokes, with the WFA
    built from the (predicted) I in ``residual`` and the depth-averaged
    field ``B``."""
    i_idx, quv_idx = _stokes_indices(stokes_names, cfg)
    wfa = dict(zip(quv_idx, weak_field_stokes(residual[:, i_idx, :], wave, *B, cfg)))
    channels = [
        residual[:, c, :] + wfa[c] if c in wfa else residual[:, c, :]
        for c in range(residual.shape[1])
    ]
    return torch.stack(channels, dim=1)
