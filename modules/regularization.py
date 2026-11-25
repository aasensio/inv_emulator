"""Spatial regularizers and their proximal operators, for the inversion of
maps of physical parameters (see :mod:`mlp_bspline.inversion`).

Every operator acts on a batch of 2-D maps ``x (..., ny, nx)`` and returns

    prox_{tau R}(x) = argmin_z  0.5 ||z - x||^2 + tau R(z)

computed exactly (or, for TV, to the tolerance of a fixed number of inner
iterations):

``"l2"``
    ``R(z) = ||grad z||^2`` (Tikhonov smoothness, forward differences with
    Neumann boundaries). The prox solves ``(I + 2 tau grad^T grad) z = x``,
    which the orthonormal DCT-II diagonalizes exactly.
``"tv"``
    Isotropic total variation ``R(z) = sum sqrt(dz_x^2 + dz_y^2)``, prox by
    the fast gradient projection (FGP) of Beck & Teboulle (2009) on the dual.
``"l1"``
    ``R(z) = ||z||_1`` (sparsity of the values themselves): soft threshold.
``"wavelet_l1"``
    ``R(z) = ||detail coefficients of W z||_1`` for an orthonormal, periodic,
    multi-level separable 2-D wavelet transform ``W`` (Haar or any
    orthogonal Daubechies/Symlet/Coiflet from pywt). Because ``W`` is
    orthonormal the prox is exactly ``W^T soft(W x)``; approximation
    coefficients are not thresholded. When a level has an odd length, its
    last row/column is left out of that and the coarser levels (and is not
    thresholded), so ``W`` stays orthonormal for any map size; sizes
    divisible by ``2**levels`` regularize every pixel.

All transforms are built as small dense matrices (cached per size), so they
run on whatever device/dtype the maps live on.

:class:`TransverseRegularizer` applies any of these to the transverse field
``(Bx, By)`` through the ambiguity-invariant components
``Bt (cos 2phi, sin 2phi)``, so that the two solutions of the 180-degree
ambiguity are the same point and the jump of ``(Bx, By)`` where the
azimuth crosses the edge of the allowed range costs nothing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Sequence, Tuple, Union

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# finite differences (Neumann boundaries)
# --------------------------------------------------------------------------- #

def grad2d(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward differences along the last (x) and second to last (y) axis,
    zero at the far boundary."""
    gx = torch.zeros_like(x)
    gy = torch.zeros_like(x)
    gx[..., :, :-1] = x[..., :, 1:] - x[..., :, :-1]
    gy[..., :-1, :] = x[..., 1:, :] - x[..., :-1, :]
    return gx, gy


def grad2d_adjoint(gx: torch.Tensor, gy: torch.Tensor) -> torch.Tensor:
    """Adjoint of :func:`grad2d` (i.e. minus the divergence)."""
    out = torch.zeros_like(gx)
    out[..., :, :-1] -= gx[..., :, :-1]
    out[..., :, 1:] += gx[..., :, :-1]
    out[..., :-1, :] -= gy[..., :-1, :]
    out[..., 1:, :] += gy[..., :-1, :]
    return out


def total_variation(x: torch.Tensor) -> torch.Tensor:
    gx, gy = grad2d(x)
    return torch.sqrt(gx ** 2 + gy ** 2).sum(dim=(-2, -1))


# --------------------------------------------------------------------------- #
# proximal operators
# --------------------------------------------------------------------------- #

def prox_l1(x: torch.Tensor, tau) -> torch.Tensor:
    return torch.sign(x) * torch.clamp(torch.abs(x) - tau, min=0.0)


@lru_cache(maxsize=32)
def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    c = np.sqrt(2.0 / n) * np.cos(np.pi * k * (2 * j + 1) / (2 * n))
    c[0] /= np.sqrt(2.0)
    return c


def _as(a: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(a, dtype=like.dtype, device=like.device)


def prox_l2_gradient(x: torch.Tensor, tau) -> torch.Tensor:
    """Exact prox of ``tau ||grad z||^2`` (Neumann), via the DCT-II."""
    ny, nx = x.shape[-2:]
    cy, cx = _as(_dct_matrix(ny), x), _as(_dct_matrix(nx), x)
    ey = _as(2.0 - 2.0 * np.cos(np.pi * np.arange(ny) / ny), x)
    ex = _as(2.0 - 2.0 * np.cos(np.pi * np.arange(nx) / nx), x)
    tau = torch.as_tensor(tau, dtype=x.dtype, device=x.device).reshape(tau_shape(tau, x))
    coeff = cy @ x @ cx.T
    coeff = coeff / (1.0 + 2.0 * tau * (ey[:, None] + ex[None, :]))
    return cy.T @ coeff @ cx


def tau_shape(tau, x: torch.Tensor):
    """Broadcast a scalar or per-map ``tau (...)`` over ``(..., ny, nx)``."""
    t = torch.as_tensor(tau)
    return tuple(t.shape) + (1, 1) if t.dim() else ()


def prox_tv(x: torch.Tensor, tau, n_iter: int = 50) -> torch.Tensor:
    """Isotropic TV prox by FGP (Beck & Teboulle 2009) on the dual variable
    ``p`` (``|p| <= 1`` pointwise), with ``z = x - tau grad^T p``."""
    tau = torch.as_tensor(tau, dtype=x.dtype, device=x.device).reshape(tau_shape(tau, x))
    if torch.all(tau == 0):
        return x.clone()
    px, py = torch.zeros_like(x), torch.zeros_like(x)
    rx, ry = px.clone(), py.clone()
    t = 1.0
    step = 1.0 / (8.0 * torch.clamp(tau, min=1e-30))
    for _ in range(n_iter):
        z = x - tau * grad2d_adjoint(rx, ry)
        gx, gy = grad2d(z)
        qx, qy = rx + step * gx, ry + step * gy
        norm = torch.clamp(torch.sqrt(qx ** 2 + qy ** 2), min=1.0)
        qx, qy = qx / norm, qy / norm
        t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        rx = qx + ((t - 1.0) / t_new) * (qx - px)
        ry = qy + ((t - 1.0) / t_new) * (qy - py)
        px, py, t = qx, qy, t_new
    return x - tau * grad2d_adjoint(px, py)


@lru_cache(maxsize=64)
def _wavelet_1d(n: int, wavelet: str, levels: int) -> Tuple[np.ndarray, np.ndarray]:
    """Orthonormal multi-level periodic DWT matrix ``W (n, n)`` for a 1-D
    signal of length ``n``, and a boolean mask of its detail rows. At each
    level only the rows holding the current approximation are transformed;
    with an odd count the last one is left out of this and all later levels
    (it is kept, unthresholded, as is). ``W`` is orthonormal for any ``n``,
    and every detail row sums to zero, so constant signals have no detail
    coefficients."""
    import pywt

    w = pywt.Wavelet(wavelet)
    if not w.orthogonal:
        raise ValueError(f"wavelet {wavelet!r} is not orthogonal")
    lo, hi = np.asarray(w.dec_lo), np.asarray(w.dec_hi)
    W = np.eye(n)
    detail = np.zeros(n, dtype=bool)
    approx = np.arange(n)  # rows of W currently holding approximation coefficients
    for _ in range(levels):
        even = len(approx) - len(approx) % 2
        if even < 2:
            break
        half = even // 2
        A = np.zeros((even, even))
        for k in range(half):
            for j in range(len(lo)):
                A[k, (2 * k + j) % even] += lo[j]
                A[half + k, (2 * k + j) % even] += hi[j]
        if not np.allclose(A @ A.T, np.eye(even), atol=1e-10):
            break  # block shorter than the filter: stop decomposing
        rows = approx[:even]
        W[rows] = A @ W[rows]
        detail[rows[half:]] = True
        # an odd leftover sample leaves the pyramid: mixing it with the
        # next level's (differently normalized) averages would produce detail
        # rows that do not sum to zero
        approx = rows[:half]
    return W, detail


def prox_wavelet_l1(x: torch.Tensor, tau, wavelet: str = "haar", levels: int = 3) -> torch.Tensor:
    ny, nx = x.shape[-2:]
    wy, dy = _wavelet_1d(ny, wavelet, levels)
    wx, dx = _wavelet_1d(nx, wavelet, levels)
    wy, wx = _as(wy, x), _as(wx, x)
    # a 2-D coefficient is "approximation" only if it is approximation along both axes
    detail = torch.as_tensor(dy[:, None] | dx[None, :], device=x.device)
    tau = torch.as_tensor(tau, dtype=x.dtype, device=x.device).reshape(tau_shape(tau, x))
    c = wy @ x @ wx.T
    c = torch.where(detail, prox_l1(c, tau), c)
    return wy.T @ c @ wx


# --------------------------------------------------------------------------- #
# user-facing regularizer spec
# --------------------------------------------------------------------------- #

@dataclass
class Regularizer:
    """One spatial regularization term ``weight * R(map)`` applied to every
    node map of a physical variable, in that variable's physical units (K,
    km/s, G).

    kind:
        ``"l2"``, ``"tv"``, ``"l1"`` or ``"wavelet_l1"`` (see module doc).
    weight:
        Scalar, or one value per node (length = number of nodes of the
        variable) to regularize heights differently; 0 disables a node.
    options:
        ``{"n_iter": 50}`` for ``"tv"``; ``{"wavelet": "db2", "levels": 3}``
        for ``"wavelet_l1"``.
    """

    kind: str
    weight: Union[float, Sequence[float]]
    options: Dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in _PROX:
            raise ValueError(f"kind must be one of {sorted(_PROX)}, got {self.kind!r}")

    def prox(self, maps: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """maps (n_nodes, ny, nx), step (n_nodes,) -> prox_{step * weight * R}."""
        w = torch.as_tensor(self.weight, dtype=maps.dtype, device=maps.device)
        w = w.expand(maps.shape[0]) if w.dim() == 0 else w
        if w.shape[0] != maps.shape[0]:
            raise ValueError(f"{self.kind}: {w.shape[0]} weights for {maps.shape[0]} nodes")
        return _PROX[self.kind](maps, step * w, **self.options)

    def value(self, maps: torch.Tensor) -> torch.Tensor:
        w = torch.as_tensor(self.weight, dtype=maps.dtype, device=maps.device)
        w = w.expand(maps.shape[0]) if w.dim() == 0 else w
        if self.kind == "l2":
            gx, gy = grad2d(maps)
            r = (gx ** 2 + gy ** 2).sum(dim=(-2, -1))
        elif self.kind == "tv":
            r = total_variation(maps)
        elif self.kind == "l1":
            r = maps.abs().sum(dim=(-2, -1))
        else:
            ny, nx = maps.shape[-2:]
            wy, dy = _wavelet_1d(ny, self.options.get("wavelet", "haar"), self.options.get("levels", 3))
            wx, dx = _wavelet_1d(nx, self.options.get("wavelet", "haar"), self.options.get("levels", 3))
            c = _as(wy, maps) @ maps @ _as(wx, maps).T
            detail = torch.as_tensor(dy[:, None] | dx[None, :], device=maps.device)
            r = torch.where(detail, c.abs(), torch.zeros_like(c)).sum(dim=(-2, -1))
        return (w * r).sum()


def to_doubled_azimuth(bx: torch.Tensor, by: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(Bx, By) = Bt (cos phi, sin phi)`` -> ``Bt (cos 2phi, sin 2phi)``,
    invariant under ``(Bx, By) -> (-Bx, -By)``."""
    bt = torch.sqrt(bx ** 2 + by ** 2)
    inv = torch.where(bt > 0, 1.0 / torch.clamp(bt, min=1e-30), torch.zeros_like(bt))
    return (bx ** 2 - by ** 2) * inv, 2.0 * bx * by * inv


def from_doubled_azimuth(u: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`to_doubled_azimuth` with ``phi`` in ``(-90, 90]``
    degrees, i.e. ``Bx >= 0``."""
    bt = torch.sqrt(u ** 2 + v ** 2)
    phi = 0.5 * torch.atan2(v, u)
    return bt * torch.cos(phi), bt * torch.sin(phi)


@dataclass
class TransverseRegularizer:
    """Regularization of the transverse field of the variables ``bx`` and
    ``by`` (same number of nodes) that is invariant under the 180-degree
    ambiguity: every regularizer in ``regularization`` (see
    :class:`Regularizer`) acts on the maps of ``u = Bt cos 2phi`` and
    ``v = Bt sin 2phi`` (G), and the result is mapped back with ``Bx >= 0``
    (the inversion must bound ``Bx`` below by 0). A jump of ``Bt`` costs as
    in ``Bx``/``By``, a change of azimuth ``dphi`` as a jump of
    ``2 Bt dphi``. The prox of the regularizers is taken in ``(u, v)``, an
    approximation of the exact prox in ``(Bx, By)``; it reduces to the
    exact one for the field strength, while azimuth changes are pulled
    back twice as strongly."""

    bx: str
    by: str
    regularization: Sequence[Regularizer]

    def prox(self, bx: torch.Tensor, by: torch.Tensor, step: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """bx, by (n_nodes, ny, nx), step (n_nodes,) -> prox in ``(u, v)``."""
        u, v = to_doubled_azimuth(bx, by)
        for reg in self.regularization:
            u, v = reg.prox(u, step), reg.prox(v, step)
        return from_doubled_azimuth(u, v)

    def value(self, bx: torch.Tensor, by: torch.Tensor) -> torch.Tensor:
        u, v = to_doubled_azimuth(bx, by)
        return sum(reg.value(u) + reg.value(v) for reg in self.regularization)


_PROX = {
    "l2": lambda x, tau, **kw: prox_l2_gradient(x, tau),
    "tv": lambda x, tau, **kw: prox_tv(x, tau, **kw),
    "l1": lambda x, tau, **kw: prox_l1(x, tau.reshape(tau_shape(tau, x))),
    "wavelet_l1": lambda x, tau, **kw: prox_wavelet_l1(x, tau, **kw),
}
