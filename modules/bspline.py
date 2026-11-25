"""B-spline design/projection matrices for stratifications and Stokes profiles.

The core idea (per channel, independently):

    f(grid) ≈ B @ c            (reconstruction: coefficients -> sampled function)
    c = P @ f(grid)             (projection: sampled function -> coefficients, least squares)

``B`` is the usual B-spline design/collocation matrix (shape ``D x n``, D =
number of grid points, n = number of B-spline coefficients). ``P`` is its
Moore-Penrose pseudo-inverse (shape ``n x D``), precomputed once so that at
synthesis time projecting a stratification onto its B-spline coefficients is
a single matrix multiply, and so is reconstructing a Stokes profile from
predicted coefficients.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
from scipy.interpolate import BSpline


def clamped_uniform_knots(x_min: float, x_max: float, num_coeffs: int, degree: int) -> np.ndarray:
    """Build a clamped knot vector with uniformly spaced interior knots.

    Produces exactly ``num_coeffs`` B-spline basis functions of the given
    degree over ``[x_min, x_max]``.
    """
    n_interior = num_coeffs - degree - 1
    if n_interior < 0:
        raise ValueError(
            f"num_coeffs ({num_coeffs}) must be >= degree + 1 ({degree + 1})"
        )
    if n_interior > 0:
        interior = np.linspace(x_min, x_max, n_interior + 2, dtype=np.float64)[1:-1]
    else:
        interior = np.zeros(0, dtype=np.float64)
    knots = np.concatenate(
        [np.full(degree + 1, x_min, dtype=np.float64),
         interior,
         np.full(degree + 1, x_max, dtype=np.float64)]
    )
    return knots


def clamped_knots_from_positions(
    x_min: float, x_max: float, positions: Iterable[float], degree: int
) -> np.ndarray:
    """Build a clamped knot vector from explicit interior knot positions.

    Like :func:`clamped_uniform_knots`, but instead of spacing the interior
    knots evenly, you place them yourself (e.g. to concentrate resolution
    near a spectral line core). ``positions`` are the interior breakpoints
    only — strictly increasing and strictly inside ``(x_min, x_max)`` — and
    the boundaries are clamped with multiplicity ``degree + 1`` at each end,
    exactly as in ``clamped_uniform_knots``. The resulting knot vector
    yields ``len(positions) + degree + 1`` B-spline coefficients, which is
    the ``num_coeffs`` to pass to :class:`~mlp_bspline.config.ChannelSplineConfig`
    alongside this knot vector.
    """
    positions = np.asarray(list(positions), dtype=np.float64)
    if positions.ndim != 1:
        raise ValueError("positions must be a 1-D sequence")
    if positions.size and (positions.min() <= x_min or positions.max() >= x_max):
        raise ValueError(
            f"positions must lie strictly inside (x_min={x_min}, x_max={x_max})"
        )
    if positions.size > 1 and np.any(np.diff(positions) <= 0):
        raise ValueError("positions must be strictly increasing")
    knots = np.concatenate(
        [np.full(degree + 1, x_min, dtype=np.float64),
         positions,
         np.full(degree + 1, x_max, dtype=np.float64)]
    )
    return knots


def bspline_design_matrix(grid: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    """Dense design matrix ``B`` of shape ``(len(grid), n_coeffs)``.

    ``grid`` need not be sorted; the result is returned in the original
    (unsorted) order of ``grid``.
    """
    grid = np.asarray(grid, dtype=np.float64)
    order = np.argsort(grid)
    x_sorted = grid[order]
    # Points must lie within [knots[degree], knots[-degree-1]] for a
    # clamped knot vector; clip tiny floating point overshoot at the edges.
    lo, hi = knots[degree], knots[-degree - 1]
    x_sorted = np.clip(x_sorted, lo, hi)
    dm = BSpline.design_matrix(x_sorted, knots, degree, extrapolate=False)
    dense_sorted = dm.toarray()
    dense = np.empty_like(dense_sorted)
    dense[order] = dense_sorted
    return dense


class BSplineProjector:
    """Precomputed B-spline design (``B``) and projection (``P``) matrices
    for a single channel evaluated on a single 1-D grid (depth or
    wavelength).
    """

    def __init__(
        self,
        grid: np.ndarray,
        num_coeffs: int,
        degree: int = 3,
        knots: Optional[np.ndarray] = None,
    ):
        self.grid = np.asarray(grid, dtype=np.float64)
        self.degree = degree
        if knots is None:
            knots = clamped_uniform_knots(
                float(self.grid.min()), float(self.grid.max()), num_coeffs, degree
            )
        self.knots = np.asarray(knots, dtype=np.float64)
        self.n = num_coeffs
        self.B = bspline_design_matrix(self.grid, self.knots, degree)  # (D, n)
        self.P = np.linalg.pinv(self.B)  # (n, D), least-squares projector

    @property
    def n_grid(self) -> int:
        return self.grid.shape[0]

    def project(self, f: np.ndarray) -> np.ndarray:
        """f: (..., D) -> coefficients (..., n)."""
        return f @ self.P.T

    def reconstruct(self, c: np.ndarray) -> np.ndarray:
        """c: (..., n) -> sampled function (..., D)."""
        return c @ self.B.T


class BSplineGroup:
    """A named collection of :class:`BSplineProjector`, one per channel
    (e.g. the 6 physical channels or the 4 Stokes channels), sharing a
    common grid but each with its own knot configuration.
    """

    def __init__(self, names: Sequence[str], grid: np.ndarray, channel_configs: Dict[str, "ChannelSplineConfig"]):
        missing = [n for n in names if n not in channel_configs]
        if missing:
            raise ValueError(f"missing spline config for channels: {missing}")
        self.names = list(names)
        self.grid = np.asarray(grid, dtype=np.float64)
        self.projectors: Dict[str, BSplineProjector] = {
            name: BSplineProjector(
                self.grid,
                channel_configs[name].num_coeffs,
                channel_configs[name].degree,
                channel_configs[name].knots,
            )
            for name in self.names
        }
        self.sizes: Dict[str, int] = {name: p.n for name, p in self.projectors.items()}
        self.total_size = sum(self.sizes.values())

    def project(self, array: np.ndarray) -> Dict[str, np.ndarray]:
        """array: (N, C, D) with C == len(self.names), channel order matching
        self.names. Returns dict name -> coefficients (N, n_name)."""
        return {
            name: self.projectors[name].project(array[:, i, :])
            for i, name in enumerate(self.names)
        }

    def reconstruct(self, coeffs: Dict[str, np.ndarray]) -> np.ndarray:
        """dict name -> (N, n_name) -> stacked array (N, C, D)."""
        outs = [self.projectors[name].reconstruct(coeffs[name]) for name in self.names]
        return np.stack(outs, axis=1)
