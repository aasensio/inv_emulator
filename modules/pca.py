"""Principal-component (PCA) basis for the Stokes profiles, an alternative to
the B-spline basis of :mod:`mlp_bspline.bspline`.

Per channel (independently for I, Q, U, V):

    f(grid) ~ mean + B @ c      (reconstruction, B = (L, n) leading eigenvectors)
    c = P @ (f(grid) - mean)     (projection, P = B^T since B is orthonormal)

so the classes below expose the same ``B``/``P``/``project``/``reconstruct``
interface as :class:`~mlp_bspline.bspline.BSplineProjector` /
:class:`~mlp_bspline.bspline.BSplineGroup`, plus the ``mean`` profile, and
the rest of the pipeline (data preparation, scaling, emulator decode) is
unchanged.

Unlike the B-spline basis, the PCA basis is learned from data: it is fit on
the training split of whatever is actually being fit (i.e. after the
weak-field residual and/or the Stokes transform, if any). For a centered
data matrix ``X`` (N profiles x L wavelengths) the principal directions can
be obtained from either covariance-like matrix, ``X^T X`` (L x L) or the
Gram matrix ``X X^T`` (N x N), which share their non-zero eigenvalues; the
smaller of the two is diagonalized. With ``X X^T u = s u`` the corresponding
direction in wavelength space is ``v = X^T u / sqrt(s)``.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from .config import ChannelPCAConfig


class PCAProjector:
    """PCA basis for one channel on a 1-D grid, fit with :meth:`fit`."""

    def __init__(self, grid: np.ndarray, cfg: ChannelPCAConfig):
        self.grid = np.asarray(grid, dtype=np.float64)
        self.cfg = cfg
        self.mean = None
        self.B = None  # (L, n)
        self.P = None  # (n, L)
        self.n = None
        self.eigenvalues = None  # all non-zero ones, descending (variance per component)

    @property
    def n_grid(self) -> int:
        return self.grid.shape[0]

    def fit(self, f: np.ndarray, chunk_size: int = 65536) -> "PCAProjector":
        """f: (N, L) training profiles."""
        n_samples, n_grid = f.shape
        if n_grid != self.n_grid:
            raise ValueError(f"profiles have {n_grid} points, grid has {self.n_grid}")
        self.mean = f.mean(axis=0, dtype=np.float64) if self.cfg.center else np.zeros(n_grid)

        if n_grid <= n_samples:
            # L x L covariance, accumulated in chunks to avoid a centered copy of f
            cov = np.zeros((n_grid, n_grid))
            for start in range(0, n_samples, chunk_size):
                x = np.asarray(f[start:start + chunk_size], dtype=np.float64) - self.mean
                cov += x.T @ x
            evals, evecs = np.linalg.eigh(cov / n_samples)
            order = np.argsort(evals)[::-1]
            evals, evecs = evals[order], evecs[:, order]
        else:
            # N x N Gram matrix; map its eigenvectors back to wavelength space
            x = np.asarray(f, dtype=np.float64) - self.mean
            evals, u = np.linalg.eigh(x @ x.T)
            order = np.argsort(evals)[::-1]
            evals, u = evals[order], u[:, order]
            keep = evals > evals[0] * 1e-12
            evals, u = evals[keep], u[:, keep]
            evecs = (x.T @ u) / np.sqrt(evals)
            evals = evals / n_samples

        evals = np.clip(evals, 0.0, None)
        self.eigenvalues = evals
        n = self._num_components(evals)
        B = evecs[:, :n]
        # Fix the arbitrary sign of each eigenvector (largest |entry| positive)
        # so refits on the same data give the same basis.
        B = B * np.sign(B[np.abs(B).argmax(axis=0), np.arange(n)])
        self.B = B
        self.P = B.T.copy()
        self.n = n
        return self

    def _num_components(self, evals: np.ndarray) -> int:
        if self.cfg.num_components is not None:
            if self.cfg.num_components > len(evals):
                raise ValueError(f"num_components={self.cfg.num_components} but only {len(evals)} "
                                 "principal components exist")
            return self.cfg.num_components
        total = evals.sum()
        if total == 0:
            return 1
        return int(np.searchsorted(np.cumsum(evals) / total, self.cfg.variance_fraction) + 1)

    @property
    def explained_variance_fraction(self) -> float:
        return float(self.eigenvalues[:self.n].sum() / self.eigenvalues.sum())

    def project(self, f: np.ndarray) -> np.ndarray:
        """f: (..., L) -> coefficients (..., n)."""
        return (f - self.mean) @ self.P.T

    def reconstruct(self, c: np.ndarray) -> np.ndarray:
        """c: (..., n) -> sampled function (..., L)."""
        return c @ self.B.T + self.mean


class PCAGroup:
    """Drop-in replacement for :class:`~mlp_bspline.bspline.BSplineGroup`
    holding one :class:`PCAProjector` per channel; ``sizes`` is only known
    after :meth:`fit`."""

    def __init__(self, names: Sequence[str], grid: np.ndarray, channel_configs: Dict[str, ChannelPCAConfig]):
        missing = [n for n in names if n not in channel_configs]
        if missing:
            raise ValueError(f"missing PCA config for channels: {missing}")
        self.names = list(names)
        self.grid = np.asarray(grid, dtype=np.float64)
        self.projectors: Dict[str, PCAProjector] = {
            name: PCAProjector(self.grid, channel_configs[name]) for name in self.names
        }
        self.sizes: Dict[str, int] = {}
        self.total_size = 0

    def fit(self, array: np.ndarray) -> "PCAGroup":
        """array: (N, C, L) training profiles, channel order matching self.names."""
        for i, name in enumerate(self.names):
            self.projectors[name].fit(array[:, i, :])
        self.sizes = {name: p.n for name, p in self.projectors.items()}
        self.total_size = sum(self.sizes.values())
        return self

    def project(self, array: np.ndarray) -> Dict[str, np.ndarray]:
        return {name: self.projectors[name].project(array[:, i, :]) for i, name in enumerate(self.names)}

    def reconstruct(self, coeffs: Dict[str, np.ndarray]) -> np.ndarray:
        return np.stack([self.projectors[name].reconstruct(coeffs[name]) for name in self.names], axis=1)
