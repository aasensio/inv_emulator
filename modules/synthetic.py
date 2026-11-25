"""Synthetic data generator used for tests and the demo script only.

Produces plausible-shaped (but not physically accurate) atmospheric
stratifications and Stokes profiles so the training/inference pipeline can
be exercised end-to-end without real radiative-transfer data.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def generate_synthetic_dataset(
    n_models: int, n_depth: int = 40, n_wave: int = 80, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (physical [N,6,D], stokes [N,4,L], mu [N], depth_grid [D],
    wave_grid [L])."""
    rng = np.random.default_rng(seed)

    depth = np.linspace(-4.0, 1.0, n_depth)  # e.g. log(tau500)
    wave = np.linspace(-1.0, 1.0, n_wave)  # wavelength offset from line center

    mu = rng.uniform(0.2, 1.0, size=n_models)

    T = np.empty((n_models, n_depth))
    vz = np.empty((n_models, n_depth))
    vmic = np.empty((n_models, n_depth))
    Bx = np.empty((n_models, n_depth))
    By = np.empty((n_models, n_depth))
    Bz = np.empty((n_models, n_depth))

    for i in range(n_models):
        T[i] = 5000 + 3000 * rng.uniform(0.5, 1.5) * np.exp(-0.3 * (depth - rng.uniform(-3, 0)) ** 2) \
            + 4000 - 600 * depth
        vz[i] = rng.uniform(-2, 2) * np.sin(depth * rng.uniform(0.5, 1.5) + rng.uniform(0, 6.28))
        vmic[i] = rng.uniform(0.5, 2.0) + rng.uniform(0.2, 1.0) * np.exp(0.3 * depth)
        Bx[i] = rng.uniform(-200, 200) * np.exp(0.3 * depth)
        By[i] = rng.uniform(-200, 200) * np.exp(0.3 * depth)
        Bz[i] = rng.uniform(200, 1500) * np.exp(0.5 * depth)

    physical = np.stack([T, vz, vmic, Bx, By, Bz], axis=1)  # (N, 6, D)

    # crude synthetic "spectral line": absorption depth modulated by
    # temperature and vz shifts the line, B components drive polarization.
    T_ref = T[:, -1:]  # photospheric-ish reference
    vz_ref = vz.mean(axis=1, keepdims=True)
    Bx_ref = Bx.mean(axis=1, keepdims=True)
    By_ref = By.mean(axis=1, keepdims=True)
    Bz_ref = Bz.mean(axis=1, keepdims=True)
    mu_col = mu[:, None]

    line_depth = 0.6 * (T_ref / T_ref.max())
    line_width = 0.25 + 0.05 * (T_ref / T_ref.max())
    shift = 0.05 * vz_ref
    profile = np.exp(-((wave[None, :] - shift) ** 2) / (2 * line_width**2))
    I = 1.0 - line_depth * profile * (0.5 + 0.5 * mu_col)

    dprofile = -(wave[None, :] - shift) / (line_width**2) * profile
    V = 3e-4 * Bz_ref * dprofile * mu_col
    Q = 1e-4 * (Bx_ref**2 - By_ref**2) * profile * (1.0 - mu_col)
    U = 2e-4 * (Bx_ref * By_ref) * profile * (1.0 - mu_col)

    stokes = np.stack([I, Q, U, V], axis=1)  # (N, 4, L)

    return physical, stokes, mu, depth, wave
