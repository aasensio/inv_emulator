"""Spatial point spread function of the telescope.

:class:`SpatialPSF` convolves maps of synthetic Stokes profiles ``(nx, ny,
...)`` with a 2-D PSF sampled on the pixel grid of the observations, for
all Stokes parameters and wavelengths at once, by FFT. The maps are padded
by the half-size of the PSF (reflecting them at the edges of the field of
view by default) before the FFT, so the circular convolution of the FFT
does not wrap around and the cost depends only weakly on the PSF size.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fft import next_fast_len


def load_psf(path: str, hdu: int = 0) -> np.ndarray:
    """PSF array from a FITS (``.fits``, ``.fit``, ``.fts``), HDF5 (dataset
    ``psf`` or the only dataset) or ``.npy`` file."""
    lower = path.lower()
    if lower.endswith((".fits", ".fit", ".fts", ".fits.gz")):
        from astropy.io import fits

        with fits.open(path) as f:
            return np.asarray(f[hdu].data, dtype=np.float64)
    if lower.endswith(".npy"):
        return np.load(path).astype(np.float64)
    import h5py

    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        key = "psf" if "psf" in f else keys[0] if len(keys) == 1 else None
        if key is None:
            raise ValueError(f"{path}: no dataset 'psf' among {keys}")
        return f[key][:].astype(np.float64)


def crop_psf(psf: np.ndarray, size: Union[int, Tuple[int, int]]) -> np.ndarray:
    """The central ``size`` (odd; one value or ``(sx, sy)``) pixels of a
    PSF around its peak, renormalized to unit sum."""
    psf = np.asarray(psf, dtype=np.float64)
    sx, sy = (size, size) if np.isscalar(size) else size
    if sx % 2 == 0 or sy % 2 == 0:
        raise ValueError(f"the cropped PSF size must be odd, got {(sx, sy)}")
    cx, cy = np.unravel_index(np.argmax(psf), psf.shape)
    hx, hy = sx // 2, sy // 2
    if cx - hx < 0 or cy - hy < 0 or cx + hx >= psf.shape[0] or cy + hy >= psf.shape[1]:
        raise ValueError(f"cannot crop {(sx, sy)} pixels around the peak {(cx, cy)} of a {psf.shape} PSF")
    out = psf[cx - hx:cx + hx + 1, cy - hy:cy + hy + 1]
    return out / out.sum()


def enclosed_energy(psf: np.ndarray, size: int) -> float:
    """Fraction of the PSF inside the central ``size x size`` pixels around
    its peak (to choose the size passed to :func:`crop_psf`)."""
    psf = np.asarray(psf, dtype=np.float64)
    cx, cy = np.unravel_index(np.argmax(psf), psf.shape)
    h = size // 2
    return float(psf[max(cx - h, 0):cx + h + 1, max(cy - h, 0):cy + h + 1].sum() / psf.sum())


class SpatialPSF(nn.Module):
    """2-D convolution of maps ``(nx, ny, ...)`` with a PSF, by FFT.

    psf:
        ``(kx, ky)`` array with odd sizes and the PSF centre in the central
        pixel (as stored for Hinode), on the pixel grid of the
        observations; normalized to unit sum here.
    size:
        Optional odd size (one value or ``(sx, sy)``) to which the PSF is
        cropped around its peak (see :func:`crop_psf`, :func:`enclosed_energy`).
    boundary:
        Padding of the maps beyond the field of view: ``"reflect"``,
        ``"replicate"`` or ``"circular"`` (periodic, no padding).
    """

    def __init__(self, psf: np.ndarray, size: Optional[Union[int, Tuple[int, int]]] = None,
                 boundary: str = "reflect"):
        super().__init__()
        psf = np.asarray(psf, dtype=np.float64)
        if psf.ndim != 2:
            raise ValueError(f"the spatial PSF must be a 2-D array, got shape {psf.shape}")
        if size is not None:
            psf = crop_psf(psf, size)
        if psf.shape[0] % 2 == 0 or psf.shape[1] % 2 == 0:
            raise ValueError(f"the spatial PSF must have odd sizes (centre in the central pixel), got {psf.shape}")
        if boundary not in ("reflect", "replicate", "circular"):
            raise ValueError(f"boundary must be 'reflect', 'replicate' or 'circular', got {boundary!r}")
        self.boundary = boundary
        self.half = (psf.shape[0] // 2, psf.shape[1] // 2)
        self.register_buffer("kernel", torch.as_tensor(psf / psf.sum(), dtype=torch.float32))
        self._otf: Dict[tuple, torch.Tensor] = {}

    def _transfer_function(self, shape: Tuple[int, int], device) -> torch.Tensor:
        """rFFT of the PSF zero-padded to ``shape`` with its centre at (0, 0)."""
        key = (shape, str(device))
        if key not in self._otf:
            kx, ky = self.kernel.shape
            if kx > shape[0] or ky > shape[1]:
                raise ValueError(f"PSF of {tuple(self.kernel.shape)} pixels larger than the padded maps {shape}")
            k = torch.zeros(shape, dtype=torch.float32, device=device)
            k[:kx, :ky] = self.kernel.to(device)
            k = torch.roll(k, shifts=(-self.half[0], -self.half[1]), dims=(0, 1))
            self._otf[key] = torch.fft.rfft2(k)
        return self._otf[key]

    def forward(self, maps: torch.Tensor) -> torch.Tensor:
        """``maps (nx, ny, ...)`` -> the same convolved with the PSF in the
        first two dimensions (all trailing ones, e.g. Stokes and
        wavelength, at once). Differentiable."""
        nx, ny = maps.shape[:2]
        rest = maps.shape[2:]
        x = maps.reshape(nx, ny, -1).permute(2, 0, 1)  # (C, nx, ny)
        hx, hy = self.half
        if self.boundary != "circular":
            mode = self.boundary
            if mode == "reflect" and (hx >= nx or hy >= ny):
                mode = "replicate"  # reflection needs more pixels than the PSF half-size
            x = F.pad(x[None], (hy, hy, hx, hx), mode=mode)[0]
            # further zero padding to sizes with small prime factors (fast
            # FFTs) does not wrap around into the cropped result
            shape = (next_fast_len(x.shape[1]), next_fast_len(x.shape[2]))
        else:
            shape = tuple(x.shape[1:])
        y = torch.fft.irfft2(torch.fft.rfft2(x, s=shape) * self._transfer_function(shape, x.device), s=shape)
        if self.boundary != "circular":
            y = y[:, hx:hx + nx, hy:hy + ny]
        return y.permute(1, 2, 0).reshape(nx, ny, *rest)
