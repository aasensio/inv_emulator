"""Telluric absorption lines, usable in any spectral region.

A terrestrial line multiplies the emergent Stokes I (Q, U and V are left
unchanged), in the observer's frame (not Doppler shifted with the Sun) and
before the instrumental profile:

    T(lambda) = 1 - d H(a, (lambda - lambda0) / delta_lambda)

with ``H(a, v) = Re w(v + i a)`` the Voigt function (``H(0, 0) = 1``, so
``d`` is the central depth for ``a = 0``; :func:`voigt`). Each line has four
per-pixel variables, named with its ``prefix`` (default ``"Tel_"``):
``<prefix>shift`` (``lambda0 - wavelength``, A; relative to the reference
``wavelength`` because an absolute wavelength has a resolution of ~1e-3 A
in float32), ``<prefix>width`` (``delta_lambda``, A), ``<prefix>depth``
(``d``) and ``<prefix>a`` (``a``). Give each line of an inversion its own
prefix (e.g. ``"O2a_"``, ``"O2b_"`` for the two O2 lines next to Fe I
6302.5), unless it is deliberately the same line seen in two regions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

TELLURIC_SUFFIXES = ("shift", "width", "depth", "a")


# --------------------------------------------------------------------------- #
# Voigt function
# --------------------------------------------------------------------------- #

def _weideman_coefficients(n: int = 32) -> Tuple[np.ndarray, float]:
    """Coefficients (highest degree first) and ``L`` of Weideman's (1994,
    SIAM J. Numer. Anal. 31, 1497) rational approximation of the Faddeeva
    function with ``n`` terms."""
    m = 2 * n
    k = np.arange(-m + 1, m)
    big_l = math.sqrt(n / math.sqrt(2.0))
    t = big_l * np.tan(k * math.pi / (2 * m))
    f = np.concatenate([[0.0], np.exp(-t * t) * (big_l * big_l + t * t)])
    a = np.real(np.fft.fft(np.fft.fftshift(f))) / (2 * m)
    return a[1:n + 1][::-1].copy(), big_l


_WEIDEMAN_A, _WEIDEMAN_L = _weideman_coefficients()


def voigt(a: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Voigt function ``H(a, v) = Re w(v + i a)`` (``a >= 0``), normalized
    to ``H(0, 0) = 1`` (``int H dv = sqrt(pi)``), differentiable in both
    arguments. Weideman's approximation with 32 terms: absolute error
    < 1e-13 in float64 (against ``scipy.special.wofz``), ~1e-6 in float32."""
    a, v = torch.broadcast_tensors(a, v)
    ctype = torch.complex128 if v.dtype == torch.float64 else torch.complex64
    iz = torch.complex(-a, v).to(ctype)                # i z, z = v + i a
    den = _WEIDEMAN_L - iz
    zz = (_WEIDEMAN_L + iz) / den
    p = torch.full_like(zz, float(_WEIDEMAN_A[0]))
    for c in _WEIDEMAN_A[1:]:
        p = p * zz + float(c)
    w = 2.0 * p / (den * den) + (1.0 / math.sqrt(math.pi)) / den
    return w.real


# --------------------------------------------------------------------------- #
# telluric lines
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TelluricLine:
    """A telluric line near ``wavelength`` (A, absolute, observer's frame),
    with per-pixel variables ``<prefix>shift``, ``<prefix>width``,
    ``<prefix>depth``, ``<prefix>a`` (see the module docstring)."""

    wavelength: float
    prefix: str = "Tel_"

    @property
    def variable_names(self) -> Tuple[str, ...]:
        return tuple(self.prefix + s for s in TELLURIC_SUFFIXES)

    def transmission(self, wave_rel: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """``1 - d H(a, (lambda - lambda0) / delta_lambda)``. ``wave_rel``:
        observer's-frame wavelengths relative to :attr:`wavelength`, ``(L,)``
        or ``(B, L)``; ``params``: variable name -> ``(B,)``. Returns ``(B, L)``."""
        p = self.prefix
        v = (wave_rel - params[p + "shift"][:, None]) / params[p + "width"][:, None]
        return 1.0 - params[p + "depth"][:, None] * voigt(params[p + "a"][:, None], v)


def as_telluric_lines(telluric) -> Tuple[TelluricLine, ...]:
    """``None``, a reference wavelength (prefix ``"Tel_"``), a
    :class:`TelluricLine`, or a sequence of them -> tuple of lines, checking
    that the prefixes are distinct."""
    if telluric is None:
        return ()
    if isinstance(telluric, (int, float, TelluricLine)):
        telluric = [telluric]
    lines = tuple(t if isinstance(t, TelluricLine) else TelluricLine(float(t)) for t in telluric)
    prefixes = [t.prefix for t in lines]
    if len(set(prefixes)) != len(prefixes):
        raise ValueError(f"telluric lines need distinct prefixes, got {prefixes}")
    return lines


def telluric_variables(line: TelluricLine, shift_range: float = 0.3, fixed: Sequence[str] = ()):
    """Default :class:`~mlp_bspline.inversion.VariableConfig` of each
    variable of ``line``: scales and bounds that cover typical telluric
    lines. ``fixed``: suffixes (e.g. ``("a",)``) of variables to keep at
    their initial value."""
    from .inversion import slab_variable

    p = line.prefix
    return {
        p + "shift": slab_variable(scale=0.01, bounds=(-shift_range, shift_range), fixed="shift" in fixed),
        p + "width": slab_variable(scale=0.01, bounds=(0.005, 0.3), fixed="width" in fixed),
        p + "depth": slab_variable(scale=0.05, bounds=(0.0, 1.0), fixed="depth" in fixed),
        p + "a": slab_variable(scale=0.1, bounds=(0.0, 2.0), fixed="a" in fixed),
    }


def telluric_initial_guess(stokes: np.ndarray, wave: np.ndarray, line: TelluricLine,
                           continuum: Optional[Tuple[float, float]] = None, search: float = 0.3,
                           width: float = 0.04, a: float = 0.5) -> Dict[str, np.ndarray]:
    """Per-pixel starting values ``name -> (nx, ny, 1)`` from observations
    ``stokes (nx, ny, 4, nl)`` on absolute wavelengths ``wave``: the position
    and depth of the I minimum within ``search`` A of the line, the depth
    relative to the 90th percentile of I in the ``continuum`` window (A,
    absolute; default: everything within ``3 * search`` A of the line except
    the search window), and a typical ``width`` (A) and damping ``a``."""
    nx, ny = stokes.shape[:2]
    flat = stokes.reshape(nx * ny, 4, -1)
    win = np.abs(wave - line.wavelength) <= search
    if win.sum() == 0:
        raise ValueError(f"no observed wavelength within {search} A of the telluric line at {line.wavelength} A")
    if continuum is None:
        cont = (np.abs(wave - line.wavelength) <= 3 * search) & ~win
    else:
        cont = (wave >= continuum[0]) & (wave <= continuum[1])
    i_win = np.nan_to_num(flat[:, 0, win], nan=np.inf)
    cont_i = np.nanpercentile(flat[:, 0, cont], 90, axis=1) if cont.any() else np.nanmax(flat[:, 0], axis=1)
    p = line.prefix
    shape = (nx, ny, 1)
    return {
        p + "shift": (wave[win][i_win.argmin(1)] - line.wavelength).reshape(shape),
        p + "depth": np.clip(1.0 - i_win.min(1) / cont_i, 0.02, 0.9).reshape(shape),
        p + "width": np.full(shape, width),
        p + "a": np.full(shape, a),
    }
