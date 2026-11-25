"""He I 10830 chromospheric slab on top of an emulated photosphere.

Emulator
--------
:class:`SiHeSynthesizer` takes either :class:`HeIEmulator` or the much
faster PCA emulator :class:`~mlp_bspline.hei_pca.HeIPCAEmulator` (same
interface, one MLP call per pixel instead of one per wavelength).

:class:`HeIEmulator` is the neural field of ``hei/`` (trained on Hazel2
syntheses, see ``hazel_neural/database/synth_all.py``): for the slab
parameters ``(Bx, By, Bz, deltav, a)`` (field in G in Hazel's *vertical*
frame, Doppler width in km/s, damping) and the geometry ``(height,
theta)`` (arcsec, LOS angle in degrees) it returns the emission vector
``eps (4)``, the absorption coefficients ``eta (4)`` (stimulated emission
included) and the magneto-optical coefficients ``rho (3)`` at any
wavelength of [10823, 10834] A. They were computed for ``v = 0``,
``tau = 1``, ``beta = 1`` and the Allen quiet-Sun illumination of the slab
(``nbar = 1``, ``omega = 0``).

Formal solution
---------------
As in Hazel2 (``synth.f90``, "slab case with exact solution") the slab has
constant properties, the optical depth ``tau`` is measured at the maximum
of ``eta_I`` (the centre of the red component), ``ds = tau / max(eta_I)``,
and, with ``K`` the propagation matrix,

    I_out = exp(-K ds) I_in + K^-1 (1 - exp(-K ds)) beta eps.

Both operators are computed analytically (:func:`slab_operators`), as
``evol_operator`` in Hazel2's maths.f90 (Landi Degl'Innocenti & Landolfi
2004, Eq. A5.20) does for ``exp(-K ds)``: the same decomposition gives
``(K ds)^-1 (1 - exp(-K ds))`` directly, without inverting ``K`` (which
would be ill-conditioned in single precision in the far wings, where
``rho >> eta_I``), and Taylor series handle the degenerate points
(unpolarized ``K``). It agrees with a matrix exponential to ~2e-7 in
float32 and ~1e-13 in float64, gradients included, and is 3-9x faster
(40-50x under ``torch.compile``). ``I_in`` is the emulated photospheric spectrum and everything is in
units of the quiet-Sun disk-centre continuum (Allen), as in Hazel:
``I_in`` is converted to absolute intensity with ``I0_allen(lambda, 1)``
and the result is divided by ``I0_allen(mean lambda, 1)``.

The LOS velocity ``v`` of the slab Doppler shifts ``eps``, ``eta`` and
``rho`` (the emulator is evaluated at ``lambda / (1 + v / c)``).

Pumping radiation
-----------------
Hazel2 scales the radiation field that illuminates the slab (``nbar``) by
``r = I_boundary(first wavelength) / I0_allen(10829.09, mu)``, so a dark
photosphere (umbra) gives a lower source function. The emulator was trained
with ``r = 1``. With ``pumping="boundary"`` (default) the source function is
scaled by ``r``, computed as Hazel does from the photospheric boundary
condition: a Hazel2 test (r = 0.5, 0.25) gives ``S_I = eps_I / max(eta_I)
proportional to r`` within 1%, while the polarization of eps and eta
(atomic polarization) changes by 6-10%, which this approximation misses.
``pumping="quiet"`` keeps ``r = 1``.

Telluric lines
--------------
With ``telluric=...`` (a reference wavelength, a
:class:`~mlp_bspline.telluric.TelluricLine` or a list of them) terrestrial
absorption lines multiply the emergent Stokes I, in the observer's frame
and before the instrumental profile; see :mod:`mlp_bspline.telluric`.

Several slabs
-------------
``prefix`` may list several slabs (e.g. ``("He1_", "He2_")``), ordered from
the photosphere outwards. They are traversed in series, as Hazel2's
``ph1->ch1->ch2`` topology: the first slab has the photospheric (Si I)
spectrum as boundary condition, every following one the Stokes vector
emerging from the previous slab, each with its own parameters, pumping
ratio (from its own boundary intensity, ``pumping="boundary"``) and, with
``hazel_boundary_tilt``, the same Allen tilt Hazel2 applies at every
boundary. Telluric lines act on the spectrum emerging from the last slab.
The emulator is called once for all slabs (stacked along the batch).

Variable names
--------------
The slab variables are ``<prefix>Bx``, ``<prefix>By``, ... (suffixes
:data:`HEI_SUFFIXES`, ``prefix="He_"`` by default, i.e.
:data:`HEI_VARIABLES`), so several slabs in one synthesizer or inversion can
be told apart. :func:`he_slab_variables` and :func:`he_slab_initial_guess`
(:func:`he_slabs_initial_guess` for several slabs) give default variable
configurations and starting values.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import WeakFieldConfig, WeakFieldLine
from .inversion import (
    C_KMS, SpectralSynthesizer, center_of_gravity_velocity, cubic_spline_interpolate, gaussian_psf,
    linear_interpolate, slab_variable, spline_second_derivative_matrix, weak_field_estimate,
)
from .telluric import TELLURIC_SUFFIXES, TelluricLine, as_telluric_lines, voigt  # noqa: F401 (voigt re-exported)

# Slab variables (after a prefix, "He_" by default), as passed to
# :meth:`SiHeSynthesizer.forward` (``slab`` dict)
HEI_SUFFIXES = ("Bx", "By", "Bz", "tau", "v", "deltav", "a", "beta", "height")
HEI_VARIABLES = tuple("He_" + n for n in HEI_SUFFIXES)
# Variables of a telluric line with the default prefix
TELLURIC_VARIABLES = tuple("Tel_" + n for n in TELLURIC_SUFFIXES)
# Centre of the (blended) red components of the triplet
HEI_LAMBDA_RED = 10830.30

# Wavelength range of the emulator's training grid (mapped to [-1, 1])
HEI_WAVE_RANGE = (10823.0, 10834.0)
# Hazel's reference wavelength of the multiplet (used for the pumping radiation)
HEI_LAMBDA0 = 10829.0911

# Parameter ranges of the emulator's training set
HEI_TRAINING_RANGE = {
    "He_deltav": (4.0, 18.0),
    "He_a": (0.0, 1.5),
    "He_height": (0.1, 10.0),
    "theta": (0.0, 89.9),
}

# symlog scalings of the training targets (hazel_neural/train/dataset.py)
_EPS_MAX = (3.6e-5, 2.4e-6, 2.4e-6, 1e-5)
_ETA_MAX = (1.0, 0.1, 0.1, 1.0)
_RHO_MAX = (0.1, 0.1, 1.0)
_EPS_FACTOR = (5e2, 1e5, 1e5, 1e5)
_ETA_FACTOR = (5e1, 5e6, 5e6, 1e5)
_RHO_FACTOR = (5e4, 5e4, 1e3)


def i0_allen(wavelength, mu):
    """Allen's quiet-Sun continuum intensity (cgs, per unit frequency) at
    ``wavelength`` (A) and ``mu`` (copy of ``hazel.util.i0_allen``)."""
    c = 2.99792458e10
    lambda_ic = 1e4 * np.asarray([0.20, 0.22, 0.245, 0.265, 0.28, 0.30, 0.32, 0.35, 0.37, 0.38, 0.40, 0.45, 0.50,
                                  0.55, 0.60, 0.80, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
    u_data = np.asarray([0.12, -1.3, -0.1, -0.1, 0.38, 0.74, 0.88, 0.98, 1.03, 0.92, 0.91, 0.99, 0.97, 0.93, 0.88,
                         0.73, 0.64, 0.57, 0.48, 0.35, 0.22, 0.15])
    v_data = np.asarray([0.33, 1.6, 0.85, 0.90, 0.57, 0.20, 0.03, -0.1, -0.16, -0.05, -0.05, -0.17, -0.22, -0.23,
                         -0.23, -0.22, -0.20, -0.21, -0.18, -0.12, -0.07, -0.07])
    lambda_i0 = 1e4 * np.asarray([0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.37, 0.38, 0.39, 0.40, 0.41,
                                  0.42, 0.43, 0.44, 0.45, 0.46, 0.48, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90,
                                  1.00, 1.10, 1.20, 1.40, 1.60, 1.80, 2.00, 2.50, 3.00, 4.00, 5.00, 6.00, 8.00, 10.0,
                                  12.0])
    i0 = np.asarray([0.06, 0.21, 0.29, 0.60, 1.30, 2.45, 3.25, 3.77, 4.13, 4.23, 4.63, 4.95, 5.15, 5.26, 5.28, 5.24,
                     5.19, 5.10, 5.00, 4.79, 4.55, 4.02, 3.52, 3.06, 2.69, 2.28, 2.03, 1.57, 1.26, 1.01, 0.81, 0.53,
                     0.36, 0.238, 0.160, 0.078, 0.041, 0.0142, 0.0062, 0.0032, 0.00095, 0.00035, 0.00018])
    i0 = i0 * 1e14 * (lambda_i0 * 1e-8) ** 2 / c
    u = np.interp(wavelength, lambda_ic, u_data)
    v = np.interp(wavelength, lambda_ic, v_data)
    return (1.0 - u - v + u * mu + v * mu ** 2) * np.interp(wavelength, lambda_i0, i0)


# --------------------------------------------------------------------------- #
# emulator (same architecture and state_dict keys as hei/mlp.py, encoding.py)
# --------------------------------------------------------------------------- #

class _GaussianEncoding(nn.Module):
    def __init__(self, encoding_size: int):
        super().__init__()
        self.register_buffer("B", torch.zeros(encoding_size, 1))

    def forward(self, v):
        vp = 2.0 * math.pi * v @ self.B.T
        return torch.cat((v, torch.cos(vp), torch.sin(vp)), dim=-1)


class _MLP(nn.Module):
    def __init__(self, n_input, n_output, dim_hidden, n_hidden):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(n_input, dim_hidden)]
                                    + [nn.Linear(dim_hidden, dim_hidden) for _ in range(n_hidden)])
        self.last_layer = nn.Linear(dim_hidden, n_output)

    def forward(self, x, scale, shift):
        for layer in self.layers:
            y = layer(x) * scale + shift
            x = x + y if layer.in_features == layer.out_features else y
            x = F.gelu(x, approximate="tanh")
        return self.last_layer(x)


class _MLPConditioning(nn.Module):
    def __init__(self, n_input, n_output, dim_hidden, n_hidden):
        super().__init__()
        layers = [nn.Linear(n_input, dim_hidden)]
        for _ in range(n_hidden):
            layers += [nn.Linear(dim_hidden, dim_hidden), nn.GELU(approximate="tanh")]
        self.layers = nn.ModuleList(layers)
        self.gamma = nn.Linear(dim_hidden, n_output)
        self.beta = nn.Linear(dim_hidden, n_output)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.gamma(x), self.beta(x)


def _inv_symlog(x):
    return torch.sign(x) * torch.expm1(torch.abs(x))


class HeIEmulator(nn.Module):
    """``eps, eta, rho`` of He I 10830 from the ``hei/`` neural field.

    checkpoint:
        ``hei/*.best.pth``.
    input_normalization:
        ``.npz`` with ``model_mean/std`` (Bx, By, Bz, deltav, a) and
        ``geom_mean/std`` (height, theta) of the training set, which the
        checkpoint does not store (``hei/input_normalization.npz``).
    """

    def __init__(self, checkpoint: str, input_normalization: str, map_location: str = "cpu"):
        super().__init__()
        chk = torch.load(checkpoint, map_location=map_location, weights_only=False)
        hp = chk["hyperparameters"]
        n_hidden = hp["mlp"]["n_hidden_mlp"]
        self.encoding = _GaussianEncoding(hp["embedding"]["encoding_size"])
        self.model = _MLP(2 * hp["embedding"]["encoding_size"] + 1, 11, n_hidden, hp["mlp"]["num_layers_mlp"])
        self.conditioning = _MLPConditioning(7, n_hidden, hp["conditioning"]["n_hidden"],
                                             hp["conditioning"]["num_layers"])
        self.encoding.load_state_dict(chk["encoding_dict"])
        self.model.load_state_dict(chk["model_dict"])
        self.conditioning.load_state_dict(chk["conditioning_dict"])
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

        norm = np.load(input_normalization)
        mean = np.concatenate([norm["model_mean"], norm["geom_mean"]]).astype(np.float32)
        std = np.concatenate([norm["model_std"], norm["geom_std"]]).astype(np.float32)
        self.register_buffer("in_mean", torch.as_tensor(mean))
        self.register_buffer("in_std", torch.as_tensor(std))

        out_mean = np.concatenate([np.asarray(m, dtype=np.float32) for m in chk["normalization"]["mean"]])
        out_std = np.concatenate([np.asarray(s, dtype=np.float32) for s in chk["normalization"]["std"]])
        out_scale = np.asarray(_EPS_MAX + _ETA_MAX + _RHO_MAX, dtype=np.float64) \
            / np.asarray(_EPS_FACTOR + _ETA_FACTOR + _RHO_FACTOR, dtype=np.float64)
        self.register_buffer("out_mean", torch.as_tensor(out_mean))
        self.register_buffer("out_std", torch.as_tensor(out_std))
        self.register_buffer("out_scale", torch.as_tensor(out_scale, dtype=torch.float32))

        self.wave_center = 0.5 * (HEI_WAVE_RANGE[0] + HEI_WAVE_RANGE[1])
        self.wave_half_width = 0.5 * (HEI_WAVE_RANGE[1] - HEI_WAVE_RANGE[0])

    def forward(self, pars: torch.Tensor, x: torch.Tensor):
        """``pars (B, 7)``: Bx, By, Bz (G, vertical frame), deltav (km/s), a,
        height (arcsec), theta (deg). ``x (B, L)``: rest-frame wavelengths
        mapped to the emulator's coordinate, ``(lambda - 10828.5) / 5.5``.
        Returns ``eps (B, 4, L)``, ``eta (B, 4, L)``, ``rho (B, 3, L)``."""
        # The checkpoint was trained with the two FiLM heads swapped
        # (``beta, gamma = conditioning(pars)`` in hazel_neural/train), so
        # the "gamma" head is the shift and the "beta" head the scale.
        shift, scale = self.conditioning((pars - self.in_mean) / self.in_std)
        out = self.model(self.encoding(x.unsqueeze(-1)), scale=scale[:, None, :], shift=shift[:, None, :])
        out = out * self.out_std + self.out_mean
        out = (_inv_symlog(out) * self.out_scale).transpose(1, 2)
        return out[:, 0:4], out[:, 4:8], out[:, 8:11]


# --------------------------------------------------------------------------- #
# formal solution
# --------------------------------------------------------------------------- #

def propagation_matrix(eta: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """``eta (..., 4, L)``, ``rho (..., 3, L)`` -> ``K (..., L, 4, 4)``, with
    Hazel's sign conventions (``fill_absorption_matrix`` in maths.f90)."""
    eI, eQ, eU, eV = eta.unbind(-2)
    rQ, rU, rV = rho.unbind(-2)
    rows = [
        torch.stack([eI, eQ, eU, eV], -1),
        torch.stack([eQ, eI, rV, -rU], -1),
        torch.stack([eU, -rV, eI, rQ], -1),
        torch.stack([eV, rU, -rQ, eI], -1),
    ]
    return torch.stack(rows, -2)


def _phi_derivatives(x, n_max=9):
    """``phi^(n)(x)``, n = 0..n_max, of ``phi(x) = (1 - exp(-x)) / x``: ``(-1)^n J_n`` with
    ``J_n = int_0^1 s^n exp(-s x) ds``. For ``x <= 4``, ``J_nmax`` from its positive series
    ``exp(-x) n! sum_m x^m / (n+m+1)!`` and the (stable) downward recursion
    ``J_{n-1} = (x J_n + exp(-x)) / n``; for ``x > 4`` the upward one, ``J_n = (n J_{n-1} - exp(-x)) / x``."""
    lo = x <= 4.0
    xs = torch.where(lo, x, torch.zeros_like(x))
    xl = torch.where(lo, torch.full_like(x, 5.0), x)
    term = torch.full_like(x, 1.0 / math.factorial(n_max + 1))
    acc = term
    for m in range(1, 24):
        term = term * xs / (n_max + m + 1)
        acc = acc + term
    es, el = torch.exp(-xs), torch.exp(-xl)
    down = [es * math.factorial(n_max) * acc]
    for n in range(n_max, 0, -1):
        down.append((xs * down[-1] + es) / n)
    down = down[::-1]
    up = [-torch.expm1(-xl) / xl]
    for n in range(1, n_max + 1):
        up.append((n * up[-1] - el) / xl)
    return [(-1) ** n * torch.where(lo, down[n], up[n]) for n in range(n_max + 1)]


def _phi(x):
    small = x.abs() < 1e-3
    xs = torch.where(small, torch.ones_like(x), x)
    return torch.where(small, 1 - x / 2 + x * x / 6, -torch.expm1(-xs) / xs)


def _poly(x, coefs):
    out = torch.full_like(x, coefs[-1])
    for c in coefs[-2::-1]:
        out = out * x + c
    return out


def _symmetric_coefficients(c, q, e1, e2, wy):
    """Template coefficients for small Theta from the series ``E(x) = sum c_k x^k`` and
    ``Q(x) = sum q_k x^k`` (``x = Lambda^2``), at the two roots ``x = L+^2, -L-^2`` of
    ``x^2 - e1 x + e2``, ``e1 = |w|^2 - |y|^2``, ``e2 = -(w.y)^2``: polynomials in ``e1, e2``
    (power sums ``p_k`` and complete homogeneous ``h_k``), smooth at the degenerate points."""
    p = [2.0, e1]
    h = [1.0, e1]
    for _ in range(2, len(c)):
        p.append(e1 * p[-1] - e2 * p[-2])
        h.append(e1 * h[-1] - e2 * h[-2])
    c1 = 0.5 * sum(ck * pk for ck, pk in zip(c, p))                         # (E_a + E_b) / 2
    cK = sum(qk * hk for qk, hk in zip(q, h))                               # p_a Q_a + p_b Q_b
    cT = -wy * sum(qk * hk for qk, hk in zip(q[1:], h))                     # (w.y)(Q_b - Q_a) / Theta
    c4 = sum(ck * hk for ck, hk in zip(c[1:], h))                           # (E_a - E_b) / Theta
    return c1, cK, cT, c4


def slab_operators(eta, rho, ds):
    """``exp(-A)`` and ``A^-1 (1 - exp(-A))`` for ``A = K ds`` (``eta (B, 4, L)``, ``rho (B, 3, L)``,
    ``ds (B,)`` -> two ``(B, L, 4, 4)``), analytically.

    Both are functions ``f(A)`` of a propagation matrix ``A = w0 1 + N`` (``w = ds (eta_Q, eta_U,
    eta_V)``, ``y = ds (rho_Q, rho_U, rho_V)``), whose eigenvalues are ``w0 +- Lambda_+`` and
    ``w0 +- i Lambda_-``. With ``g(x) = f(w0 + x)``, ``E_a = (g(L+) + g(-L+)) / 2``,
    ``E_b = Re g(i L-)``, ``Q_a = (g(L+) - g(-L+)) / (2 L+)``, ``Q_b = Im g(i L-) / L-``,
    ``Theta = L+^2 + L-^2``, ``p_a,b = L+-^2 / Theta``,

        f(A) = (E_a + E_b) / 2  1 + (p_a Q_a + p_b Q_b) N + (w.y) (Q_b - Q_a) / Theta  N~
               + (E_a - E_b) / Theta  M4

    (``N~``: dual of N, ``M4``: Landi Degl'Innocenti & Landolfi 2004, Eq. A5.20; ``evol_operator``
    in Hazel2's maths.f90 for ``f = exp(-x)``). The second operator is ``f = phi(x) = (1 - e^-x) / x``
    directly, not ``A^-1 (1 - exp(-A))``: the inverse is ill-conditioned in single precision where
    ``rho >> eta_I`` (line wings) or ``|eta_pol| ~ eta_I``. Small ``Lambda``'s and ``Theta`` use Taylor
    series in ``Lambda^2`` (exact and differentiable at the degenerate points ``Theta = 0``)."""
    w0 = eta[:, 0] * ds[:, None]                                  # (B, L)
    w = eta[:, 1:].transpose(1, 2) * ds[:, None, None]            # (B, L, 3)
    y = rho.transpose(1, 2) * ds[:, None, None]
    w1, w2, w3 = w.unbind(-1)
    y1, y2, y3 = y.unbind(-1)
    ww, yy, wy = (w * w).sum(-1), (y * y).sum(-1), (w * y).sum(-1)
    one, zero = torch.ones_like(w0), torch.zeros_like(w0)

    # u = Lambda_+^2 = t1 + half, v = Lambda_-^2 = t1 - half (the smaller one as (w.y)^2 / the larger)
    half = 0.5 * (ww - yy)
    t2 = half * half + wy * wy
    small = t2 < 0.0025                                           # Theta = 2 sqrt(t2) < 0.1: series below
    t1 = torch.sqrt(torch.where(small, one, t2))                  # (placeholders keep unused branches finite)
    big2 = t1 + half.abs()
    small2 = wy * wy / big2
    u = torch.where(half >= 0, big2, small2)
    v = torch.where(half >= 0, small2, big2)
    th = 2 * t1                                                   # Theta = u + v
    pa, pb, wy_th = u / th, v / th, wy / th
    # Lambda's only where they are not small (series in u, v otherwise)
    big_a, big_b = u > 1e-2, v > 1e-2
    a = torch.sqrt(torch.where(big_a, u, one))
    b = torch.sqrt(torch.where(big_b, v, one))

    # functions of Lambda_- through v for small Lambda_-
    cb = torch.where(big_b, torch.cos(b), _poly(-v, [1, 1 / 2, 1 / 24, 1 / 720, 1 / 40320]))
    sb = torch.where(big_b, torch.sinc(b / torch.pi), _poly(-v, [1, 1 / 6, 1 / 120, 1 / 5040, 1 / 362880]))
    sb2 = torch.where(big_b, torch.sinc(b / (2 * torch.pi)) ** 2,                       # sinc^2(L-/2)
                      _poly(-v, [1, 1 / 12, 1 / 360, 1 / 20160, 1 / 1814400]))

    # ---- O = exp(-A) ----
    e0 = torch.exp(-w0)
    em, ep = torch.exp(-(w0 - a)), torch.exp(-(w0 + a))           # e0 e^(+-L+), finite for large L+
    cha = torch.where(big_a, 0.5 * (em + ep), e0 * _poly(u, [1, 1 / 2, 1 / 24, 1 / 720, 1 / 40320]))
    sha = torch.where(big_a, (em - ep) / (2 * a), e0 * _poly(u, [1, 1 / 6, 1 / 120, 1 / 5040, 1 / 362880]))
    sha2 = torch.where(big_a, (cha - e0) * 2 / torch.where(big_a, u, one),
                       e0 * _poly(u, [1, 1 / 12, 1 / 360, 1 / 20160, 1 / 1814400]))     # e0 sinhc^2(L+/2)
    g1 = 0.5 * (cha + e0 * cb)
    gK = -(pa * sha + pb * e0 * sb)
    gT = wy_th * (sha - e0 * sb)
    g4 = 0.5 * (pa * sha2 + pb * e0 * sb2)

    # ---- P = phi(A) ----
    dphi = _phi_derivatives(w0)
    fm, fp = _phi(w0 - a), _phi(w0 + a)
    Ea = torch.where(big_a, 0.5 * (fm + fp),
                     dphi[0] + u * (dphi[2] / 2 + u * (dphi[4] / 24 + u * (dphi[6] / 720 + u * dphi[8] / 40320))))
    zz = w0 * w0 + v
    zs = torch.where(zz > 0, zz, one)
    nr = -torch.expm1(-w0) * cb + 0.5 * v * sb2                   # Re(1 - exp(-w0 - i L-))
    Eb = torch.where(zz > 0, (nr * w0 + e0 * sb * v) / zs, one)   # Re phi(w0 + i L-)

    def ser_q(x):                                                 # Q for small Lambda, x = +-Lambda^2
        return dphi[1] + x * (dphi[3] / 6 + x * (dphi[5] / 120 + x * (dphi[7] / 5040 + x * dphi[9] / 362880)))

    sa_, sb_ = u > 0.09, v > 0.09
    Qa = torch.where(sa_, (fp - fm) / (2 * a), ser_q(u))
    Qb = torch.where(sb_, (e0 * sb * w0 - nr) / zs, ser_q(-v))    # Im phi(w0 + i L-) / L-
    P1 = 0.5 * (Ea + Eb)
    PK = pa * Qa + pb * Qb
    PT = wy_th * (Qb - Qa)
    P4 = (Ea - Eb) / th

    # ---- small Theta: symmetric polynomials (no sqrt, exact gradients at Theta = 0) ----
    e1, e2 = ww - yy, -wy * wy
    fact = [math.factorial(k) for k in range(12)]
    o1, oK, oT, o4 = _symmetric_coefficients([e0 / fact[2 * k] for k in range(6)],
                                             [-e0 / fact[2 * k + 1] for k in range(6)], e1, e2, wy)
    q1, qK, qT, q4 = _symmetric_coefficients([dphi[2 * k] / fact[2 * k] for k in range(5)],
                                             [dphi[2 * k + 1] / fact[2 * k + 1] for k in range(5)], e1, e2, wy)
    g1, gK, gT, g4 = (torch.where(small, x, y) for x, y in ((o1, g1), (oK, gK), (oT, gT), (o4, g4)))
    P1, PK, PT, P4 = (torch.where(small, x, y) for x, y in ((q1, P1), (qK, PK), (qT, PT), (q4, P4)))

    # ---- matrices ----
    N = torch.stack([torch.stack([zero, w1, w2, w3], -1),
                     torch.stack([w1, zero, y3, -y2], -1),
                     torch.stack([w2, -y3, zero, y1], -1),
                     torch.stack([w3, y2, -y1, zero], -1)], -2)
    Nt = torch.stack([torch.stack([zero, -y1, -y2, -y3], -1),
                      torch.stack([-y1, zero, w3, -w2], -1),
                      torch.stack([-y2, -w3, zero, w1], -1),
                      torch.stack([-y3, w2, -w1, zero], -1)], -2)
    tt = 0.5 * (ww + yy)
    c1, c2, c3 = w3 * y2 - w2 * y3, w1 * y3 - w3 * y1, w2 * y1 - w1 * y2
    m12, m13, m23 = w1 * w2 + y1 * y2, w3 * w1 + y3 * y1, w2 * w3 + y2 * y3
    M4 = torch.stack([torch.stack([tt, c1, c2, c3], -1),
                      torch.stack([-c1, w1 * w1 + y1 * y1 - tt, m12, m13], -1),
                      torch.stack([-c2, m12, w2 * w2 + y2 * y2 - tt, m23], -1),
                      torch.stack([-c3, m13, m23, w3 * w3 + y3 * y3 - tt], -1)], -2)
    eye = torch.eye(4, dtype=w0.dtype, device=w0.device)
    off = gK[..., None, None] * N + gT[..., None, None] * Nt + g4[..., None, None] * M4
    O = g1[..., None, None] * eye + off
    P = P1[..., None, None] * eye + PK[..., None, None] * N + PT[..., None, None] * Nt + P4[..., None, None] * M4
    return O, P


def slab_formal_solution(eps: torch.Tensor, eta: torch.Tensor, rho: torch.Tensor, ds: torch.Tensor,
                         stokes_in: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Exact solution through a constant slab of geometrical thickness
    ``ds (B,)`` (in units of 1 / eta). ``eps (B, 4, L)``, ``eta (B, 4, L)``,
    ``rho (B, 3, L)``, ``stokes_in (B, 4, L)`` (same units as ``eps / eta``),
    ``beta (B,)`` -> ``(B, 4, L)``."""
    O, P = slab_operators(eta, rho, ds)  # exp(-A), A^-1 (1 - exp(-A))
    return slab_step(O, P, eps * (beta * ds)[:, None, None], stokes_in)


def slab_step(O: torch.Tensor, P: torch.Tensor, src: torch.Tensor, stokes_in: torch.Tensor) -> torch.Tensor:
    """``O stokes_in + P src`` with the operators of :func:`slab_operators`
    ``(B, L, 4, 4)``, ``src = beta ds eps (B, 4, L)`` and ``stokes_in (B, 4, L)``."""
    s_in = stokes_in.transpose(1, 2).unsqueeze(-1)  # (B, L, 4, 1)
    return (O @ s_in + P @ src.transpose(1, 2).unsqueeze(-1)).squeeze(-1).transpose(1, 2)


# --------------------------------------------------------------------------- #
# Si I + He I synthesizer
# --------------------------------------------------------------------------- #

class SiHeSynthesizer(nn.Module):
    """Photospheric emulator(s) + He I slab(s) -> Stokes profiles at the
    observed wavelengths. ``chromosphere`` is a :class:`HeIEmulator` or a
    :class:`~mlp_bspline.hei_pca.HeIPCAEmulator`. Drop-in replacement for
    :class:`~mlp_bspline.inversion.SpectralSynthesizer` in
    :class:`~mlp_bspline.inversion.Inverter`, whose ``slab`` variables
    (:data:`HEI_VARIABLES`) it consumes.

    The photospheric spectrum (emulator grid, Doppler shifted by ``v_ref``)
    is interpolated with a cubic spline (on the Si I emulator's 0.026 A grid,
    tested against Hazel syntheses with the velocity added to the model, it
    keeps the Q/U/V peaks within ~1% and the I core within ~1e-4 for shifts
    of -2.3..3.1 km/s; linear interpolation flattens the peaks by 2-10%) to
    a uniform synthesis grid of step ``step``
    (A) over ``wave_range`` (A, default: the photospheric emulators' range),
    the He I slab(s) are added there, the result is convolved with the
    instrumental profile ``psf`` (as in SpectralSynthesizer) and linearly
    interpolated to the observed wavelengths.

    hazel_boundary_tilt:
        Multiply the boundary condition of every slab by ``I0_allen(lambda,
        1) / I0_allen(mean lambda, 1)`` (a +-0.1% slope across the window),
        as Hazel2 does when it feeds an atmosphere into a chromosphere.
    pumping:
        ``"boundary"`` scales each slab's source function with the
        intensity of its boundary condition (photosphere or previous slab)
        at the first wavelength of the synthesis grid (which must be
        continuum), as Hazel2 does; ``"quiet"`` uses the quiet-Sun
        illumination the emulator was trained with (see the module
        docstring).
    telluric:
        ``None``, the reference wavelength (A, e.g. 10832.1) of a telluric
        Voigt line with the default prefix ``"Tel_"``, a
        :class:`~mlp_bspline.telluric.TelluricLine`, or a list of them; their
        variables are then also read from ``slab``. Each must lie inside the
        synthesis grid.
    prefix:
        Prefix of the slab variables (default ``"He_"``: ``He_Bx``, ...), or
        a sequence of distinct prefixes, one per slab, ordered from the
        photosphere outwards (e.g. ``("He1_", "He2_")``: the Si I spectrum
        illuminates ``He1_``, whose emergent spectrum illuminates ``He2_``).
    """

    def __init__(self, photosphere: SpectralSynthesizer, chromosphere: nn.Module, step: float = 0.02,
                 wave_range: Optional[Sequence[float]] = None, psf=None, hazel_boundary_tilt: bool = True,
                 pumping: str = "boundary", telluric=None, prefix: Union[str, Sequence[str]] = "He_"):
        super().__init__()
        if pumping not in ("boundary", "quiet"):
            raise ValueError(f"pumping must be 'boundary' or 'quiet', got {pumping!r}")
        self.pumping = pumping
        self.photosphere = photosphere
        self.chromosphere = chromosphere
        self.phys_names = photosphere.phys_names
        self.depth_size = photosphere.depth_size
        self.prefixes = (prefix,) if isinstance(prefix, str) else tuple(prefix)
        if not self.prefixes or len(set(self.prefixes)) != len(self.prefixes):
            raise ValueError(f"give one or more distinct slab prefixes, got {prefix!r}")
        self.prefix = self.prefixes[0]
        self.tellurics = as_telluric_lines(telluric)
        self.slab_names = (tuple(p + n for p in self.prefixes for n in HEI_SUFFIXES)
                           + sum((t.variable_names for t in self.tellurics), ()))
        if len(set(self.slab_names)) != len(self.slab_names):
            raise ValueError(f"slab and telluric variable names clash: {self.slab_names}")

        lo, hi = wave_range if wave_range is not None else (photosphere.wave64[0], photosphere.wave64[-1])
        lo, hi = max(lo, HEI_WAVE_RANGE[0]), min(hi, HEI_WAVE_RANGE[1])
        n = int(round((hi - lo) / step)) + 1
        wave = np.linspace(lo, hi, n)
        self.wave64 = wave
        self.wave_ref = photosphere.wave_ref
        self.register_buffer("wave_rel", torch.as_tensor(wave - self.wave_ref, dtype=torch.float32))
        for t in self.tellurics:
            if not wave[0] < t.wavelength < wave[-1]:
                raise ValueError(f"telluric reference {t.wavelength} A outside the synthesis grid "
                                 f"[{wave[0]:.3f}, {wave[-1]:.3f}] A")
        # rest-frame emulator coordinate = (wave_ref + rel - centre) / half_width
        self.x_offset = self.wave_ref - chromosphere.wave_center
        self.register_buffer("phot_d2", torch.as_tensor(
            spline_second_derivative_matrix(photosphere.wave64 - self.wave_ref), dtype=torch.float32))

        mean_wave = 0.5 * (photosphere.wave64[0] + photosphere.wave64[-1])
        tilt = i0_allen(wave, 1.0) / i0_allen(mean_wave, 1.0) if hazel_boundary_tilt else np.ones_like(wave)
        self.register_buffer("tilt", torch.as_tensor(tilt, dtype=torch.float32))
        # eps is in absolute units: normalize to the quiet-Sun disk-centre continuum
        self.i0 = float(i0_allen(mean_wave, 1.0))
        # I0_allen(HEI_LAMBDA0, mu) = i0_line * (a0 + a1 mu + a2 mu^2), for the pumping ratio
        i0_line = i0_allen(HEI_LAMBDA0, 1.0)
        c1 = i0_allen(HEI_LAMBDA0, 0.5) / i0_line
        c2 = i0_allen(HEI_LAMBDA0, 0.25) / i0_line
        # solve (1 - u - v) + u mu + v mu^2 through mu = 1, 0.5, 0.25
        m = np.array([[1.0, 1.0, 1.0], [1.0, 0.5, 0.25], [1.0, 0.25, 0.0625]])
        self.clv = tuple(np.linalg.solve(m, [1.0, c1, c2]))
        self.i0_line = float(i0_line)

        kernel = None
        if isinstance(psf, dict):
            kernel = gaussian_psf(psf["fwhm"], step)
        elif psf is not None:
            dl, prof = (np.asarray(a, dtype=np.float64) for a in psf)
            half = int(np.floor(max(-dl.min(), dl.max()) / step))
            grid = np.arange(-half, half + 1) * step
            kernel = np.interp(grid, dl, prof, left=0.0, right=0.0)
            kernel = kernel / kernel.sum()
        self.register_buffer("psf", None if kernel is None else torch.as_tensor(kernel, dtype=torch.float32))

    def relative_wavelengths(self, wave_obs: np.ndarray) -> np.ndarray:
        return np.asarray(wave_obs, dtype=np.float64) - self.wave_ref

    @staticmethod
    def _shifted(wave_rel: torch.Tensor, ref: float, v: torch.Tensor) -> torch.Tensor:
        """Rest-frame wavelengths (relative to ``ref``) of the grid seen from
        a source moving at ``v (B,)`` km/s: ``lambda / (1 + v / c)``."""
        b = v[:, None] / C_KMS
        return (wave_rel[None, :] - ref * b) / (1.0 + b)

    def chromosphere_coefficients(self, slab: Dict[str, torch.Tensor], mu: torch.Tensor,
                                  wave_rel: torch.Tensor, prefixes: Optional[Sequence[str]] = None):
        """``eps, eta, rho`` of the slabs ``prefixes`` (default: all, in
        order) on the grid ``wave_rel`` (relative to ``wave_ref``), each
        Doppler shifted by its ``slab['<prefix>v']``, with one emulator call:
        ``(n_slabs * B, ., L)``, slab ``k`` in rows ``k * B .. (k + 1) * B``."""
        prefixes = self.prefixes if prefixes is None else prefixes
        theta = torch.rad2deg(torch.arccos(mu.clamp(-1.0, 1.0)))
        pars = torch.cat([torch.stack([slab[p + "Bx"], slab[p + "By"], slab[p + "Bz"], slab[p + "deltav"],
                                       slab[p + "a"], slab[p + "height"], theta], dim=1) for p in prefixes])
        v = torch.cat([slab[p + "v"] for p in prefixes])
        x = (self.x_offset + self._shifted(wave_rel, self.wave_ref, v)) / self.chromosphere.wave_half_width
        return self.chromosphere(pars, x)

    def synthesize(self, physical, mu, v_ref, slab):
        """Emergent Stokes ``(B, 4, L)`` on ``self.wave_rel`` and the mask of
        the points covered by the (shifted) photospheric emulators."""
        s_phot = self.photosphere.synthesize(physical, mu)
        xq = self._shifted(self.wave_rel, self.wave_ref, v_ref)
        x = self.photosphere.wave_rel
        _, valid = linear_interpolate(x, s_phot[:, :1], xq)
        out, _ = cubic_spline_interpolate(x, s_phot, self.phot_d2, xq.clamp(float(x[0]), float(x[-1])))
        # the operators of all slabs at once (they do not depend on the
        # incoming radiation), then the slabs in series from the photosphere
        eps, eta, rho = self.chromosphere_coefficients(slab, mu, self.wave_rel)
        tau = torch.cat([slab[p + "tau"] for p in self.prefixes])
        ds = tau / eta[:, 0].amax(dim=-1)
        O, P = slab_operators(eta, rho, ds)
        src = eps * (ds / self.i0)[:, None, None]
        n = mu.shape[0]
        for k, p in enumerate(self.prefixes):
            out = out * self.tilt
            beta = slab[p + "beta"]
            if self.pumping == "boundary":
                beta = beta * self.pumping_ratio(out[:, 0, 0], mu)
            rows = slice(k * n, (k + 1) * n)
            out = slab_step(O[rows], P[rows], src[rows] * beta[:, None, None], out)
        if self.tellurics:
            out = torch.cat([out[:, :1] * self.telluric_transmission(slab)[:, None, :], out[:, 1:]], dim=1)
        return out, valid

    def telluric_transmission(self, slab: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Product of the transmissions of the telluric lines, ``(B, L)`` on
        the synthesis grid (observer's frame)."""
        out = 1.0
        for t in self.tellurics:
            out = out * t.transmission(self.wave_rel + (self.wave_ref - t.wavelength), slab)
        return out

    def pumping_ratio(self, intensity: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """Hazel's ``nbar`` scaling from the boundary intensity (in units of
        the disk-centre continuum at the window's mean wavelength)."""
        a0, a1, a2 = self.clv
        return intensity * self.i0 / (self.i0_line * (a0 + a1 * mu + a2 * mu ** 2))

    def forward(self, physical, mu, v_ref, wave_obs_rel, slab):
        s, valid = self.synthesize(physical, mu, v_ref, slab)
        if self.psf is not None:
            b, c, n = s.shape
            k = self.psf.flip(0)[None, None]
            pad = (self.psf.shape[0] - 1) // 2
            s = F.conv1d(F.pad(s.reshape(b * c, 1, n), (pad, pad), mode="replicate"), k).reshape(b, c, n)
        s = torch.cat([s, valid.to(s.dtype)[:, None, :]], dim=1)
        out, inside = linear_interpolate(self.wave_rel, s, wave_obs_rel[None, :].expand(s.shape[0], -1))
        return out[:, :4], inside & (out[:, 4] > 1.0 - 1e-4)


# --------------------------------------------------------------------------- #
# defaults for inversions
# --------------------------------------------------------------------------- #

def he_slab_variables(prefix: str = "He_", fixed: Sequence[str] = ("beta", "height"),
                      field_bound: float = 1500.0) -> Dict[str, "VariableConfig"]:
    """Default :class:`~mlp_bspline.inversion.VariableConfig` of each slab
    variable: scales of the typical size of each parameter and bounds that
    keep the slab inside the emulator's training ranges (``deltav`` 4-18
    km/s, ``a`` 0-1.5, ``height`` 0.1-10 arcsec), field components within
    +-``field_bound`` G (Hazel's vertical frame). ``fixed``: suffixes of the
    variables kept at their initial value (by default the source-function
    factor ``beta`` and the ``height``)."""
    b = (-field_bound, field_bound)
    cfg = {
        "Bx": (100.0, b), "By": (100.0, b), "Bz": (100.0, b), "tau": (0.3, (0.01, 5.0)), "v": (1.0, (-20.0, 20.0)),
        "deltav": (1.0, HEI_TRAINING_RANGE["He_deltav"]), "a": (0.1, HEI_TRAINING_RANGE["He_a"]),
        "beta": (0.1, (0.0, 5.0)), "height": (0.5, HEI_TRAINING_RANGE["He_height"]),
    }
    unknown = set(fixed) - set(cfg)
    if unknown:
        raise ValueError(f"unknown slab variables in fixed: {sorted(unknown)}; expected suffixes of {HEI_SUFFIXES}")
    return {prefix + k: slab_variable(scale=sc, bounds=bd, fixed=k in fixed) for k, (sc, bd) in cfg.items()}


def he_slab_initial_guess(stokes: np.ndarray, wave: np.ndarray, mu, prefix: str = "He_", height: float = 3.0,
                          continuum: Tuple[float, float] = (10831.3, 10832.5), deltav: float = 8.0,
                          a: float = 0.3, beta: float = 1.0, geff_red: float = 1.13,
                          bperp_floor: float = 30.0, field_bound: float = 1500.0) -> Dict[str, np.ndarray]:
    """Per-pixel starting values ``name -> (nx, ny, 1)`` of the slab from
    observations ``stokes (nx, ny, 4, nl)`` on absolute wavelengths ``wave``
    (A) and ``mu`` (scalar or ``(nx, ny)``):

    - velocity: centre of gravity of the red component;
    - ``tau``: ``-log`` of its depth relative to the ``continuum`` window;
    - field: weak-field approximation of the red component (textbook Q/U
      sign, effective Lande factor ``geff_red`` = 1.13 of the red blend,
      calibrated on Hazel2 syntheses), the transverse part raised to
      ``bperp_floor`` G (Bx = By = 0 is a stationary point of Q and U),
      rotated from the LOS to Hazel's vertical frame with ``mu``;
    - ``deltav``, ``a``, ``beta``, ``height``: the given constants.
    """
    nx, ny = stokes.shape[:2]
    flat = stokes.reshape(nx * ny, 4, -1)
    shape = (nx, ny, 1)
    mu_map = np.broadcast_to(np.asarray(mu, dtype=np.float64), (nx, ny)).ravel()
    red = np.abs(wave - HEI_LAMBDA_RED) < 0.25
    cont = (wave > continuum[0]) & (wave < continuum[1])
    if red.sum() == 0 or cont.sum() == 0:
        raise ValueError("the observations must cover the He I red component and the continuum window")
    depth = np.nanmin(flat[:, 0, red], axis=1) / np.nanmedian(flat[:, 0, cont], axis=1)
    wfa = WeakFieldConfig(lines=[WeakFieldLine(lambda0=HEI_LAMBDA_RED, geff=geff_red, Geff=geff_red ** 2,
                                               wave_range=(HEI_LAMBDA_RED - 0.5, HEI_LAMBDA_RED + 0.5))],
                          linear_polarization_sign=1.0)
    bx, by, bz = weak_field_estimate(flat, wave, wfa)
    bperp = np.maximum(np.hypot(bx, by), bperp_floor)
    chi = np.arctan2(by, bx)
    bx, by = bperp * np.cos(chi), bperp * np.sin(chi)
    m, s = mu_map, np.sqrt(1.0 - mu_map ** 2)
    p = prefix
    return {
        p + "v": center_of_gravity_velocity(flat, wave, [(HEI_LAMBDA_RED, 0.35)]).reshape(shape),
        p + "tau": np.clip(-np.log(np.clip(depth, 1e-3, None)), 0.1, 3.0).reshape(shape),
        p + "Bx": np.clip(bx * m + bz * s, -field_bound, field_bound).reshape(shape),
        p + "By": np.clip(by, -field_bound, field_bound).reshape(shape),
        p + "Bz": np.clip(-bx * s + bz * m, -field_bound, field_bound).reshape(shape),
        p + "deltav": np.full(shape, deltav),
        p + "a": np.full(shape, a),
        p + "beta": np.full(shape, beta),
        p + "height": np.full(shape, height),
    }


def he_slabs_initial_guess(stokes: np.ndarray, wave: np.ndarray, mu, prefixes: Sequence[str] = ("He_",),
                           velocity_offsets: Optional[Sequence[float]] = None, **kwargs) -> Dict[str, np.ndarray]:
    """:func:`he_slab_initial_guess` for several slabs in series (``prefixes``
    ordered from the photosphere outwards, as in :class:`SiHeSynthesizer`):
    every slab starts from the single-slab estimate with ``tau`` divided
    among them (so the total opacity matches the observed depth) and its
    velocity shifted by ``velocity_offsets`` (km/s, one per slab; default
    ``linspace(-2, 2, n)`` for ``n > 1``), which breaks the symmetry
    between otherwise identical slabs. ``kwargs``: those of
    :func:`he_slab_initial_guess`."""
    n = len(prefixes)
    if velocity_offsets is None:
        velocity_offsets = np.linspace(-2.0, 2.0, n) if n > 1 else [0.0]
    if len(velocity_offsets) != n:
        raise ValueError(f"give one velocity offset per slab ({n}), got {len(velocity_offsets)}")
    one = he_slab_initial_guess(stokes, wave, mu, prefix="", **kwargs)
    out = {}
    for p, dv in zip(prefixes, velocity_offsets):
        out |= {p + k: v.copy() for k, v in one.items()}
        out[p + "tau"] = one["tau"] / n
        out[p + "v"] = one["v"] + dv
    return out
