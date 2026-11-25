"""Pixel-by-pixel (optionally spatially regularized) inversion of Stokes
observations with one or several trained emulators.

Parameterization
----------------
Each physical variable (``T``, ``vz``, ``vmic``, ``Bx``, ``By``, ``Bz``) is
described in every pixel by its values at a few nodes in log(tau) (its own
number and positions per variable). A fixed interpolating B-spline of
degree ``min(3, n_nodes - 1)`` through the nodes gives the stratification on
the emulators' depth grid (1 node = constant, 2 = linear, >= 4 = cubic);
outside the outermost nodes the stratification is held constant.

LOS velocity
------------
The emulators were trained with ``vz(log tau = -1) = 0``, so they cannot
produce a bulk shift of the lines. The ``vz`` nodes are *absolute*
velocities; the stratification ``vz(tau)`` built from them is split as
``v_ref + dv(tau)`` with ``v_ref = vz(log tau = -1)``: the emulators get
``dv`` (which vanishes at log tau = -1, as in training) and ``v_ref``
Doppler-shifts the synthetic spectrum, ``S_obs(l) = S(l / (1 + v_ref /
c))``. Positive velocities are redshifts (downflows), as in SIR/Hazel.

Spectral synthesis
------------------
The emulators' outputs are concatenated on their (absolute) wavelength
grids, optionally convolved with an instrumental profile, and linearly
interpolated to the observed wavelengths (shifted by ``v_ref``). Observed
points outside the synthetic range (after the shift) or non-finite in the
data are ignored.

Several spectral regions
------------------------
:class:`MultiRegionInverter` fits several :class:`Region` s (e.g. Fe I 6302
with :class:`SpectralSynthesizer` and Si I + He I 10830 with
:class:`~mlp_bspline.hei.SiHeSynthesizer`) observed on the same pixel grid
with one set of variables: every region builds its stratification from the
same nodes (on its own emulators' depth grid, with its own velocity
reference) and reads the per-pixel variables its synthesizer names, so
equal names are shared and different names (prefixes) are not. Each region
has its own noise, Stokes weights, PSF, telluric lines and weight; the data
term of a pixel is the weighted sum of the regions' chi2 per point. See
``invert_multiregion.py``. :class:`Inverter` is the single-region case.

Spatial PSF
-----------
A region (or :class:`Inverter`) may have a spatial PSF
(:class:`~mlp_bspline.spatial_psf.SpatialPSF`): the synthetic Stokes maps of
the whole field of view are then convolved with it, at all wavelengths at
once by FFT, before they are compared with the observations. Since the PSF
couples pixels of different chunks, the gradient is then computed in two
passes when there are several chunks: the maps are synthesized chunk by
chunk without autograd graph, the gradient of the data term with respect to
the full maps is computed, and it is back-propagated chunk by chunk through
a second synthesis (about 1.5 times the cost of an iteration without PSF).
With a single chunk the graph is kept and there is one pass.

Optimization
------------
The objective is

    J = sum_pixels chi2_pixel + sum_variables sum_regularizers weight * R(node maps)
    chi2_pixel = sum_{Stokes s} w_s sum_l ((synthetic - observed) / sigma_s)^2

minimized over the node values with Adam (or SGD) plus a proximal step for
the (non-smooth) spatial regularizers after every gradient step, followed
by a projection onto box bounds. The parameters are the node values divided
by a per-variable ``scale``. The proximal step size of each node map is
Adam's current effective step size ``lr / (sqrt(v_hat) + eps)`` averaged
over the map (``lr`` for SGD), converted to physical units, so that the
fixed point of the iteration approximates the minimizer of ``J`` and the
regularization weights keep the meaning they have in ``J``. Several
regularizers on one variable are applied one after the other (an
approximation to the prox of their sum). ``InversionConfig.transverse_regularization``
regularizes pairs ``(Bx, By)`` jointly, invariant under the 180-degree
ambiguity (see :class:`~mlp_bspline.regularization.TransverseRegularizer`). Gradients are accumulated over
chunks of pixels, so any field of view fits in memory, while the
regularization always acts on the full maps.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import make_interp_spline

from .config import WeakFieldConfig, WeakFieldLine
from .emulator import load_emulator
from .regularization import Regularizer, TransverseRegularizer
from .spatial_psf import SpatialPSF
from .telluric import as_telluric_lines
from .weak_field import weak_field_stokes
import time

C_KMS = 299792.458
PHYSICAL_VARIABLES = ("T", "vz", "vmic", "Bx", "By", "Bz")
STOKES = ("I", "Q", "U", "V")


# --------------------------------------------------------------------------- #
# nodes -> stratification
# --------------------------------------------------------------------------- #

def node_interpolation_matrix(grid: np.ndarray, nodes: Sequence[float]) -> np.ndarray:
    """``M (D, n)`` such that ``stratification = M @ node_values``: the
    interpolating B-spline of degree ``min(3, n - 1)`` through the nodes,
    constant beyond the outermost nodes."""
    grid = np.asarray(grid, dtype=np.float64)
    nodes = np.asarray(nodes, dtype=np.float64)
    if nodes.ndim != 1 or len(nodes) == 0:
        raise ValueError("nodes must be a non-empty 1-D sequence")
    if len(nodes) > 1 and np.any(np.diff(nodes) <= 0):
        raise ValueError(f"nodes must be strictly increasing, got {nodes}")
    if len(nodes) == 1:
        return np.ones((len(grid), 1))
    spline = make_interp_spline(nodes, np.eye(len(nodes)), k=min(3, len(nodes) - 1))
    return spline(np.clip(grid, nodes[0], nodes[-1]))


def interpolate_in_depth(x, logtau: np.ndarray, profile: np.ndarray) -> np.ndarray:
    """``np.interp`` for a depth grid in either order (model atmospheres
    usually run from the deepest point upwards, i.e. decreasing log tau)."""
    logtau = np.asarray(logtau, dtype=np.float64)
    order = np.argsort(logtau)
    return np.interp(x, logtau[order], np.asarray(profile, dtype=np.float64)[order])


# --------------------------------------------------------------------------- #
# forward model
# --------------------------------------------------------------------------- #

@dataclass
class LineEmulator:
    """A trained emulator and the absolute wavelengths (A) of its output
    grid (the grid it was trained on, which checkpoints without a
    weak-field baseline do not store)."""

    checkpoint: str
    wave: np.ndarray


def gaussian_psf(fwhm: float, step: float) -> np.ndarray:
    """Gaussian instrumental profile of the given FWHM (A) sampled with the
    given step (A), out to 4 sigma, normalized to unit sum."""
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    half = max(1, int(math.ceil(4.0 * sigma / step)))
    x = np.arange(-half, half + 1) * step
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def linear_interpolate(x: torch.Tensor, y: torch.Tensor, xq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``y (B, C, n)`` sampled at increasing ``x (n,)`` -> values at ``xq (B,
    m)``, differentiable in ``y`` and ``xq``; also returns the mask of
    ``xq`` inside ``[x[0], x[-1]]``."""
    n = x.shape[0]
    idx = torch.searchsorted(x, xq.contiguous()).clamp(1, n - 1)
    x0, x1 = x[idx - 1], x[idx]
    w = ((xq - x0) / (x1 - x0)).unsqueeze(1)
    gi = idx.unsqueeze(1).expand(-1, y.shape[1], -1)
    y0 = torch.gather(y, 2, gi - 1)
    y1 = torch.gather(y, 2, gi)
    valid = (xq >= x[0]) & (xq <= x[-1])
    return y0 + w * (y1 - y0), valid


def spline_second_derivative_matrix(x: np.ndarray) -> np.ndarray:
    """``D (n, n)`` mapping values at the knots ``x`` to the second
    derivatives of their (not-a-knot) cubic interpolating spline."""
    from scipy.interpolate import CubicSpline

    x = np.asarray(x, dtype=np.float64)
    return CubicSpline(x, np.eye(len(x)), axis=0).derivative(2)(x)


def cubic_spline_interpolate(x: torch.Tensor, y: torch.Tensor, d2: torch.Tensor,
                             xq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """As :func:`linear_interpolate`, with the cubic interpolating spline of
    ``y (B, C, n)`` at ``x (n,)``; ``d2`` from
    :func:`spline_second_derivative_matrix`."""
    n = x.shape[0]
    m = y @ d2.T  # second derivatives at the knots
    idx = torch.searchsorted(x, xq.contiguous()).clamp(1, n - 1)
    x0, x1 = x[idx - 1], x[idx]
    h = (x1 - x0).unsqueeze(1)
    a = ((x1 - xq) / (x1 - x0)).unsqueeze(1)
    b = 1.0 - a
    gi = idx.unsqueeze(1).expand(-1, y.shape[1], -1)
    y0, y1 = torch.gather(y, 2, gi - 1), torch.gather(y, 2, gi)
    m0, m1 = torch.gather(m, 2, gi - 1), torch.gather(m, 2, gi)
    out = a * y0 + b * y1 + ((a ** 3 - a) * m0 + (b ** 3 - b) * m1) * h ** 2 / 6.0
    valid = (xq >= x[0]) & (xq <= x[-1])
    return out, valid


class SpectralSynthesizer(nn.Module):
    """Stratifications ``(B, 6, D)`` (with ``dv``, i.e. ``vz(log tau=-1) =
    0``), ``mu (B,)`` and ``v_ref (B,)`` -> Stokes profiles ``(B, 4, nl)`` at
    the observed wavelengths.

    psf:
        ``None``; ``{"fwhm": x}`` for a Gaussian of FWHM ``x`` A; or a pair
        ``(dlambda, profile)`` of arrays (A offsets and values), interpolated
        to the synthetic grid step. Requires a uniform synthetic grid.
    telluric:
        ``None``, a reference wavelength (A), a
        :class:`~mlp_bspline.telluric.TelluricLine` or a list of them:
        telluric lines multiplying Stokes I in the observer's frame, before
        the PSF, whose variables are read from ``slab`` (:attr:`slab_names`).
    """

    def __init__(self, lines: Sequence[LineEmulator], psf=None, map_location: str = "cpu", telluric=None):
        super().__init__()
        emulators, waves = [], []
        for line in lines:
            em, _ = load_emulator(line.checkpoint, map_location=map_location)
            em.eval()
            for p in em.parameters():
                p.requires_grad_(False)
            wave = np.asarray(line.wave, dtype=np.float64)
            n_out = em.B_stokes[em.stokes_names[0]].shape[0]
            if len(wave) != n_out:
                raise ValueError(f"{line.checkpoint}: emulator outputs {n_out} wavelengths, "
                                 f"but {len(wave)} were given")
            if list(em.stokes_names) != list(STOKES):
                raise ValueError(f"{line.checkpoint}: Stokes channels {em.stokes_names}, expected {STOKES}")
            if emulators and list(em.phys_names) != list(emulators[0].phys_names):
                raise ValueError("all emulators must use the same physical channels")
            emulators.append(em)
            waves.append(wave)
        order = np.argsort([w[0] for w in waves])
        self.emulators = nn.ModuleList([emulators[i] for i in order])
        wave = np.concatenate([waves[i] for i in order])
        if np.any(np.diff(wave) <= 0):
            raise ValueError("the emulators' wavelength grids overlap or are not increasing")
        self.phys_names = list(self.emulators[0].phys_names)
        self.depth_size = self.emulators[0].P_phys[self.phys_names[0]].shape[1]
        # Wavelengths are handled relative to a reference inside the window:
        # in float32 an absolute ~6300 A has a resolution of ~5e-4 A (~0.02
        # km/s), which would quantize the Doppler shift.
        self.wave_ref = float(0.5 * (wave[0] + wave[-1]))
        self.register_buffer("wave_rel", torch.as_tensor(wave - self.wave_ref, dtype=torch.float32))
        self.wave64 = wave

        kernel = None
        if psf is not None:
            step = np.diff(wave)
            if not np.allclose(step, step.mean(), rtol=1e-3):
                raise ValueError("a PSF requires a uniform synthetic wavelength grid")
            step = step.mean()
            if isinstance(psf, dict):
                kernel = gaussian_psf(psf["fwhm"], step)
            else:
                dl, prof = (np.asarray(a, dtype=np.float64) for a in psf)
                half = int(np.floor(max(-dl.min(), dl.max()) / step))
                grid = np.arange(-half, half + 1) * step
                kernel = np.interp(grid, dl, prof, left=0.0, right=0.0)
                kernel = kernel / kernel.sum()
        self.register_buffer("psf", None if kernel is None else torch.as_tensor(kernel, dtype=torch.float32))

        self.tellurics = as_telluric_lines(telluric)
        for t in self.tellurics:
            if not wave[0] < t.wavelength < wave[-1]:
                raise ValueError(f"telluric reference {t.wavelength} A outside the synthesis grid "
                                 f"[{wave[0]:.3f}, {wave[-1]:.3f}] A")
        # per-pixel variables read from `slab` in forward()
        self.slab_names = sum((t.variable_names for t in self.tellurics), ())

    def synthesize(self, physical: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """Concatenated emulator output ``(B, 4, L_syn)`` on ``self.wave``."""
        return torch.cat([em(physical, mu) for em in self.emulators], dim=-1)

    def relative_wavelengths(self, wave_obs: np.ndarray) -> np.ndarray:
        """Observed absolute wavelengths (A) -> the relative ones that
        :meth:`forward` expects (computed in float64)."""
        return np.asarray(wave_obs, dtype=np.float64) - self.wave_ref

    def forward(self, physical: torch.Tensor, mu: torch.Tensor, v_ref: torch.Tensor,
                wave_obs_rel: torch.Tensor, slab=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """``wave_obs_rel``: observed wavelengths relative to ``wave_ref``
        (see :meth:`relative_wavelengths`). ``slab``: per-pixel variables
        (only those of the telluric lines are used)."""
        s = self.synthesize(physical, mu)
        beta = v_ref[:, None] / C_KMS
        if self.tellurics:
            # the rest-frame grid seen by the observer: l (1 + b), relative to each line
            obs_frame = self.wave_rel[None, :] * (1.0 + beta) + self.wave_ref * beta
            trans = 1.0
            for t in self.tellurics:
                trans = trans * t.transmission(obs_frame + (self.wave_ref - t.wavelength), slab)
            s = torch.cat([s[:, :1] * trans[:, None, :], s[:, 1:]], dim=1)
        if self.psf is not None:
            b, c, n = s.shape
            k = self.psf.flip(0)[None, None]
            pad = (self.psf.shape[0] - 1) // 2
            s = torch.nn.functional.conv1d(
                torch.nn.functional.pad(s.reshape(b * c, 1, n), (pad, pad), mode="replicate"), k
            ).reshape(b, c, n)
        # observed wavelength l corresponds to rest-frame l / (1 + v/c); with
        # l = ref + r:  l / (1 + b) - ref = (r - ref * b) / (1 + b)
        xq = (wave_obs_rel[None, :] - self.wave_ref * beta) / (1.0 + beta)
        return linear_interpolate(self.wave_rel, s, xq)


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #

@dataclass
class VariableConfig:
    """Nodes, scaling, bounds and regularization of one physical variable.

    nodes:
        log(tau) positions of the nodes.
    scale:
        Typical size of the variable (K, km/s, G): the optimizer works on
        ``value / scale``, so ``lr`` is a step in units of ``scale``.
    bounds:
        ``None`` or ``(lower, upper)``, each a scalar or one value per node,
        in physical units; enforced by projection after every step.
    regularization:
        Spatial regularizers (:class:`~mlp_bspline.regularization.Regularizer`)
        of the node maps, in physical units.
    fixed:
        Keep the variable at its initial value (not optimized).
    depth:
        ``False`` for a variable with a single value per pixel that is
        passed to the synthesizer as is (e.g. the parameters of a
        chromospheric slab, see :func:`slab_variable`); ``nodes`` is then
        ignored.
    """

    nodes: Sequence[float]
    scale: float
    bounds: Optional[Tuple] = None
    regularization: Sequence[Regularizer] = ()
    fixed: bool = False
    depth: bool = True

    def __post_init__(self):
        self.nodes = tuple(float(n) for n in self.nodes) if self.depth else (0.0,)
        if self.scale <= 0:
            raise ValueError(f"scale must be > 0, got {self.scale}")


def slab_variable(scale: float, bounds: Optional[Tuple] = None, regularization: Sequence[Regularizer] = (),
                  fixed: bool = False) -> VariableConfig:
    """A variable with one value per pixel (no depth dependence)."""
    return VariableConfig((), scale, bounds, regularization, fixed, depth=False)


@dataclass
class InversionConfig:
    """Variables and optimizer settings. ``noise`` and ``stokes_weights``
    are the defaults of every :class:`Region` that does not set its own;
    ``reference_depth`` is used by the single-region :class:`Inverter`
    (each :class:`Region` has its own). ``transverse_regularization`` lists
    :class:`~mlp_bspline.regularization.TransverseRegularizer` of pairs of
    variables ``(Bx, By)``, which then must not have regularizers of their
    own."""

    variables: Dict[str, VariableConfig]
    stokes_weights: Dict[str, float] = field(default_factory=lambda: {"I": 1.0, "Q": 1.0, "U": 1.0, "V": 1.0})
    noise: Dict[str, float] = field(default_factory=lambda: {"I": 1e-3, "Q": 1e-3, "U": 1e-3, "V": 1e-3})
    optimizer: str = "adam"  # "adam" or "sgd"
    lr: float = 0.02
    lr_final_fraction: float = 0.05  # cosine decay from lr to lr * lr_final_fraction
    n_iter: int = 500
    chunk_size: int = 65536  # pixels per forward/backward pass
    reference_depth: float = -1.0  # log(tau) where the emulators have vz = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every: int = 25
    transverse_regularization: Sequence[TransverseRegularizer] = ()

    def __post_init__(self):
        missing = [v for v in PHYSICAL_VARIABLES if v not in self.variables]
        if missing:
            raise ValueError(f"missing variable configs: {missing}")
        for tr in self.transverse_regularization:
            for name in (tr.bx, tr.by):
                if name not in self.variables:
                    raise ValueError(f"transverse regularization of unknown variable {name!r}")
                if self.variables[name].regularization:
                    raise ValueError(f"{name!r} has its own regularization and a transverse one")
            if len(self.variables[tr.bx].nodes) != len(self.variables[tr.by].nodes):
                raise ValueError(f"{tr.bx!r} and {tr.by!r} must have the same number of nodes")
        if self.optimizer not in ("adam", "sgd", "adamw"):
            raise ValueError(f"optimizer must be 'adam' or 'sgd', got {self.optimizer!r}")


@dataclass
class InversionResult:
    nodes: Dict[str, np.ndarray]           # variable -> (nx, ny, n_nodes), physical units (vz absolute)
    node_positions: Dict[str, np.ndarray]  # variable -> (n_nodes,) log(tau)
    stratification: Dict[str, np.ndarray]  # depth variable -> (nx, ny, D) on the emulators' depth grid (vz absolute)
    logtau: np.ndarray                     # (D,)
    v_ref: np.ndarray                      # (nx, ny) vz at log(tau) = reference_depth, km/s
    synthetic: np.ndarray                  # (nx, ny, 4, nl) (convolved with the spatial PSF, if any)
    chi2: np.ndarray                       # (nx, ny) per-pixel chi2 (weighted, / number of valid points)
    history: Dict[str, list]
    synthetic_nopsf: Optional[np.ndarray] = None  # (nx, ny, 4, nl) before the spatial PSF (if any)


# --------------------------------------------------------------------------- #
# default photospheric variables
# --------------------------------------------------------------------------- #

# columns of the model atmospheres of the training database
MODEL_COLUMNS = {"logtau": 0, "T": 1, "vmic": 3, "vz": 4, "Bx": 5, "By": 6, "Bz": 7}
DEFAULT_SCALES = {"T": 500.0, "vz": 1.0, "vmic": 0.5, "Bx": 200.0, "By": 200.0, "Bz": 200.0}


@dataclass
class AtmosphereStatistics:
    """Depth grid and, per physical variable, the median and the lower/upper
    percentile stratifications of a set of model atmospheres (velocity at
    the reference depth removed, as in training)."""

    logtau: np.ndarray
    median: Dict[str, np.ndarray]
    lower: Dict[str, np.ndarray]
    upper: Dict[str, np.ndarray]

    def bounds(self, name: str, nodes: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Per-node ``(lower, upper)`` at the nodes' heights."""
        return (interpolate_in_depth(nodes, self.logtau, self.lower[name]),
                interpolate_in_depth(nodes, self.logtau, self.upper[name]))


def training_statistics(models_file: str, percentiles: Tuple[float, float] = (0.1, 99.9),
                        reference_index: int = 22, max_models: Optional[int] = None) -> AtmosphereStatistics:
    """:class:`AtmosphereStatistics` of the models in an HDF5 file of the
    training database (dataset ``model (N, D, 8)``: logtau, T, Pe, vmic, v,
    Bx, By, Bz), with ``v`` relative to depth index ``reference_index``
    (log tau = -1), as the emulators were trained."""
    import h5py

    with h5py.File(models_file, "r") as f:
        models = f["model"][:max_models]
    models[:, :, MODEL_COLUMNS["vz"]] -= models[:, reference_index:reference_index + 1, MODEL_COLUMNS["vz"]]
    cols = {k: c for k, c in MODEL_COLUMNS.items() if k != "logtau"}
    return AtmosphereStatistics(
        logtau=models[0, :, 0].astype(np.float64),
        median={k: np.median(models[:, :, c], axis=0) for k, c in cols.items()},
        lower={k: np.percentile(models[:, :, c], percentiles[0], axis=0) for k, c in cols.items()},
        upper={k: np.percentile(models[:, :, c], percentiles[1], axis=0) for k, c in cols.items()},
    )


def photosphere_variables(nodes: Dict[str, Sequence[float]], stats: AtmosphereStatistics,
                          scales: Optional[Dict[str, float]] = None,
                          regularization: Optional[Dict[str, Sequence[Regularizer]]] = None,
                          fixed: Sequence[str] = (), vz_bounds: Tuple[float, float] = (-20.0, 20.0),
                          positive_bx: bool = True) -> Dict[str, VariableConfig]:
    """:class:`VariableConfig` of ``T``, ``vz``, ``vmic``, ``Bx``, ``By``,
    ``Bz`` at the given ``nodes`` (log tau, per variable), bounded by the
    training percentiles at each node's height so the emulators never
    extrapolate, except ``vz`` (absolute velocities, ``vz_bounds``). With
    ``positive_bx`` the lower bound of ``Bx`` is 0, which removes the
    180-degree ambiguity of the transverse field. ``scales`` and
    ``regularization`` (variable -> list of regularizers) override
    :data:`DEFAULT_SCALES` / no regularization; ``fixed`` lists variables
    kept at their initial value."""
    scales = dict(DEFAULT_SCALES, **(scales or {}))
    regularization = regularization or {}
    missing = [v for v in PHYSICAL_VARIABLES if v not in nodes]
    if missing:
        raise ValueError(f"missing nodes for {missing}")
    out = {}
    for name in PHYSICAL_VARIABLES:
        if name == "vz":
            bounds = vz_bounds
        else:
            lo, hi = stats.bounds(name, nodes[name])
            if name == "vmic":
                lo = np.maximum(lo, 0.0)
            if name == "Bx" and positive_bx:
                lo = np.zeros_like(lo)
            bounds = (lo, hi)
        out[name] = VariableConfig(nodes[name], scale=scales[name], bounds=bounds,
                                   regularization=list(regularization.get(name, ())), fixed=name in fixed)
    return out


# --------------------------------------------------------------------------- #
# initial guess
# --------------------------------------------------------------------------- #

def center_of_gravity_velocity(stokes: np.ndarray, wave: np.ndarray,
                               lines: Sequence[Tuple[float, float]]) -> np.ndarray:
    """LOS velocity (km/s, + = redshift) from the centre of gravity of I in
    windows ``(lambda0, half_width)``, averaged over the lines.
    ``stokes (N, 4, nl)``."""
    vs = []    
    for l0, hw in lines:
        m = np.abs(wave - l0) <= hw
        if m.sum() < 3:
            continue
        I = stokes[:, 0, m]
        d = np.clip(np.nanmax(I, axis=1, keepdims=True) - I, 0.0, None)
        d = np.nan_to_num(d)
        lc = (d * wave[m]).sum(1) / np.clip(d.sum(1), 1e-30, None)
        vs.append(C_KMS * (lc / l0 - 1.0))
    if not vs:
        return np.zeros(stokes.shape[0])
    return np.mean(vs, axis=0)


def weak_field_estimate(stokes: np.ndarray, wave: np.ndarray, cfg: WeakFieldConfig) -> Tuple[np.ndarray, ...]:
    """Per-pixel least-squares weak-field ``(Bx, By, Bz)`` (G) from I, Q, U, V
    ``(N, 4, nl)`` on absolute wavelengths ``wave`` (A); the line windows in
    ``cfg`` are absolute wavelengths too."""
    s = np.nan_to_num(np.asarray(stokes, dtype=np.float64))
    n = s.shape[0]
    one, zero = np.ones(n), np.zeros(n)
    _, _, v_basis = weak_field_stokes(s[:, 0], wave, zero, zero, one, cfg)   # V for Bz = 1 G
    q_basis, _, _ = weak_field_stokes(s[:, 0], wave, one, zero, zero, cfg)   # Q for Bx^2 - By^2 = 1 G^2
    bz = (s[:, 3] * v_basis).sum(1) / np.clip((v_basis ** 2).sum(1), 1e-30, None)
    a = (s[:, 1] * q_basis).sum(1) / np.clip((q_basis ** 2).sum(1), 1e-30, None)   # Bx^2 - By^2
    c = (s[:, 2] * q_basis).sum(1) / np.clip((q_basis ** 2).sum(1), 1e-30, None)   # 2 Bx By
    bperp2 = np.sqrt(a ** 2 + c ** 2)
    chi = 0.5 * np.arctan2(c, a)
    bperp = np.sqrt(bperp2)
    return bperp * np.cos(chi), bperp * np.sin(chi), bz


def initial_guess(
    observations: np.ndarray,
    wave_obs: np.ndarray,
    config: InversionConfig,
    logtau: np.ndarray,
    mean_stratification: Dict[str, np.ndarray],
    wfa: Optional[WeakFieldConfig] = None,
    cog_lines: Sequence[Tuple[float, float]] = (),
    bperp_floor: float = 30.0,
) -> Dict[str, np.ndarray]:
    """Starting node values ``variable -> (nx, ny, n_nodes)``:

    - ``T``, ``vmic`` (and anything not listed below): the mean
      stratification interpolated at the nodes, in every pixel (for
      per-pixel variables, ``mean_stratification[name]`` is a scalar).
    - ``vz``: centre-of-gravity velocity of I in the ``cog_lines`` windows
      (0 if none), at every node.
    - ``Bx``, ``By``, ``Bz``: weak-field estimate (if ``wfa`` is given), at
      every node. The transverse field is raised to at least
      ``bperp_floor`` G, since ``Bx = By = 0`` is a stationary point of Q
      and U (both depend on them quadratically) from which the optimizer
      could not escape.
    """
    nx, ny = observations.shape[:2]
    flat = observations.reshape(nx * ny, *observations.shape[2:])
    init = {}
    for name, vc in config.variables.items():
        prof = mean_stratification.get(name)
        if not vc.depth:
            val = np.full(1, 0.0 if prof is None else float(prof))
        elif prof is not None:
            val = interpolate_in_depth(vc.nodes, logtau, prof)
        else:
            val = np.zeros(len(vc.nodes))
        init[name] = np.broadcast_to(val, (nx, ny, len(vc.nodes))).copy()

    v = center_of_gravity_velocity(flat, wave_obs, cog_lines)
    init["vz"][:] = v.reshape(nx, ny, 1)
    if wfa is not None:
        bx, by, bz = weak_field_estimate(flat, wave_obs, wfa)

        # AAR: Choose the solution with Bx>0 to break the 180-degree ambiguity of the transverse field. The optimizer will then be able to explore the full range of azimuths, but it will not be able to cross the Bx=0 line (which is a stationary point of Q and U).
        ind = bx < 0
        bx = np.where(ind, -bx, bx)
        by = np.where(ind, -by, by)        

        bperp = np.hypot(bx, by)
        chi = np.arctan2(by, bx)
        
        bperp = np.maximum(bperp, bperp_floor)
        
        for name, b in (("Bx", bperp * np.cos(chi)), ("By", bperp * np.sin(chi)), ("Bz", bz)):
            init[name][:] = b.reshape(nx, ny, 1)
        
        
    return init


# --------------------------------------------------------------------------- #
# inversion
# --------------------------------------------------------------------------- #

def _per_node(value, n: int, name: str) -> np.ndarray:
    a = np.asarray(value, dtype=np.float64)
    if a.ndim == 0:
        return np.full(n, float(a))
    if a.shape != (n,):
        raise ValueError(f"{name}: expected a scalar or {n} values, got shape {a.shape}")
    return a


class _DepthGeometry:
    """Node -> stratification matrices of the depth variables on one depth
    grid, and the index of the velocity reference depth (``None``: the
    emulators take absolute velocities and there is no Doppler shift)."""

    def __init__(self, variables: Dict[str, VariableConfig], logtau: np.ndarray, reference_depth: Optional[float],
                 device):
        self.logtau = np.asarray(logtau, dtype=np.float64)
        self.ref_index = None if reference_depth is None else int(np.argmin(np.abs(self.logtau - reference_depth)))
        self.M = {
            name: torch.as_tensor(node_interpolation_matrix(self.logtau, vc.nodes), dtype=torch.float32, device=device)
            for name, vc in variables.items() if vc.depth
        }

    def inputs(self, nodes: Dict[str, torch.Tensor], phys_names: Sequence[str]):
        """Physical node values ``name -> (B, n)`` -> emulator input ``(B, C,
        D)`` (``vz`` relative to the reference depth if there is one), ``v_ref
        (B,)`` and the per-pixel variables ``name -> (B,)``."""
        strat = {name: nodes[name] @ M.T for name, M in self.M.items()}
        if self.ref_index is None:
            v_ref = torch.zeros_like(strat["vz"][:, 0])
        else:
            v_ref = strat["vz"][:, self.ref_index]
            strat["vz"] = strat["vz"] - v_ref[:, None]
        slab = {name: nodes[name][:, 0] for name in nodes if name not in self.M}
        return torch.stack([strat[n] for n in phys_names], dim=1), v_ref, slab


@dataclass
class Region:
    """One spectral region of an inversion: its observations and the
    synthesizer that models them.

    name:
        Label used in the logs and the output file.
    synthesizer:
        :class:`SpectralSynthesizer` (photospheric lines, optionally with
        telluric lines), :class:`~mlp_bspline.hei.SiHeSynthesizer` (+ He I
        slab), or any module with the same interface (``phys_names``,
        ``depth_size``, ``slab_names``, ``relative_wavelengths``,
        ``forward(physical, mu, v_ref, wave_obs_rel, slab)``).
    stokes, wavelength:
        Observations ``(nx, ny, 4, nl)`` (NaN = missing) and their absolute
        wavelengths ``(nl,)`` (A). All regions of an inversion share the
        pixel grid ``(nx, ny)``.
    mu:
        Scalar or ``(nx, ny)``; may be left ``None`` if another region or
        :meth:`MultiRegionInverter.run` gives it.
    noise, stokes_weights:
        Per-Stokes noise (scalar or ``{"I": ..}``) and weights of this
        region's chi2; ``None`` takes ``InversionConfig.noise`` /
        ``stokes_weights``.
    weight:
        Weight of this region in the objective.
    wave_range:
        Optional ``(min, max)`` (A): only observed points inside it are fit.
    logtau:
        Depth grid of this region's emulators; ``None``: the inversion's.
    reference_depth:
        log(tau) where this region's emulators were trained with ``vz = 0``
        (the stratification's velocity there becomes a Doppler shift of the
        spectrum). ``None``: emulators trained with absolute velocities (no
        shift is applied).
    spatial_psf:
        ``None``, a :class:`~mlp_bspline.spatial_psf.SpatialPSF` or a 2-D
        array (odd sizes, centre in the central pixel, on the pixel grid of
        the observations): the synthetic maps are convolved with it before
        the comparison with the observations.
    """

    name: str
    synthesizer: nn.Module
    stokes: np.ndarray
    wavelength: np.ndarray
    mu: Optional[Union[float, np.ndarray]] = None
    noise: Optional[Union[float, Dict[str, float]]] = None
    stokes_weights: Optional[Dict[str, float]] = None
    weight: float = 1.0
    wave_range: Optional[Tuple[float, float]] = None
    logtau: Optional[np.ndarray] = None
    reference_depth: Optional[float] = -1.0
    spatial_psf: Optional[Union[SpatialPSF, np.ndarray]] = None

    def __post_init__(self):
        if self.spatial_psf is not None and not isinstance(self.spatial_psf, nn.Module):
            self.spatial_psf = SpatialPSF(self.spatial_psf)
        self.stokes = np.asarray(self.stokes)
        self.wavelength = np.asarray(self.wavelength, dtype=np.float64)
        if self.stokes.ndim != 4 or self.stokes.shape[2] != 4 or self.stokes.shape[3] != len(self.wavelength):
            raise ValueError(f"region {self.name!r}: stokes must be (nx, ny, 4, nl) with nl = "
                             f"{len(self.wavelength)} wavelengths, got {self.stokes.shape}")
        if self.weight < 0:
            raise ValueError(f"region {self.name!r}: weight must be >= 0")

    @classmethod
    def from_file(cls, name: str, path: str, synthesizer: nn.Module, **kwargs) -> "Region":
        """Region with the observations of an HDF5 file with datasets
        ``stokes (nx, ny, 4, nl)``, ``wavelength (nl,)`` (absolute A) and,
        optionally, ``mu`` (scalar or ``(nx, ny)``). ``kwargs``: any other
        :class:`Region` field (``noise``, ``weight``, ...)."""
        import h5py

        with h5py.File(path, "r") as f:
            stokes = f["stokes"][:]
            wave = f["wavelength"][:].astype(np.float64)
            mu = f["mu"][()] if "mu" in f else None
        if kwargs.get("mu") is None:
            kwargs["mu"] = mu
        return cls(name, synthesizer, stokes, wave, **kwargs)


@dataclass
class RegionResult:
    synthetic: np.ndarray    # (nx, ny, 4, nl), NaN where not synthesized
    chi2: np.ndarray         # (nx, ny) weighted chi2 per valid point of this region
    v_ref: np.ndarray        # (nx, ny) Doppler shift applied to this region (km/s)
    wavelength: np.ndarray   # (nl,)
    observations: np.ndarray  # (nx, ny, 4, nl)
    synthetic_nopsf: Optional[np.ndarray] = None  # (nx, ny, 4, nl) before the spatial PSF (if any)


@dataclass
class MultiRegionResult:
    nodes: Dict[str, np.ndarray]           # variable -> (nx, ny, n_nodes), physical units (vz absolute)
    node_positions: Dict[str, np.ndarray]  # variable -> (n_nodes,) log(tau)
    stratification: Dict[str, np.ndarray]  # depth variable -> (nx, ny, D) on `logtau` (vz absolute)
    logtau: np.ndarray                     # (D,)
    mu: np.ndarray                         # (nx, ny)
    objective: np.ndarray                  # (nx, ny) data term of each pixel
    regions: Dict[str, RegionResult]
    history: Dict[str, list]


def _noise_weights(region: Region, config: InversionConfig) -> List[float]:
    noise = config.noise if region.noise is None else region.noise
    if not isinstance(noise, dict):
        noise = {s: float(noise) for s in STOKES}
    weights = config.stokes_weights if region.stokes_weights is None else region.stokes_weights
    return [weights.get(s, 1.0) / noise.get(s, 1.0) ** 2 for s in STOKES]


class MultiRegionInverter:
    """Joint inversion of several spectral regions observed on the same
    pixel grid, with one set of per-pixel variables (node values of the
    stratification and per-pixel variables such as He I slab or telluric
    parameters) shared by every region that uses them: a region reads the
    variables named by its synthesizer (``phys_names`` and ``slab_names``),
    so using the same name in two regions ties them together and different
    names (e.g. prefixes ``"Tel630_"`` / ``"Tel_"``) keep them apart.

    The data term of a pixel is ``sum_r weight_r chi2_r / n_r`` with ``n_r``
    the number of valid points of region ``r`` in that pixel
    (``normalize=True``), so every region counts by its weight independently
    of its number of wavelengths, and regularization weights are relative to
    a chi2 per point (O(1) for a good fit); ``normalize=False`` sums the
    plain ``weight_r chi2_r``. See :class:`Inverter` for the optimization.
    """

    def __init__(self, regions: Sequence[Region], config: InversionConfig, logtau: np.ndarray,
                 normalize: bool = True):
        if not regions:
            raise ValueError("give at least one region")
        names = [r.name for r in regions]
        if len(set(names)) != len(names):
            raise ValueError(f"region names must be unique, got {names}")
        self.cfg = config
        self.device = torch.device(config.device)
        self.regions = list(regions)
        self.normalize = normalize
        self.logtau = np.asarray(logtau, dtype=np.float64)

        shapes = {r.name: r.stokes.shape[:2] for r in regions}
        if len(set(shapes.values())) != 1:
            raise ValueError(f"all regions must share the pixel grid, got {shapes}")
        self.shape = regions[0].stokes.shape[:2]

        missing, used = {}, set()
        for r in regions:
            need = list(r.synthesizer.phys_names) + list(getattr(r.synthesizer, "slab_names", ()))
            used.update(need)
            miss = [n for n in need if n not in config.variables]
            if miss:
                missing[r.name] = miss
        if missing:
            raise ValueError(f"variables needed by the regions but missing from the configuration: {missing}")
        unused = [n for n in config.variables if n not in used]
        if unused:
            print(f"warning: variables not used by any region (they will not change): {unused}")

        self.geometry = []
        for r in regions:
            grid = self.logtau if r.logtau is None else np.asarray(r.logtau, dtype=np.float64)
            if len(grid) != r.synthesizer.depth_size:
                raise ValueError(f"region {r.name!r}: depth grid has {len(grid)} points, its emulators expect "
                                 f"{r.synthesizer.depth_size}")
            if r.reference_depth is not None and not grid.min() <= r.reference_depth <= grid.max():
                raise ValueError(f"region {r.name!r}: reference_depth {r.reference_depth} outside its depth grid")
            self.geometry.append(_DepthGeometry(config.variables, grid, r.reference_depth, self.device))
            r.synthesizer.to(self.device)
            if r.spatial_psf is not None:
                r.spatial_psf.to(self.device)
        self.spatial = any(r.spatial_psf is not None for r in regions)
        self.main_geometry = _DepthGeometry(config.variables, self.logtau, None, self.device)

        self.bounds = {}
        for name, vc in config.variables.items():
            if vc.bounds is not None:
                n = len(vc.nodes)
                lo, hi = (_per_node(b, n, f"{name} bounds") for b in vc.bounds)
                self.bounds[name] = (torch.as_tensor(lo, dtype=torch.float32, device=self.device),
                                     torch.as_tensor(hi, dtype=torch.float32, device=self.device))

    # -- helpers --------------------------------------------------------------
    def _mu(self, mu) -> np.ndarray:
        nx, ny = self.shape
        if mu is not None:
            return np.broadcast_to(np.asarray(mu, dtype=np.float64), (nx, ny)).copy()
        given = [(r.name, np.broadcast_to(np.asarray(r.mu, dtype=np.float64), (nx, ny))) for r in self.regions
                 if r.mu is not None]
        if not given:
            raise ValueError("no mu given: pass it to run() or to one of the regions")
        for name, m in given[1:]:
            if not np.allclose(m, given[0][1], atol=1e-4):
                raise ValueError(f"mu of region {name!r} differs from that of region {given[0][0]!r}; "
                                 f"all regions share the pixel grid, so pass one mu to run()")
        return given[0][1].copy()

    def _prepare(self, region: Region):
        """Observations of a region on the device, their validity mask,
        relative wavelengths and Stokes weights."""
        dev = self.device
        nx, ny = self.shape
        nl = len(region.wavelength)
        obs_np = np.asarray(region.stokes, dtype=np.float32).reshape(nx * ny, 4, nl)
        valid = np.isfinite(obs_np)
        if region.wave_range is not None:
            lo, hi = sorted(region.wave_range)
            valid &= ((region.wavelength >= lo) & (region.wavelength <= hi))[None, None, :]
        return dict(
            obs=torch.as_tensor(np.nan_to_num(obs_np), device=dev),
            valid=torch.as_tensor(valid, device=dev),
            wave=torch.as_tensor(region.synthesizer.relative_wavelengths(region.wavelength), dtype=torch.float32,
                                 device=dev),
            w=torch.as_tensor(_noise_weights(region, self.cfg), dtype=torch.float32, device=dev),
        )

    @staticmethod
    def _chi2(synthetic, valid, data):
        """Per-pixel weighted chi2 (B,) and number of valid points (B,)."""
        mask = (valid[:, None, :] & data["valid"]).to(synthetic.dtype)
        r = (synthetic - data["obs"]) ** 2 * mask * data["w"][None, :, None]
        return r.sum(dim=(1, 2)), mask.sum(dim=(1, 2))

    def _nodes(self, params, sl):
        """Physical node values ``name -> (B, n)`` of the pixels in ``sl``."""
        npix = self.shape[0] * self.shape[1]
        return {n: params[n].reshape(npix, -1)[sl] * vc.scale for n, vc in self.cfg.variables.items()}

    def _synthesize(self, k: int, nodes, mu, data):
        region, geo = self.regions[k], self.geometry[k]
        physical, v_ref, slab = geo.inputs(nodes, region.synthesizer.phys_names)
        synthetic, valid = region.synthesizer(physical, mu, v_ref, data["wave"], slab)
        return synthetic, valid, v_ref

    def _forward(self, k: int, nodes, mu, data, sl):
        synthetic, valid, v_ref = self._synthesize(k, nodes, mu, data)
        chi2, npts = self._chi2(synthetic, valid, {key: (v[sl] if key in ("obs", "valid") else v)
                                                   for key, v in data.items()})
        return synthetic, valid, v_ref, chi2, npts

    def _maps(self, params, mu_t, data, chunks, grad: bool):
        """Synthetic profiles ``(npix, 4, nl)``, validity masks and ``v_ref``
        of every region over the whole field of view, chunk by chunk (with
        autograd graph if ``grad``)."""
        out = [([], [], []) for _ in self.regions]
        with torch.set_grad_enabled(grad):
            for sl in chunks:
                nodes = self._nodes(params, sl)
                for k in range(len(self.regions)):
                    for lst, t in zip(out[k], self._synthesize(k, nodes, mu_t[sl], data[k])):
                        lst.append(t)
        return [tuple(torch.cat(t) for t in o) for o in out]

    def _degrade(self, k: int, synthetic: torch.Tensor) -> torch.Tensor:
        """Synthetic maps ``(npix, 4, nl)`` of region ``k`` convolved with its
        spatial PSF (unchanged if it has none)."""
        psf = self.regions[k].spatial_psf
        if psf is None:
            return synthetic
        nx, ny = self.shape
        return psf(synthetic.reshape(nx, ny, *synthetic.shape[1:])).reshape(synthetic.shape)

    def _spatial_backward(self, params, mu_t, data, chunks):
        """Data term of the whole field of view with the spatial PSFs and its
        gradient with respect to the parameters (accumulated in ``.grad``);
        returns the data term and the chi2 and number of points per region.
        With several chunks, the gradient with respect to the full synthetic
        maps is back-propagated chunk by chunk through a second synthesis."""
        single = len(chunks) == 1
        maps = self._maps(params, mu_t, data, chunks, grad=single)
        loss, per_region, leaves = 0.0, np.zeros((len(self.regions), 2)), []
        for k, r in enumerate(self.regions):
            s, valid, _ = maps[k]
            if not single:
                s = s.detach().requires_grad_(True)
                leaves.append(s)
            chi2, npts = self._chi2(self._degrade(k, s), valid, data[k])
            loss = loss + self._data_term(chi2, npts, r.weight).sum()
            per_region[k] = (float(chi2.sum()), float(npts.sum()))
        if loss.requires_grad:
            loss.backward()
        if not single:
            for sl in chunks:
                nodes = self._nodes(params, sl)
                outs, grads = [], []
                for k in range(len(self.regions)):
                    s = self._synthesize(k, nodes, mu_t[sl], data[k])[0]
                    if s.requires_grad:
                        outs.append(s)
                        grads.append(leaves[k].grad[sl])
                if outs:
                    torch.autograd.backward(outs, grads)
        return float(loss), per_region

    def _data_term(self, chi2, npts, weight):
        if self.normalize:
            return weight * chi2 / npts.clamp(min=1.0)
        return weight * chi2

    def _prox_tau(self, p, vc, optimizer, lr: float):
        """Proximal step size of each node map of a variable, in physical units."""
        if self.cfg.optimizer in ("adam", "adamw"):
            st = optimizer.state[p]
            step_count = float(st["step"])
            beta2, eps = optimizer.defaults["betas"][1], optimizer.defaults["eps"]
            v_hat = st["exp_avg_sq"] / (1.0 - beta2 ** step_count)
            tau_p = (lr / (v_hat.sqrt() + eps)).mean(dim=(0, 1))  # (n,)
        else:
            tau_p = torch.full((p.shape[-1],), lr, device=p.device)
        return tau_p * vc.scale ** 2

    def _prox_step(self, params, optimizer, lr: float):
        """Proximal step of the regularizers + box projection, in place."""
        for name, vc in self.cfg.variables.items():
            if vc.fixed:
                continue
            p = params[name]  # (nx, ny, n), normalized
            if vc.regularization:
                tau_x = self._prox_tau(p, vc, optimizer, lr)
                maps = (p * vc.scale).permute(2, 0, 1)  # (n, nx, ny)
                for reg in vc.regularization:
                    maps = reg.prox(maps, tau_x)
                p.copy_(maps.permute(1, 2, 0) / vc.scale)
        for tr in self.cfg.transverse_regularization:
            vx, vy = self.cfg.variables[tr.bx], self.cfg.variables[tr.by]
            if vx.fixed or vy.fixed:
                continue
            px, py = params[tr.bx], params[tr.by]
            tau_x = 0.5 * (self._prox_tau(px, vx, optimizer, lr) + self._prox_tau(py, vy, optimizer, lr))
            bx, by = tr.prox((px * vx.scale).permute(2, 0, 1), (py * vy.scale).permute(2, 0, 1), tau_x)
            px.copy_(bx.permute(1, 2, 0) / vx.scale)
            py.copy_(by.permute(1, 2, 0) / vy.scale)
        self._project(params)

    def _project(self, params):
        for name, vc in self.cfg.variables.items():
            if name in self.bounds:
                lo, hi = self.bounds[name]
                p = params[name]
                p.copy_(torch.maximum(torch.minimum(p, hi / vc.scale), lo / vc.scale))

    def _regularization_value(self, params) -> float:
        total = 0.0
        for name, vc in self.cfg.variables.items():
            maps = (params[name] * vc.scale).permute(2, 0, 1)
            for reg in vc.regularization:
                total += float(reg.value(maps))
        for tr in self.cfg.transverse_regularization:
            vx, vy = self.cfg.variables[tr.bx], self.cfg.variables[tr.by]
            total += float(tr.value((params[tr.bx] * vx.scale).permute(2, 0, 1),
                                    (params[tr.by] * vy.scale).permute(2, 0, 1)))
        return total

    # -- main entry point -----------------------------------------------------
    def run(self, init: Dict[str, np.ndarray], mu=None) -> MultiRegionResult:
        """``init``: variable -> starting node values broadcastable to ``(nx,
        ny, n_nodes)`` (see :func:`initial_guess`,
        :func:`~mlp_bspline.hei.he_slab_initial_guess`,
        :func:`~mlp_bspline.telluric.telluric_initial_guess`). ``mu``: scalar
        or ``(nx, ny)``; default: from the regions."""
        cfg, dev = self.cfg, self.device
        nx, ny = self.shape
        npix = nx * ny
        missing = [n for n in cfg.variables if n not in init]
        if missing:
            raise ValueError(f"no initial value for {missing}")
        mu_np = self._mu(mu)
        if mu_np.min() < 0.2 - 1e-6 or mu_np.max() > 1.0 + 1e-6:
            print("warning: mu outside the training range [0.2, 1] of the emulators")
        mu_t = torch.as_tensor(mu_np.reshape(npix), dtype=torch.float32, device=dev)
        data = [self._prepare(r) for r in self.regions]

        params = {}
        for name, vc in cfg.variables.items():
            x0 = np.broadcast_to(np.asarray(init[name], dtype=np.float32), (nx, ny, len(vc.nodes)))
            params[name] = torch.tensor(x0 / vc.scale, device=dev, requires_grad=not vc.fixed)
        free = [params[n] for n, vc in cfg.variables.items() if not vc.fixed]
        opt_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}[cfg.optimizer]
        optimizer = opt_cls(free, lr=cfg.lr)
        with torch.no_grad():
            self._project(params)

        history = {"chi2": [], "regularization": [], "lr": []}
        history.update({f"chi2_{r.name}": [] for r in self.regions})
        chunks = [slice(s, min(s + cfg.chunk_size, npix)) for s in range(0, npix, cfg.chunk_size)]

        start_total = time.time()

        for it in range(cfg.n_iter):

            start = time.time()

            lr = cfg.lr * (cfg.lr_final_fraction + (1 - cfg.lr_final_fraction)
                           * 0.5 * (1 + math.cos(math.pi * it / max(1, cfg.n_iter - 1))))
            for g in optimizer.param_groups:
                g["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            if self.spatial:
                total, per_region = self._spatial_backward(params, mu_t, data, chunks)
            else:
                total = 0.0
                per_region = np.zeros((len(self.regions), 2))  # chi2, points
                for sl in chunks:
                    nodes = self._nodes(params, sl)
                    loss = 0.0
                    for k, r in enumerate(self.regions):
                        _, _, _, chi2, npts = self._forward(k, nodes, mu_t[sl], data[k], sl)
                        loss = loss + self._data_term(chi2, npts, r.weight).sum()
                        per_region[k] += (float(chi2.sum()), float(npts.sum()))
                    loss.backward()
                    total += float(loss)
            optimizer.step()
            with torch.no_grad():
                self._prox_step(params, optimizer, lr)
            log = cfg.log_every and (it % cfg.log_every == 0 or it == cfg.n_iter - 1)
            reg = self._regularization_value(params) if log or it == cfg.n_iter - 1 else None
            # data term per pixel (normalized) or per point (plain chi2)
            history["chi2"].append(total / npix if self.normalize else total / max(per_region[:, 1].sum(), 1.0))
            for k, r in enumerate(self.regions):
                history[f"chi2_{r.name}"].append(per_region[k, 0] / max(per_region[k, 1], 1.0))
            history["regularization"].append(reg)
            history["lr"].append(lr)

            iter_time = time.time() - start

            if log:
                regions = "  ".join(f"{r.name} {history[f'chi2_{r.name}'][-1]:.3e}" for r in self.regions)
                print(f"iter {it:5d}  (t={iter_time:.2f}s/{time.time() - start_total:.2f}s) data {history['chi2'][-1]:.4e}  ({regions})  "
                      f"regularization {reg:.4e}  lr {lr:.2e}")

        return self._result(params, mu_t, mu_np, data, chunks, history)

    @torch.no_grad()
    def _result(self, params, mu_t, mu_np, data, chunks, history) -> MultiRegionResult:
        cfg = self.cfg
        nx, ny = self.shape
        npix = nx * ny
        objective = np.zeros(npix, dtype=np.float32)
        out = {}
        maps = self._maps(params, mu_t, data, chunks, grad=False)
        for k, r in enumerate(self.regions):
            nl = len(r.wavelength)
            s, valid, v_ref = maps[k]
            sd = self._degrade(k, s)
            c, npts = self._chi2(sd, valid, data[k])
            nan = torch.full_like(s, float("nan"))
            synthetic = torch.where(valid[:, None, :], sd, nan).cpu().numpy().reshape(nx, ny, 4, nl)
            nopsf = None
            if r.spatial_psf is not None:
                nopsf = torch.where(valid[:, None, :], s, nan).cpu().numpy().reshape(nx, ny, 4, nl)
            objective += self._data_term(c, npts, r.weight).cpu().numpy()
            out[r.name] = RegionResult(synthetic, (c / npts.clamp(min=1)).cpu().numpy().reshape(nx, ny),
                                       v_ref.cpu().numpy().reshape(nx, ny), r.wavelength, np.asarray(r.stokes),
                                       synthetic_nopsf=nopsf)
        nodes = {n: (params[n] * vc.scale).cpu().numpy() for n, vc in cfg.variables.items()}
        strat = {n: nodes[n] @ M.cpu().numpy().T for n, M in self.main_geometry.M.items()}
        return MultiRegionResult(
            nodes=nodes,
            node_positions={n: np.asarray(vc.nodes) for n, vc in cfg.variables.items()},
            stratification=strat,
            logtau=self.logtau,
            mu=mu_np,
            objective=objective.reshape(nx, ny),
            regions=out,
            history=history,
        )


class Inverter:
    """Inversion of one spectral region (``Region`` with
    ``InversionConfig.noise``/``stokes_weights``/``reference_depth`` and the
    plain chi2), see :class:`MultiRegionInverter` for several.

    The objective is ``sum_pixels chi2 + sum regularization``, minimized
    over the node values with Adam (or SGD) plus a proximal step for the
    spatial regularizers and a projection onto the bounds after every
    gradient step (see the module docstring)."""

    def __init__(self, synthesizer: nn.Module, config: InversionConfig, logtau: np.ndarray,
                 spatial_psf: Optional[Union[SpatialPSF, np.ndarray]] = None):
        """``spatial_psf``: ``None`` or a spatial PSF (see :class:`Region`)
        with which the synthetic maps are convolved before the comparison."""
        self.cfg = config
        self.spatial_psf = spatial_psf
        self.device = torch.device(config.device)
        self.synth = synthesizer.to(self.device)
        self.logtau = np.asarray(logtau, dtype=np.float64)
        if len(self.logtau) != synthesizer.depth_size:
            raise ValueError(f"logtau has {len(self.logtau)} points, the emulators expect {synthesizer.depth_size}")
        self._geometry = _DepthGeometry(config.variables, self.logtau, config.reference_depth, self.device)
        self.ref_index = self._geometry.ref_index

    def _stratification(self, nodes: Dict[str, torch.Tensor]):
        """Physical node values ``name -> (B, n)`` -> emulator input ``(B, 6,
        D)`` (with ``dv``), ``v_ref (B,)`` and the per-pixel variables
        ``name -> (B,)``."""
        return self._geometry.inputs(nodes, self.synth.phys_names)

    def run(self, observations: np.ndarray, wave_obs: np.ndarray, mu, init: Dict[str, np.ndarray]) -> InversionResult:
        """observations (nx, ny, 4, nl), wave_obs (nl,) absolute A, mu scalar
        or (nx, ny), init: variable -> node values broadcastable to (nx, ny,
        n_nodes) (see :func:`initial_guess`)."""
        region = Region("main", self.synth, observations, wave_obs, mu=mu,
                        reference_depth=self.cfg.reference_depth, spatial_psf=self.spatial_psf)
        res = MultiRegionInverter([region], self.cfg, self.logtau, normalize=False).run(init)
        r = res.regions["main"]
        history = {k: v for k, v in res.history.items() if k != "chi2_main"}
        return InversionResult(nodes=res.nodes, node_positions=res.node_positions,
                               stratification=res.stratification, logtau=res.logtau, v_ref=r.v_ref,
                               synthetic=r.synthetic, chi2=r.chi2, history=history,
                               synthetic_nopsf=r.synthetic_nopsf)


def save_result(path: str, result: InversionResult, wave_obs: np.ndarray, observations: Optional[np.ndarray] = None):
    """Writes an :class:`InversionResult` to HDF5: ``nodes/<var>``,
    ``node_positions/<var>``, ``stratification/<var>``, ``logtau``, ``v_ref``,
    ``synthetic``, ``chi2``, ``wavelength`` (and ``observations`` if given),
    plus the ``history`` of chi2 per point. With a spatial PSF, ``synthetic``
    is convolved with it and ``synthetic_nopsf`` holds the profiles before
    the convolution."""
    import h5py

    with h5py.File(path, "w") as f:
        for group, d in (("nodes", result.nodes), ("node_positions", result.node_positions),
                         ("stratification", result.stratification)):
            g = f.create_group(group)
            for k, v in d.items():
                g.create_dataset(k, data=v)
        f.create_dataset("logtau", data=result.logtau)
        f.create_dataset("v_ref", data=result.v_ref)
        f.create_dataset("synthetic", data=result.synthetic)
        f.create_dataset("chi2", data=result.chi2)
        f.create_dataset("wavelength", data=np.asarray(wave_obs))
        f.create_dataset("history_chi2", data=np.asarray(result.history["chi2"]))
        if result.synthetic_nopsf is not None:
            f.create_dataset("synthetic_nopsf", data=result.synthetic_nopsf)
        if observations is not None:
            f.create_dataset("observations", data=observations)


def save_multiregion_result(path: str, result: MultiRegionResult):
    """Writes a :class:`MultiRegionResult` to HDF5: the shared ``nodes/<var>``,
    ``node_positions/<var>``, ``stratification/<var>``, ``logtau``, ``mu``,
    ``objective`` (data term per pixel) and ``history/<quantity>``, and for
    every region ``regions/<name>/{synthetic, observations, wavelength,
    chi2, v_ref}`` (and ``synthetic_nopsf`` with a spatial PSF)."""
    import h5py

    with h5py.File(path, "w") as f:
        for group, d in (("nodes", result.nodes), ("node_positions", result.node_positions),
                         ("stratification", result.stratification)):
            g = f.create_group(group)
            for k, v in d.items():
                g.create_dataset(k, data=v)
        f.create_dataset("logtau", data=result.logtau)
        f.create_dataset("mu", data=result.mu)
        f.create_dataset("objective", data=result.objective)
        g = f.create_group("history")
        for k, v in result.history.items():
            g.create_dataset(k, data=np.asarray([np.nan if x is None else x for x in v], dtype=np.float64))
        for name, r in result.regions.items():
            g = f.create_group(f"regions/{name}")
            for k in ("synthetic", "observations", "wavelength", "chi2", "v_ref", "synthetic_nopsf"):
                if getattr(r, k) is not None:
                    g.create_dataset(k, data=getattr(r, k))
