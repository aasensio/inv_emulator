"""Configuration dataclasses for B-spline channels and scalers.

Every physical-atmosphere channel (T, vz, vmic, Bx, By, Bz) and every Stokes
channel (I, Q, U, V) gets its own independent configuration: its own number
of B-spline coefficients, its own degree, and optionally its own explicit
knot vector. This is what lets e.g. Q/U/V use fewer or differently placed
knots than I, or Bz use more knots than vz.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


@dataclass
class ChannelSplineConfig:
    """B-spline configuration for a single physical or Stokes channel.

    Parameters
    ----------
    num_coeffs:
        Number of B-spline basis functions / coefficients for this channel.
    degree:
        Degree of the B-spline (3 = cubic).
    knots:
        Optional explicit *full* clamped knot vector (length
        ``num_coeffs + degree + 1``). If ``None``, a clamped, uniformly
        spaced knot vector spanning the channel's grid is generated
        automatically.
    """

    num_coeffs: int
    degree: int = 3
    knots: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.knots is not None:
            self.knots = np.asarray(self.knots, dtype=np.float64)
            expected = self.num_coeffs + self.degree + 1
            if len(self.knots) != expected:
                raise ValueError(
                    f"knots must have length num_coeffs + degree + 1 = "
                    f"{expected}, got {len(self.knots)}"
                )


@dataclass
class ChannelPCAConfig:
    """PCA basis configuration for a single Stokes channel (used with
    ``stokes_basis="pca"``; see :mod:`mlp_bspline.pca`). Give exactly one of:

    num_components:
        Number of leading principal components kept.
    variance_fraction:
        Keep the fewest components whose cumulative explained variance
        reaches this fraction (e.g. 0.9999).

    center:
        Subtract the mean training profile before the decomposition (and add
        it back on reconstruction). Almost always wanted, in particular for
        Stokes I.
    """

    num_components: Optional[int] = None
    variance_fraction: Optional[float] = None
    center: bool = True

    def __post_init__(self):
        if (self.num_components is None) == (self.variance_fraction is None):
            raise ValueError("give exactly one of num_components or variance_fraction")
        if self.num_components is not None and self.num_components < 1:
            raise ValueError(f"num_components must be >= 1, got {self.num_components}")
        if self.variance_fraction is not None and not 0 < self.variance_fraction <= 1:
            raise ValueError(f"variance_fraction must be in (0, 1], got {self.variance_fraction}")


@dataclass
class ScalerConfig:
    """Configuration for scaling one channel's B-spline coefficients.

    kind:
        - "standard": zero mean / unit std, fitted from training data.
        - "minmax": rescale to [0, 1], fitted from training data.
        - "fixed": user-supplied constant ``offset``/``scale`` (not fitted).
        - "none": identity.
    per_coeff:
        If True (default) statistics are computed independently for each
        B-spline coefficient of the channel. If False, a single scalar
        offset/scale is used for the whole channel (useful e.g. to keep the
        relative shape across coefficients of Q/U/V while normalizing the
        overall amplitude by, say, a fraction of the continuum).
    scale, offset:
        Required (and used as-is) only when kind == "fixed".
    """

    kind: str = "standard"
    per_coeff: bool = True
    scale: Optional[float] = None
    offset: Optional[float] = None

    def __post_init__(self):
        valid = {"standard", "minmax", "fixed", "none"}
        if self.kind not in valid:
            raise ValueError(f"kind must be one of {valid}, got {self.kind!r}")
        if self.kind == "fixed" and (self.scale is None or self.offset is None):
            raise ValueError("kind='fixed' requires both scale and offset")


@dataclass
class DepthFeatureConfig:
    """Extra model inputs describing how the stratification changes with
    depth (see :mod:`mlp_bspline.features`), computed from the physical
    B-spline coefficients and appended to them.

    derivatives:
        Channels whose derivative with respect to the depth grid (e.g.
        log tau) is added, as a feature named ``"d<channel>"``. ``"B"``
        (the field strength ``|B|``) is also accepted.
    mode:
        How the derivatives are represented. ``"spline"``: the analytic
        derivative of the B-spline evaluated at ``eval_points``.
        ``"diff"``: the differences of consecutive B-spline coefficients
        divided by the knot spacing, ``p (c[i+1] - c[i]) / (t[i+p+1] -
        t[i+1])``, which are the coefficients of the derivative spline
        (one fewer than the channel has). Products always use ``"spline"``
        values at ``eval_points``.
    products:
        Cross-coupling terms, each ``"<a>*<b>"`` where a factor is a channel
        name (its value), ``"d"`` + a channel name (its derivative), ``"B"``
        (``|B|``) or ``"dB"`` (``d|B|``), e.g. ``"dvz*dT"``, ``"dvz*dBz"``,
        ``"dvz*B"``.
    eval_points:
        Depth points where values and derivatives are evaluated. ``None``:
        the distinct knots of the first physical channel's B-spline.
    squash_products:
        Apply ``asinh`` to the standardized products, which are much more
        heavy-tailed than the coefficients (a few training models sit tens
        of standard deviations out).
    field_channels:
        Names of the ``(Bx, By, Bz)`` channels used for ``B``/``dB``.
    """

    derivatives: Sequence[str] = ()
    mode: str = "spline"
    products: Sequence[str] = ()
    eval_points: Optional[Sequence[float]] = None
    squash_products: bool = True
    field_channels: Sequence[str] = ("Bx", "By", "Bz")

    def __post_init__(self):
        if self.mode not in ("spline", "diff"):
            raise ValueError(f"mode must be 'spline' or 'diff', got {self.mode!r}")
        self.derivatives = tuple(self.derivatives)
        self.products = tuple(self.products)
        self.field_channels = tuple(self.field_channels)
        if self.eval_points is not None:
            self.eval_points = [float(x) for x in self.eval_points]
        for p in self.products:
            if len(p.split("*")) != 2:
                raise ValueError(f"products must look like 'a*b', got {p!r}")
        if not self.derivatives and not self.products:
            raise ValueError("give at least one of derivatives or products")


PHYSICAL_CHANNELS = ("T", "vz", "vmic", "Bx", "By", "Bz")
STOKES_CHANNELS = ("I", "Q", "U", "V")


@dataclass
class StokesTransformConfig:
    """Optional preprocessing applied to raw Stokes profiles before they are
    fit (see :mod:`mlp_bspline.transforms` for the full rationale):
    normalizes every ``ratio_channels`` entry by ``intensity_channel`` (e.g.
    Q/I, U/I, V/I) and applies ``symlog(amplitude * x) = sign(amplitude * x)
    * log1p(|amplitude * x|)`` to each ratio. ``intensity_channel`` itself
    is left untouched.

    amplitude:
        The ``a`` in ``symlog(a * x)``. Sets where the transform switches
        from roughly linear (``|x| << 1/a``) to logarithmic (``|x| >>
        1/a``); tune it to the typical size of the ratios in your dataset.
    intensity_channel:
        Name of the channel used as the normalizing denominator.
    ratio_channels:
        Names of the channels to normalize + symlog. Defaults to every
        channel except ``intensity_channel``.
    """

    amplitude: float = 1.0
    intensity_channel: str = "I"
    ratio_channels: Optional[Sequence[str]] = None

    def __post_init__(self):
        if self.amplitude <= 0:
            raise ValueError(f"amplitude must be > 0, got {self.amplitude}")
        if self.ratio_channels is not None:
            self.ratio_channels = tuple(self.ratio_channels)


def _wave_range(wave_range, what: str):
    if wave_range is None:
        return None
    wave_range = tuple(wave_range)
    if len(wave_range) != 2:
        raise ValueError(f"{what} must be (min, max), got {wave_range}")
    return tuple(sorted(wave_range))


@dataclass
class WeakFieldLine:
    """One spectral line of a :class:`WeakFieldConfig`.

    lambda0:
        Rest wavelength of the line in A (absolute, independent of any offset
        applied to the wavelength grid, since only derivatives with respect
        to wavelength enter).
    geff:
        Effective Lande factor ``gbar`` (Stokes V).
    Geff:
        Second-order effective Lande factor ``Gbar`` (Stokes Q and U); see
        :func:`mlp_bspline.weak_field.second_order_lande`.
    wave_range:
        Optional ``(min, max)`` window (inclusive) **in the units of the
        wavelength grid passed to training** (so e.g. relative to whatever
        offset was subtracted from it) over which this line's WFA is
        applied; outside it the line contributes nothing. ``None`` applies
        it over the whole grid.
    """

    lambda0: float
    geff: float
    Geff: float
    wave_range: Optional[Sequence[float]] = None

    def __post_init__(self):
        self.wave_range = _wave_range(self.wave_range, "wave_range")


@dataclass
class WeakFieldConfig:
    """Fit Q, U, V as residuals with respect to the weak-field approximation
    (WFA) built from Stokes I and the depth-averaged magnetic field (see
    :mod:`mlp_bspline.weak_field` for the formulae). Stokes I is still fit
    directly.

    The line(s) are given either as a single line through ``lambda0``,
    ``geff``, ``Geff`` (and optionally ``wave_range``), or as several
    through ``lines`` -- e.g. the Fe I 6301.5/6302.5 doublet -- each with its
    own Lande factors and its own wavelength window, since the WFA of each
    line must only be applied around that line. Windows of different lines
    may not overlap; wavelengths outside every window get no WFA baseline.
    Choose the boundaries in the continuum between the lines (where dI/dl
    ~ 0), since the baseline jumps to zero at the window edges.

    lambda0, geff, Geff, wave_range:
        Single-line shorthand; see :class:`WeakFieldLine`. Leave
        ``lambda0``/``geff``/``Geff`` as ``None`` when using ``lines``.
    lines:
        Sequence of :class:`WeakFieldLine` (or dicts with the same keys).
        Each line's ``wave_range`` is required when there is more than one.
    linear_polarization_sign:
        ``+1`` for the textbook sign of Q/U (Landi Degl'Innocenti &
        Landolfi 2004), ``-1`` for a reference direction of positive Q
        rotated by 90 deg, which is the convention of SIR (and Hazel's
        photospheres).
    depth_range:
        Optional ``(min, max)`` window of the depth grid (inclusive) over
        which the field is averaged; ``None`` averages over all depth
        points. Shared by all lines.
    field_channels:
        Names of the physical channels holding ``(Bx, By, Bz)`` in the
        line-of-sight frame (``Bz`` along the LOS), in G.
    intensity_channel, stokes_channels:
        Names of the Stokes I channel and of the ``(Q, U, V)`` channels that
        are fit as residuals.
    """

    lambda0: Optional[float] = None
    geff: Optional[float] = None
    Geff: Optional[float] = None
    linear_polarization_sign: float = 1.0
    depth_range: Optional[Sequence[float]] = None
    field_channels: Sequence[str] = ("Bx", "By", "Bz")
    intensity_channel: str = "I"
    stokes_channels: Sequence[str] = ("Q", "U", "V")
    wave_range: Optional[Sequence[float]] = None
    lines: Optional[Sequence[WeakFieldLine]] = None

    def __post_init__(self):
        if self.linear_polarization_sign not in (1.0, -1.0):
            raise ValueError(f"linear_polarization_sign must be +1 or -1, got {self.linear_polarization_sign}")
        self.depth_range = _wave_range(self.depth_range, "depth_range")
        self.wave_range = _wave_range(self.wave_range, "wave_range")
        self.field_channels = tuple(self.field_channels)
        self.stokes_channels = tuple(self.stokes_channels)
        if len(self.field_channels) != 3 or len(self.stokes_channels) != 3:
            raise ValueError("field_channels and stokes_channels must each name exactly 3 channels")

        single = (self.lambda0, self.geff, self.Geff)
        if self.lines is None:
            if any(v is None for v in single):
                raise ValueError("give either lambda0, geff and Geff (single line) or lines")
            return
        if any(v is not None for v in single) or self.wave_range is not None:
            raise ValueError("lambda0/geff/Geff/wave_range and lines are mutually exclusive")
        # dicts come back from dataclasses.asdict() in saved checkpoints
        self.lines = tuple(l if isinstance(l, WeakFieldLine) else WeakFieldLine(**l) for l in self.lines)
        if not self.lines:
            raise ValueError("lines must contain at least one line")
        if len(self.lines) > 1:
            if any(l.wave_range is None for l in self.lines):
                raise ValueError("every line needs a wave_range when there is more than one")
            ranges = sorted(l.wave_range for l in self.lines)
            for (_, hi), (lo, _) in zip(ranges[:-1], ranges[1:]):
                if lo <= hi:
                    raise ValueError(f"wave_range windows of different lines overlap: {ranges}")

    @property
    def all_lines(self) -> tuple:
        """The line(s) as a tuple of :class:`WeakFieldLine`, whichever way
        they were given."""
        if self.lines is not None:
            return self.lines
        return (WeakFieldLine(self.lambda0, self.geff, self.Geff, self.wave_range),)
