"""Common visual style for the figures of the paper (analysis.py, precision.py).

    import plotstyle as ps
    ps.apply()

Sets a clean, sans-serif matplotlib style (recessive axes and hairline grid,
no top/right spines, frameless legends, left-aligned titles) and provides a
validated categorical palette and one-hue sequential / two-hue diverging
colormaps with a neutral gray midpoint.
"""
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as pl
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
# Categorical slots, used in this fixed order (validated for colour-vision
# deficiencies on adjacent pairs; the first three also on all pairs)
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
MAGENTA, GREEN, VIOLET, RED = '#e87ba4', '#008300', '#6250d6', '#e34948'
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

INK = '#0b0b0b'             # primary text
INK2 = '#52514e'            # secondary text (axis labels)
MUTED = '#898781'           # tick labels, reference lines
GRID = '#e1e0d9'            # hairline grid
BASELINE = '#c3c2b7'        # axes spines
NEUTRAL = '#f0efec'         # midpoint of the diverging maps

BLUE_RAMP = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281',
             '#0d366b']

# One-hue sequential maps (light = low, dark = high)
SEQ_BLUE = LinearSegmentedColormap.from_list('seq_blue', ['#f6f9fe'] + BLUE_RAMP)
SEQ_ORANGE = LinearSegmentedColormap.from_list('seq_orange', ['#fdf5f0', '#f9d2bd', '#f3a47f', ORANGE, '#c14a1b',
                                                              '#8e3210', '#5c1f08'])
SEQ_VIOLET = LinearSegmentedColormap.from_list('seq_violet', ['#f7f6fd', '#d6d1f6', '#a99eec', VIOLET, '#4a3aa7',
                                                              '#2f2470', '#1b1442'])
# Two-hue diverging map, blue (negative) -- gray -- red (positive)
DIVERGING = LinearSegmentedColormap.from_list('diverging', ['#0d366b', '#1c5cab', '#3987e5', '#9ec5f4', NEUTRAL,
                                                            '#f4b3ab', '#e34948', '#b42d2d', '#6e1515'])
# Continuum intensity: neutral warm gray
INTENSITY = LinearSegmentedColormap.from_list('intensity', ['#141413', '#52514e', '#a6a59d', '#e8e7e1', '#fcfcfb'])

# Pixel markers drawn over maps: dark ring with a white halo, readable on any colormap
HALO = [pe.withStroke(linewidth=2.6, foreground='white')]


def apply():
    mpl.rcParams.update({
        # typography
        'font.family': 'sans-serif',
        'font.sans-serif': ['Roboto', 'Lato', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10.5,
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Roboto',
        'mathtext.it': 'Roboto:italic',
        'mathtext.bf': 'Roboto:bold',
        'mathtext.sf': 'Roboto',
        'mathtext.cal': 'Roboto',
        'mathtext.fallback': 'stixsans',
        'text.color': INK,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        # figure
        'figure.facecolor': 'white',
        'figure.dpi': 120,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.04,
        'figure.constrained_layout.use': True,
        'figure.constrained_layout.h_pad': 0.06,
        'figure.constrained_layout.w_pad': 0.06,
        # axes
        'axes.facecolor': 'white',
        'axes.edgecolor': BASELINE,
        'axes.linewidth': 0.8,
        'axes.labelcolor': INK2,
        'axes.labelsize': 10.5,
        'axes.labelpad': 3.0,
        'axes.titlesize': 11,
        'axes.titleweight': 'medium',
        'axes.titlelocation': 'left',
        'axes.titlepad': 6.0,
        'axes.titlecolor': INK,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.axisbelow': True,
        'axes.prop_cycle': cycler(color=CATEGORICAL),
        'axes.formatter.use_mathtext': True,
        'axes.formatter.limits': (-3, 5),
        'axes.formatter.useoffset': False,
        'grid.color': GRID,
        'grid.linewidth': 0.6,
        'grid.linestyle': '-',
        # ticks
        'xtick.color': BASELINE,
        'ytick.color': BASELINE,
        'xtick.labelcolor': MUTED,
        'ytick.labelcolor': MUTED,
        'xtick.labelsize': 10.5,
        'ytick.labelsize': 10.5,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 3.0,
        'ytick.major.size': 3.0,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.visible': False,
        'ytick.minor.visible': False,
        # marks
        'lines.linewidth': 1.6,
        'lines.solid_capstyle': 'round',
        'lines.dash_capstyle': 'round',
        'lines.markersize': 4.5,
        'lines.markeredgewidth': 0.0,
        'patch.linewidth': 0.0,
        'image.interpolation': 'nearest',
        'image.origin': 'lower',
        # legend
        'legend.frameon': False,
        'legend.fontsize': 10.5,
        'legend.labelcolor': INK2,
        'legend.handlelength': 1.6,
        'legend.handletextpad': 0.5,
        'legend.borderaxespad': 0.3,
        'legend.columnspacing': 1.0,
    })


def map_axes(a):
    """Image panel: no spines, no grid, light ticks."""
    a.grid(False)
    for s in a.spines.values():
        s.set_visible(False)
    a.tick_params(length=2.0)


def colorbar(im, ax, label=None, shrink=1.0, **kw):
    """Thin colorbar without outline."""
    cb = pl.colorbar(im, ax=ax, shrink=shrink, pad=0.02, aspect=28, fraction=0.05, **kw)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2.0, width=0.6, labelsize=9, color=BASELINE, labelcolor=MUTED)
    cb.ax.grid(False)
    if label is not None:
        cb.set_label(label, color=INK2, fontsize=9.5)
    return cb


def mark(a, x, y, label, left=False):
    """Labelled marker of a pixel over a map (label to the right of the marker, or to the left if left=True)."""
    a.plot(x, y, marker='o', mfc='none', mec=INK, ms=6.5, mew=1.2, path_effects=HALO, zorder=5)
    a.annotate(label, (x, y), xytext=(-5 if left else 5, 4), textcoords='offset points', color=INK, fontsize=7.5,
               fontweight='bold', path_effects=HALO, zorder=5, ha='right' if left else 'left')


def zero_line(a, y=0.0, **kw):
    a.axhline(y, color=kw.pop('color', BASELINE), lw=kw.pop('lw', 0.8), zorder=kw.pop('zorder', 1), **kw)
