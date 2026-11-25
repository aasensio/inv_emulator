"""Statistical properties of the difference between the emulated and the
Hazel2 Stokes profiles for the Fe I 6301/6302 doublet and the Si I 10827 +
He I 10830 region, and between the emulated and the Lightweaver ones for
Ca II 8542, computed for 2000 models of the test set (see quality_fe6301.py,
quality_he10830.py and quality_caii8542.py, which produce the files in
quality/). Produces

    paper/figs/precision_percentiles.pdf   percentiles of the error vs wavelength
    paper/figs/precision_rms.pdf           distribution of the rms error per profile
                                           and its dependence on the atmosphere

and prints the numbers quoted in the paper.

    python precision.py
"""
import h5py
import matplotlib.pyplot as pl
import numpy as np

import plotstyle as ps

MODELS = '/scratch/Dropbox/GIT/DeepLearning/universal_stokes_flow/database/models/models_testing.h5'
WAVE_FE = np.linspace(6300.8921, 6303.2671, 150)
WAVE_HE = np.linspace(10824.3, 10832.7, 421)
SPLIT_HE = 10829.0          # separates the Si I line from the He I multiplet
STOKES = ('I', 'Q', 'U', 'V')
NOISE = 1e-3                # typical noise level of the observations (in units of I_c)

ps.apply()


def ylabel(i):
    return r'$\Delta I$' if i == 0 else rf'$\Delta {STOKES[i]}/I_c$'


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
with h5py.File('quality/quality_fe6301.h5', 'r') as f:
    emulator_fe = f['stokes_emulator_6301'][:].astype(np.float64)
    hazel_fe = f['stokes_hazel_6301'][:]

with h5py.File('quality/quality_he10830.h5', 'r') as f:
    emulator_he = f['stokes_emulator_10830'][:].astype(np.float64)
    hazel_he = f['stokes_hazel_10830'][:]

with h5py.File('quality/quality_caii8542.h5', 'r') as f:
    wave_ca = f['wavelength'][:]
    emulator_ca = f['stokes_emulator_8542'][:].astype(np.float64)
    database_ca = f['stokes_database_8542'][:]

n_models = emulator_fe.shape[0]

# Same photospheric models used in quality_*.py (the first n_models of the test set)
with h5py.File(MODELS, 'r') as f:
    models = f['model'][:n_models]

logtau = models[0, :, 0]
forming = (logtau > -3.0) & (logtau < 0.5)          # line-forming region
delta_v = np.ptp(models[:, forming, 4], axis=1)       # velocity excursion, km/s
B = np.sqrt(np.sum(models[:, :, 5:8] ** 2, axis=-1))
B_forming = np.mean(B[:, forming], axis=1)            # mean field strength, G

regions = {
    'Fe I 6301/6302': (WAVE_FE, emulator_fe - hazel_fe, hazel_fe),
    'Si I 10827 + He I 10830': (WAVE_HE, emulator_he - hazel_he, hazel_he),
    'Ca II 8542': (wave_ca, emulator_ca - database_ca, database_ca),
}
XTICKS = {'Si I 10827 + He I 10830': [10826, 10829, 10832], 'Ca II 8542': [8541, 8542, 8543]}

# --------------------------------------------------------------------------- #
# Numbers quoted in the paper
# --------------------------------------------------------------------------- #
for name, (wave, diff, hazel) in regions.items():
    print(f'== {name} ({diff.shape[0]} models)')
    for i, s in enumerate(STOKES):
        d = diff[:, i, :]
        rms = np.sqrt(np.mean(d ** 2, axis=1))
        amp = 1.0 - np.min(hazel[:, 0], axis=1) if i == 0 else np.max(np.abs(hazel[:, i]), axis=1)
        print(f'  {s}: bias {d.mean():9.2e}  68% [{np.percentile(d, 16):9.2e}, {np.percentile(d, 84):9.2e}]'
              f'  95% [{np.percentile(d, 2.5):9.2e}, {np.percentile(d, 97.5):9.2e}]'
              f'  99.9% |d| {np.percentile(np.abs(d), 99.9):8.2e}  max |d| {np.abs(d).max():8.2e}')
        print(f'     rms per profile: median {np.median(rms):8.2e}  p95 {np.percentile(rms, 95):8.2e}'
              f'  frac < {NOISE:.0e}: {np.mean(rms < NOISE):5.3f}'
              f'  | median signal amplitude {np.median(amp):8.2e}'
              f'  | median rms / amplitude {np.median(rms / np.maximum(amp, 1e-6)):8.2e}')
    if wave is WAVE_HE:
        for part, sel in (('Si I', wave < SPLIT_HE), ('He I', wave >= SPLIT_HE)):
            rms = np.sqrt(np.mean(diff[:, :, sel] ** 2, axis=2))
            print(f'  {part}: median rms per profile ' + '  '.join(f'{s} {r:8.2e}' for s, r in zip(STOKES, np.median(rms, 0))))
    rms = np.sqrt(np.mean(diff ** 2, axis=2))
    for lo, hi in ((0, 2), (2, 4), (4, 8), (8, np.inf)):
        k = (delta_v >= lo) & (delta_v < hi)
        print(f'  delta_v in [{lo}, {hi}) km/s ({k.sum():4d}): median rms '
              + '  '.join(f'{s} {r:8.2e}' for s, r in zip(STOKES, np.median(rms[k], 0))))

print(f'delta_v percentiles 50/90/99: {np.percentile(delta_v, [50, 90, 99]).round(1)} km/s')

# --------------------------------------------------------------------------- #
# Figure 1: percentiles of the error as a function of wavelength
# --------------------------------------------------------------------------- #
fig, ax = pl.subplots(nrows=3, ncols=4, figsize=(10, 7.2))
for row, (name, (wave, diff, hazel)) in enumerate(regions.items()):
    pct_68 = np.percentile(diff, [50.0 - 68 / 2.0, 50.0 + 68 / 2.0], axis=0)
    pct_95 = np.percentile(diff, [50.0 - 95 / 2.0, 50.0 + 95 / 2.0], axis=0)
    for i in range(4):
        a = ax[row, i]
        a.fill_between(wave, pct_95[0, i], pct_95[1, i], color=ps.BLUE_RAMP[1], lw=0, label='95%')
        a.fill_between(wave, pct_68[0, i], pct_68[1, i], color=ps.BLUE_RAMP[5], lw=0, label='68%')
        ps.zero_line(a, color=ps.INK2, lw=0.6, zorder=3)
        a.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
        a.ticklabel_format(axis='x', useOffset=False)
        a.set_xlim(wave[0], wave[-1])
        a.set_ylabel(ylabel(i))
        if row == 0:
            a.set_title(f'Stokes {STOKES[i]}', fontweight='bold')
        if name in XTICKS:
            a.set_xticks(XTICKS[name])
        a.set_xlabel(r'wavelength [$\AA$]')
    ax[row, 0].annotate(name, (0.0, 1.0), xycoords='axes fraction', xytext=(4, -4), textcoords='offset points',
                        va='top', fontsize=7.5, color=ps.INK2, fontweight='medium')
ax[0, 3].legend(loc='lower right', ncols=2)
fig.savefig('paper/figs/precision_percentiles.pdf')

# --------------------------------------------------------------------------- #
# Figure 2: rms error per profile, cumulative distribution and dependence on
# the velocity gradient and on the field strength
# --------------------------------------------------------------------------- #
fig, ax = pl.subplots(nrows=1, ncols=3, figsize=(10, 3.2))
linestyles = ('-', '--')       # Fe I and Si I + He I only: zip() below leaves out Ca II
for (name, (wave, diff, hazel)), ls in zip(regions.items(), linestyles):
    rms = np.sqrt(np.mean(diff ** 2, axis=2))
    for i, s in enumerate(STOKES):
        x = np.sort(rms[:, i])
        ax[0].plot(x, np.arange(1, len(x) + 1) / len(x), color=ps.CATEGORICAL[i], ls=ls,
                   label=s if ls == '-' else None)

    for a, x, edges in ((ax[1], delta_v, np.array([0, 1, 2, 3, 4, 6, 8, 12, 25])),
                        (ax[2], B_forming, np.array([0, 100, 200, 400, 600, 800, 1000, 1400, 2000, 3000]))):
        idx = np.digitize(x, edges) - 1
        centers = 0.5 * (edges[1:] + edges[:-1])
        for i in range(4):
            med = np.array([np.median(rms[idx == k, i]) if np.sum(idx == k) >= 10 else np.nan
                            for k in range(len(centers))])
            a.semilogy(centers, med, color=ps.CATEGORICAL[i], ls=ls, marker='o', ms=4.5, mec='white', mew=0.8)

ax[0].axvline(NOISE, color=ps.MUTED, ls=(0, (1, 2)), lw=1.2)
ax[0].annotate('noise', (NOISE, 0.02), xytext=(3, 0), textcoords='offset points', fontsize=7, color=ps.MUTED)
ax[0].set_xscale('log')
ax[0].set_xlabel('rms error per profile [$I_c$]')
ax[0].set_ylabel('cumulative fraction')
leg = ax[0].legend(loc='upper left', title='Stokes', title_fontsize=7.5, alignment='left')
ax[0].add_artist(leg)
lines = [pl.Line2D([], [], color=ps.INK2, ls=ls) for ls in linestyles]
ax[0].legend(lines, ['Fe I', 'Si I + He I'], loc='lower right')
ax[1].set_xlabel(r'$\Delta v$ in $-3<\log\tau_{500}<0.5$ [km s$^{-1}$]')
ax[1].set_ylabel('median rms error [$I_c$]')
ax[2].set_xlabel(r'$\langle B \rangle$ in $-3<\log\tau_{500}<0.5$ [G]')
ax[2].set_ylabel('median rms error [$I_c$]')
for a, t in zip(ax, ('(a)  rms error per profile', r'(b)  vs velocity gradient', r'(c)  vs field strength')):
    a.set_title(t)
for a in ax[1:]:
    a.axhline(NOISE, color=ps.MUTED, ls=(0, (1, 2)), lw=1.2)
fig.savefig('paper/figs/precision_rms.pdf')
