"""Analysis of the inversions of the Hinode SOT/SP (Fe I 6301/6302, AR 11967)
and the GREGOR/GRIS (Si I 10827 + He I 10830) maps carried out with
invert_fe6301_6302.py and invert_si10827_he10830.py. Produces

    paper/figs/hinode_maps.pdf            maps of the Hinode inversion
    paper/figs/hinode_stratification.pdf  stratifications in the umbra, penumbra and surroundings
    paper/figs/gris_maps.pdf              maps of the GRIS inversion (photosphere + He I slabs)
    paper/figs/gris_chromosphere.pdf      maps of the photosphere (log tau=-1) and of the two He I slabs
    paper/figs/gris_chromosphere_blos_btrans.pdf  same with B_LOS and B_trans, plus I_c and He I / Si I core images
    paper/figs/gris_analysis.pdf          telluric drift, chromospheric vs photospheric field
                                          and the fit of a pixel with a fast He I downflow
    paper/figs/gris_stokes.pdf            Stokes profiles of the four marked GRIS pixels
    paper/figs/crisp_caii8542_maps.pdf    maps of the SST/CRISP Ca II 8542 inversion
    paper/figs/crisp_caii8542_maps_azimuth.pdf  same with the field azimuth instead of B_trans

and prints the numbers quoted in the paper.

    python analysis.py
"""
import h5py
import matplotlib.pyplot as pl
import numpy as np
from scipy.ndimage import gaussian_filter

import plotstyle as ps

C_KMS = 299792.458
LAMBDA_6301, LAMBDA_6302 = 6301.5012, 6302.4936
LAMBDA_SI = 10827.089
LAMBDA_HE_RED = 10830.30
TELLURIC = 10832.0          # reference wavelength of the telluric line in the inversion
STOKES = ('I', 'Q', 'U', 'V')
DARK_IC = 0.15              # I_c / I_QS below which the Hinode maps are masked (emulators not valid there)

ps.apply()


def at(logtau, x):
    """Index of the depth point closest to log tau = x."""
    return int(np.argmin(np.abs(logtau - x)))


def field(strat, k):
    """Field strength [G] and inclination with respect to the LOS [deg] at depth index k."""
    bx, by, bz = (strat[c][:, :, k] for c in ('Bx', 'By', 'Bz'))
    B = np.sqrt(bx ** 2 + by ** 2 + bz ** 2)
    return B, np.degrees(np.arccos(np.clip(bz / np.maximum(B, 1e-3), -1.0, 1.0)))


def stokes_i_error(i_obs, i_syn):
    """Relative rms residual of Stokes I [%] of each pixel. Unlike chi2, whose noise is fixed
    in units of the quiet-Sun continuum, it does not vanish in dark regions. Wavelengths
    outside the emulators (NaN in the synthesis) are ignored."""
    return 100.0 * np.sqrt(np.nanmean((i_syn - i_obs) ** 2, axis=-1)) / i_obs.mean(axis=-1)


def show(a, img, title, cmap=ps.INTENSITY, symmetric=False, pct=(1, 99), cbar_label=None, transpose=True,
         shrink=1.0, mask=None, mask_color=ps.MUTED, **kw):
    if mask is not None:
        img = np.where(mask, np.nan, img)
        cmap = cmap.with_extremes(bad=mask_color)
    img = img.T if transpose else img
    lo, hi = np.nanpercentile(img, pct)
    if symmetric:
        hi = max(abs(lo), abs(hi))
        lo = -hi
    im = a.imshow(img, cmap=cmap, vmin=kw.pop('vmin', lo), vmax=kw.pop('vmax', hi), **kw)
    a.set_title(title)
    ps.map_axes(a)
    ps.colorbar(im, a, label=cbar_label, shrink=shrink)
    return im


# =========================================================================== #
# Hinode SOT/SP, Fe I 6301/6302
# =========================================================================== #
with h5py.File('inversions/inversion_result_fe6301_6302_nopsf.h5', 'r') as f:
    logtau = f['logtau'][:]
    strat = {k: f['stratification'][k][:] for k in f['stratification']}
    nodes = {k: f['nodes'][k][:] for k in f['nodes']}
    node_pos = {k: f['node_positions'][k][:] for k in f['node_positions']}
    chi2 = f['chi2'][:]
    v_ref = f['v_ref'][:]
    obs = f['observations'][:, :, :, 0]               # first wavelength, continuum
    eps_i = stokes_i_error(f['observations'][:, :, 0, :], f['synthetic'][:, :, 0, :])

ic = obs[:, :, 0]
ic_qs = np.median(ic[ic > 0.9])
dark = gaussian_filter(ic / ic_qs, 1.0) < DARK_IC
umbra = ic < 0.5 * ic_qs
penumbra = (ic >= 0.5 * ic_qs) & (ic < 0.9 * ic_qs)
outside = ic >= 0.9 * ic_qs
regions = {'umbra': (umbra, ps.INK), 'penumbra': (penumbra, ps.ORANGE), 'surroundings': (outside, ps.BLUE)}

k0, k1, k25 = at(logtau, 0.0), at(logtau, -1.0), at(logtau, -2.5)
B1, gamma1 = field(strat, k1)
v0 = np.median(v_ref[umbra])                          # umbra taken at rest
vlos = v_ref - v0

print('== Hinode')
print(f'  quiet-Sun continuum {ic_qs:.3f}, umbra {umbra.sum()} px, penumbra {penumbra.sum()} px, '
      f'surroundings {outside.sum()} px')
print(f'  velocity zero point (median umbral v at log tau=-1): {v0:.2f} km/s')
print(f'  chi2 per point: median {np.median(chi2):.2f}, 95% {np.percentile(chi2, 95):.1f}')
bz_lo, bz_hi = nodes['Bz'].min(axis=(0, 1)), nodes['Bz'].max(axis=(0, 1))
at_bound = (nodes['Bz'] <= bz_lo + 1.0) | (nodes['Bz'] >= bz_hi - 1.0)
print(f'  Bz node bounds: {bz_lo.round(0)} / {bz_hi.round(0)} G at log tau = {node_pos["Bz"]}')
for name, (m, _) in regions.items():
    Bk = [np.median(field(strat, k)[0][m]) for k in (k0, k1, k25)]
    print(f'  {name:12s}: chi2 median {np.median(chi2[m]):5.2f} | B(0,-1,-2.5) = {Bk[0]:5.0f} {Bk[1]:5.0f} {Bk[2]:5.0f} G'
          f' | min(gamma, 180-gamma)(-1) {np.median(np.minimum(gamma1, 180.0 - gamma1)[m]):5.1f} deg'
          f' | T(0) {np.median(strat["T"][:, :, k0][m]):5.0f} K | |v|(-1) {np.median(np.abs(vlos[m])):4.2f} km/s'
          f' | Bz nodes at bound {at_bound[m].mean(axis=0).round(2)}')
print(f'  eps_I median: ' + ' | '.join(f'{name} {np.median(eps_i[m]):.1f}%' for name, (m, _) in regions.items()))
print(f'  masked (I_c < {DARK_IC} I_QS): {100 * dark.mean():.1f}% of the map, {100 * dark[umbra].mean():.0f}% of the umbra; '
      f'unmasked umbra: eps_I {np.median(eps_i[umbra & ~dark]):.1f}%, B(-1) {np.median(B1[umbra & ~dark]):.0f} G, '
      f'masked: eps_I {np.median(eps_i[dark]):.1f}%, B(-1) {np.median(B1[dark]):.0f} G')

# --------------------------------------------------------------------------- #
# Figure: Hinode maps
# --------------------------------------------------------------------------- #
fig, ax = pl.subplots(nrows=2, ncols=3, figsize=(10, 6.4), sharex=True, sharey=True, constrained_layout=True)
show(ax[0, 0], ic, r'$I_c$')
show(ax[0, 1], strat['T'][:, :, k0], r'$T$ ($\log\tau_{500}=0$) [K]', cmap=ps.SEQ_ORANGE.reversed(), mask=dark)
show(ax[0, 2], vlos, r'$v$ ($\log\tau_{500}=-1$) [km s$^{-1}$]', cmap=ps.DIVERGING, symmetric=True, pct=(2, 98),
     mask=dark)
show(ax[1, 0], B1, r'$B$ ($\log\tau_{500}=-1$) [G]', cmap=ps.SEQ_BLUE, mask=dark)
show(ax[1, 1], gamma1, r'$\gamma$ ($\log\tau_{500}=-1$) [deg]', cmap=ps.DIVERGING, vmin=0, vmax=180, mask=dark)
show(ax[1, 2], eps_i, r'$\epsilon_I$ [%]', cmap=ps.SEQ_VIOLET, vmin=0, vmax=30)
ic_smooth = gaussian_filter(ic, 3.0)
for a in ax.flat:
    a.contour(ic_smooth.T, levels=[0.5 * ic_qs, 0.9 * ic_qs], colors=ps.ORANGE if a is ax[0, 0] else ps.INK,
              linewidths=0.6, alpha=0.9 if a is ax[0, 0] else 0.45)
for a in ax[1]:
    a.set_xlabel('x [pixel]')
for a in ax[:, 0]:
    a.set_ylabel('y [pixel]')
fig.savefig('figs/hinode_maps_nopsf.pdf', )


# =========================================================================== #
# Hinode SOT/SP, Fe I 6301/6302
# =========================================================================== #
with h5py.File('inversions/inversion_result_fe6301_6302_psf.h5', 'r') as f:
    logtau = f['logtau'][:]
    strat = {k: f['stratification'][k][:] for k in f['stratification']}
    nodes = {k: f['nodes'][k][:] for k in f['nodes']}
    node_pos = {k: f['node_positions'][k][:] for k in f['node_positions']}
    chi2 = f['chi2'][:]
    v_ref = f['v_ref'][:]
    obs = f['observations'][:, :, :, 0]               # first wavelength, continuum
    eps_i = stokes_i_error(f['observations'][:, :, 0, :], f['synthetic'][:, :, 0, :])

ic = obs[:, :, 0]
ic_qs = np.median(ic[ic > 0.9])
dark = gaussian_filter(ic / ic_qs, 1.0) < DARK_IC
umbra = ic < 0.5 * ic_qs
penumbra = (ic >= 0.5 * ic_qs) & (ic < 0.9 * ic_qs)
outside = ic >= 0.9 * ic_qs
regions = {'umbra': (umbra, ps.INK), 'penumbra': (penumbra, ps.ORANGE), 'surroundings': (outside, ps.BLUE)}

k0, k1, k25 = at(logtau, 0.0), at(logtau, -1.0), at(logtau, -2.5)
B1, gamma1 = field(strat, k1)
v0 = np.median(v_ref[umbra])                          # umbra taken at rest
vlos = v_ref - v0

print('== Hinode')
print(f'  quiet-Sun continuum {ic_qs:.3f}, umbra {umbra.sum()} px, penumbra {penumbra.sum()} px, '
      f'surroundings {outside.sum()} px')
print(f'  velocity zero point (median umbral v at log tau=-1): {v0:.2f} km/s')
print(f'  chi2 per point: median {np.median(chi2):.2f}, 95% {np.percentile(chi2, 95):.1f}')
bz_lo, bz_hi = nodes['Bz'].min(axis=(0, 1)), nodes['Bz'].max(axis=(0, 1))
at_bound = (nodes['Bz'] <= bz_lo + 1.0) | (nodes['Bz'] >= bz_hi - 1.0)
print(f'  Bz node bounds: {bz_lo.round(0)} / {bz_hi.round(0)} G at log tau = {node_pos["Bz"]}')
for name, (m, _) in regions.items():
    Bk = [np.median(field(strat, k)[0][m]) for k in (k0, k1, k25)]
    print(f'  {name:12s}: chi2 median {np.median(chi2[m]):5.2f} | B(0,-1,-2.5) = {Bk[0]:5.0f} {Bk[1]:5.0f} {Bk[2]:5.0f} G'
          f' | min(gamma, 180-gamma)(-1) {np.median(np.minimum(gamma1, 180.0 - gamma1)[m]):5.1f} deg'
          f' | T(0) {np.median(strat["T"][:, :, k0][m]):5.0f} K | |v|(-1) {np.median(np.abs(vlos[m])):4.2f} km/s'
          f' | Bz nodes at bound {at_bound[m].mean(axis=0).round(2)}')
print(f'  eps_I median: ' + ' | '.join(f'{name} {np.median(eps_i[m]):.1f}%' for name, (m, _) in regions.items()))
print(f'  masked (I_c < {DARK_IC} I_QS): {100 * dark.mean():.1f}% of the map, {100 * dark[umbra].mean():.0f}% of the umbra; '
      f'unmasked umbra: eps_I {np.median(eps_i[umbra & ~dark]):.1f}%, B(-1) {np.median(B1[umbra & ~dark]):.0f} G, '
      f'masked: eps_I {np.median(eps_i[dark]):.1f}%, B(-1) {np.median(B1[dark]):.0f} G')

# --------------------------------------------------------------------------- #
# Figure: Hinode maps
# --------------------------------------------------------------------------- #
fig, ax = pl.subplots(nrows=2, ncols=3, figsize=(10, 6.4), sharex=True, sharey=True, constrained_layout=True)
show(ax[0, 0], ic, r'$I_c$')
show(ax[0, 1], strat['T'][:, :, k0], r'$T$ ($\log\tau_{500}=0$) [K]', cmap=ps.SEQ_ORANGE.reversed(), mask=dark)
show(ax[0, 2], vlos, r'$v$ ($\log\tau_{500}=-1$) [km s$^{-1}$]', cmap=ps.DIVERGING, symmetric=True, pct=(2, 98),
     mask=dark)
show(ax[1, 0], B1, r'$B$ ($\log\tau_{500}=-1$) [G]', cmap=ps.SEQ_BLUE, mask=dark)
show(ax[1, 1], gamma1, r'$\gamma$ ($\log\tau_{500}=-1$) [deg]', cmap=ps.DIVERGING, vmin=0, vmax=180, mask=dark)
show(ax[1, 2], eps_i, r'$\epsilon_I$ [%]', cmap=ps.SEQ_VIOLET, vmin=0, vmax=30)
ic_smooth = gaussian_filter(ic, 3.0)
for a in ax.flat:
    a.contour(ic_smooth.T, levels=[0.5 * ic_qs, 0.9 * ic_qs], colors=ps.ORANGE if a is ax[0, 0] else ps.INK,
              linewidths=0.6, alpha=0.9 if a is ax[0, 0] else 0.45)
for a in ax[1]:
    a.set_xlabel('x [pixel]')
for a in ax[:, 0]:
    a.set_ylabel('y [pixel]')
fig.savefig('figs/hinode_maps_psf.pdf', )

# --------------------------------------------------------------------------- #
# Figure: Hinode stratifications per region
# --------------------------------------------------------------------------- #
depth = (logtau >= -3.0) & (logtau <= 0.5)
B_all = np.sqrt(strat['Bx'] ** 2 + strat['By'] ** 2 + strat['Bz'] ** 2)
gamma_all = np.degrees(np.arccos(np.clip(strat['Bz'] / np.maximum(B_all, 1e-3), -1.0, 1.0)))
# In the umbra the inclination is folded to [0, 90] so that both polarities can be compared
quantities = [
    (strat['T'], r'$T$ [K]'),
    (strat['vz'] - v0, r'$v$ [km s$^{-1}$]'),
    (B_all, r'$B$ [G]'),
    (np.minimum(gamma_all, 180.0 - gamma_all), r'$\min(\gamma, 180^\circ-\gamma)$ [deg]'),
]
# fig, ax = pl.subplots(nrows=1, ncols=4, figsize=(10, 2.9), sharex=True)
# for a, (q, lab) in zip(ax, quantities):
#     for name, (m, color) in regions.items():
#         p = np.percentile(q[m][:, depth], [16, 50, 84], axis=0)
#         a.fill_between(logtau[depth], p[0], p[2], color=color, alpha=0.12, lw=0)
#         a.plot(logtau[depth], p[1], color=color, lw=1.8, label=name)
#     a.set_ylabel(lab)
#     a.set_xlabel(r'$\log\tau_{500}$')
#     a.invert_xaxis()
# # largest |Bz| reachable within the bounds of the nodes (linear stratification between the two nodes)
# order = np.argsort(node_pos['Bz'])
# bz_max = np.interp(logtau, node_pos['Bz'][order], np.minimum(-bz_lo, bz_hi)[order])
# ax[2].plot(logtau[depth], bz_max[depth], color=ps.MUTED, ls=(0, (1, 2)), lw=1.2, label=r'max. $|B_z|$')
# fig.legend(*ax[2].get_legend_handles_labels(), loc='outside upper center', ncols=4)
# fig.savefig('figs/hinode_stratification.pdf')

# =========================================================================== #
# GREGOR/GRIS, Si I 10827 + He I 10830
# =========================================================================== #
with h5py.File('inversions/inversion_result_si10827_he10830_l1.h5', 'r') as f:
    logtau_g = f['logtau'][:]
    strat_g = {k: f['stratification'][k][:] for k in f['stratification']}
    slab = {k: f['nodes'][k][:, :, 0] for k in f['nodes'] if f['nodes'][k].shape[2] == 1}
    chi2_g = f['chi2'][:]
    v_ref_g = f['v_ref'][:]
    wave_g = f['wavelength'][:]
    obs_g = f['observations'][:]
    syn_g = f['synthetic'][:]

nx, ny = chi2_g.shape                                 # (along the slit, scan step)
k1g = at(logtau_g, -1.0)
ic_g = obs_g[:, :, 0, 0]

# Wavelength drift during the scan, measured by the telluric line: median along the slit
drift = np.median(slab['Tel_shift'], axis=0)          # (ny,) A
drift_kms = C_KMS * (drift - np.median(drift)) / TELLURIC
# all velocities referred to the median (drift-corrected) photospheric velocity
v0g = np.median(v_ref_g - drift_kms[None, :])
v_phot = v_ref_g - drift_kms[None, :] - v0g
v_he1 = slab['He1_v'] - drift_kms[None, :] - v0g
v_he2 = slab['He2_v'] - drift_kms[None, :] - v0g
vref_col = np.median(v_ref_g, axis=0)
slope, intercept = np.polyfit(drift_kms, vref_col, 1)

tau_he = slab['He1_tau'] + slab['He2_tau']
bz_he = (slab['He1_tau'] * slab['He1_Bz'] + slab['He2_tau'] * slab['He2_Bz']) / tau_he
bz_ph = strat_g['Bz'][:, :, k1g]
fit = np.polyfit(bz_ph.ravel(), bz_he.ravel(), 1)
strong = np.abs(bz_ph) > 500.0
fast = slab['He2_v'] > 10.0

print('== GRIS')
print(f'  velocity zero point (median photospheric v_ref after drift correction): {v0g:.2f} km/s')
print(f'  chi2 per point: median {np.median(chi2_g):.2f}, 95% {np.percentile(chi2_g, 95):.2f}')
print(f'  telluric drift along the scan: {np.ptp(drift) * 1e3:.0f} mA = {np.ptp(drift_kms):.2f} km/s; '
      f'correlation with median v_ref per scan step {np.corrcoef(drift_kms, vref_col)[0, 1]:.2f}, slope {slope:.2f}')
print(f'  Bz(He, tau-weighted) vs Bz(phot, log tau=-1): correlation {np.corrcoef(bz_ph.ravel(), bz_he.ravel())[0, 1]:.2f}, '
      f'slope {fit[0]:.2f}; median |Bz(He)|/|Bz(phot)| for |Bz(phot)|>500 G: '
      f'{np.median(np.abs(bz_he[strong]) / np.abs(bz_ph[strong])):.2f}')
for p, v in (('He1', v_he1), ('He2', v_he2)):
    print(f'  {p}: v median {np.median(v):5.2f} km/s, tau median {np.median(slab[p + "_tau"]):.2f}, '
          f'deltav median {np.median(slab[p + "_deltav"]):.1f} km/s, |B| median '
          f'{np.median(np.sqrt(slab[p + "_Bx"] ** 2 + slab[p + "_By"] ** 2 + slab[p + "_Bz"] ** 2)):.0f} G')
print(f'  He2 v > 10 km/s in {fast.mean() * 100:.1f}% of the pixels ({(slab["He2_v"] > 19.9).mean() * 100:.1f}% at the '
      f'20 km/s bound); there deltav = {np.median(slab["He2_deltav"][fast]):.1f} km/s (vs '
      f'{np.median(slab["He2_deltav"][~fast]):.1f}), chi2 = {np.median(chi2_g[fast]):.2f} (vs {np.median(chi2_g[~fast]):.2f})')

# Example pixels: (A) two well separated components with a good fit, (B) the
# fastest downflow among those with a significant second component
separation = (slab['He2_v'] - slab['He1_v']) * np.minimum(slab['He1_tau'], slab['He2_tau'])
good = (slab['He2_v'] < 18.0) & (slab['He1_tau'] > 0.3) & (slab['He2_tau'] > 0.3) & (chi2_g < 1.8)

# pixels = {'A': np.unravel_index(np.argmax(np.where(good, separation, -np.inf)), (nx, ny)),
#           'B': np.unravel_index(np.argmax(np.where(slab['He2_tau'] > 0.3, slab['He2_v'], -np.inf)), (nx, ny))}
# # (C) the strongest photospheric field and (D) the strongest transverse field of the
# # first He I slab, both with a good fit and away from the borders of the map
# bt_he1 = np.sqrt(slab['He1_Bx'] ** 2 + slab['He1_By'] ** 2)
# inner = np.zeros((nx, ny), dtype=bool)
# inner[10:-10, 10:-10] = True
# ok = inner & (chi2_g < 1.8)
# pixels['C'] = np.unravel_index(np.argmax(np.where(ok, np.abs(bz_ph), -np.inf)), (nx, ny))
# pixels['D'] = np.unravel_index(np.argmax(np.where(ok & (slab['He1_tau'] > 0.3), bt_he1, -np.inf)), (nx, ny))

pixels = {'A': (85, 66), 'B': (25, 22), 'C': (297, 11)}  # manually selected pixels

for name, (ix, iy) in pixels.items():
    print(f'  pixel {name} ({ix}, {iy}): v1={v_he1[ix, iy]:.1f} tau1={slab["He1_tau"][ix, iy]:.2f}; '
          f'v2={v_he2[ix, iy]:.1f} tau2={slab["He2_tau"][ix, iy]:.2f}; chi2={chi2_g[ix, iy]:.2f}')

def mark_pixels(a):
    for name, (ix, iy) in pixels.items():
        ps.mark(a, iy, ix, name, left=(name == 'D'))       # D lies close to the right border


# --------------------------------------------------------------------------- #
# Figure: GRIS maps
# --------------------------------------------------------------------------- #
# fig, ax = pl.subplots(nrows=1, ncols=7, figsize=(10, 3.6), sharex=True, sharey=True)
# kw = dict(transpose=False, shrink=0.6)
# show(ax[0], ic_g, r'$I_c$', **kw)
# show(ax[1], bz_ph, r'$B_z^\mathrm{ph}$ [G]', cmap=ps.DIVERGING, symmetric=True, **kw)
# show(ax[2], v_phot, r'$v^\mathrm{ph}$ [km s$^{-1}$]', cmap=ps.DIVERGING, symmetric=True, **kw)
# show(ax[3], tau_he, r'$\tau_1+\tau_2$', cmap=ps.SEQ_BLUE, **kw)
# show(ax[4], bz_he, r'$B_z^\mathrm{He}$ [G]', cmap=ps.DIVERGING, symmetric=True, **kw)
# show(ax[5], v_he1, r'$v_1^\mathrm{He}$ [km s$^{-1}$]', cmap=ps.DIVERGING, symmetric=True, **kw)
# show(ax[6], v_he2, r'$v_2^\mathrm{He}$ [km s$^{-1}$]', cmap=ps.DIVERGING, vmin=-20, vmax=20, **kw)
# for a in ax:
#     mark_pixels(a)
#     a.set_xlabel('scan step')
# ax[0].set_ylabel('position along the slit [pixel]')
# fig.savefig('figs/gris_maps.pdf', )


# --------------------------------------------------------------------------- #
# Figure: GRIS maps, photosphere at log tau=-1 and one row per He I slab
# --------------------------------------------------------------------------- #
fig, ax = pl.subplots(nrows=3, ncols=5, figsize=(10, 12), sharex=True, sharey=True)
kw = dict(transpose=False, shrink=0.8)
for j, c in enumerate(('Bx', 'By', 'Bz')):
    show(ax[0, j], strat_g[c][:, :, k1g], rf'$B_{c[1]}^\mathrm{{ph}}$ [G]', cmap=ps.DIVERGING, symmetric=True, **kw)
show(ax[0, 3], v_phot, r'$v^\mathrm{ph}$ [km s$^{-1}$]', cmap=ps.DIVERGING, symmetric=True, **kw)
show(ax[0, 4], strat_g['T'][:, :, k1g], r'$T^\mathrm{ph}$ [K]', cmap=ps.SEQ_ORANGE.reversed(), **kw)
for i, (p, v) in enumerate((('He1', v_he1), ('He2', v_he2)), start=1):

    mask = (slab[p + '_tau'] > 0.1)

    print(f'  {p}: {mask.mean() * 100:.1f}% of the pixels have tau > 0.1')

    show(ax[i, 0], mask * slab[p + '_Bx'], rf'$B_x^{{\mathrm{{He}},{i}}}$ [G]', cmap=ps.DIVERGING, symmetric=True, **kw)
    show(ax[i, 1], mask * slab[p + '_By'], rf'$B_y^{{\mathrm{{He}},{i}}}$ [G]', cmap=ps.DIVERGING, symmetric=True, **kw)
    show(ax[i, 2], mask * slab[p + '_Bz'], rf'$B_z^{{\mathrm{{He}},{i}}}$ [G]', cmap=ps.DIVERGING, symmetric=True, **kw)
    show(ax[i, 3], mask * v, rf'$v_{i}^\mathrm{{He}}$ [km s$^{{-1}}$]', cmap=ps.DIVERGING, symmetric=True, **kw)
    if p == 'He1':
        show(ax[i, 4], mask * slab[p + '_tau'], rf'$\tau_{i}$', cmap=ps.SEQ_BLUE, **kw)
    else:
        show(ax[i, 4], mask * slab[p + '_tau'], rf'$\tau_{i}$', cmap=ps.SEQ_BLUE, vmax=0.85, **kw)
for a in ax.ravel():
    mark_pixels(a)
for a in ax[-1]:
    a.set_xlabel('scan step')
for a in ax[:, 0]:
    a.set_ylabel('position along the slit [pixel]')
fig.savefig('figs/gris_chromosphere.pdf', )


# --------------------------------------------------------------------------- #
# Figure: as gris_chromosphere.pdf but with B_LOS and B_trans, and the first
# column showing the continuum and monochromatic images in the line cores
# --------------------------------------------------------------------------- #
def line_core(lam0, width=0.5):
    """Wavelength index of the minimum of the map-averaged Stokes I within +-width A of lam0."""
    sel = np.where(np.abs(wave_g - lam0) < width)[0]
    return sel[np.argmin(obs_g[:, :, 0, sel].mean(axis=(0, 1)))]


i_si, i_he = line_core(LAMBDA_SI), line_core(LAMBDA_HE_RED)
print(f'  line cores: Si I at {wave_g[i_si]:.3f} A, He I red at {wave_g[i_he]:.3f} A')

fig, ax = pl.subplots(nrows=3, ncols=5, figsize=(10, 12), sharex=True, sharey=True)
kw = dict(transpose=False, shrink=0.8)
show(ax[0, 0], ic_g, r'$I_c$', **kw)
show(ax[1, 0], obs_g[:, :, 0, i_he], rf'$I$ (He I, {wave_g[i_he]:.2f} $\AA$)', **kw)
show(ax[2, 0], obs_g[:, :, 0, i_si], rf'$I$ (Si I, {wave_g[i_si]:.2f} $\AA$)', **kw)
bx, by, bz = (strat_g[c][:, :, k1g] for c in ('Bx', 'By', 'Bz'))
show(ax[0, 1], bz, r'$B_\mathrm{LOS}^\mathrm{ph}$ [G]', cmap=ps.DIVERGING, symmetric=True, **kw)
show(ax[0, 2], np.sqrt(bx ** 2 + by ** 2), r'$B_\mathrm{trans}^\mathrm{ph}$ [G]', cmap=ps.SEQ_BLUE, **kw)
show(ax[0, 3], v_phot, r'$v^\mathrm{ph}$ [km s$^{-1}$]', cmap=ps.DIVERGING, symmetric=True, **kw)
show(ax[0, 4], strat_g['T'][:, :, k1g], r'$T^\mathrm{ph}$ [K]', cmap=ps.SEQ_ORANGE.reversed(), **kw)
for i, (p, v) in enumerate((('He1', v_he1), ('He2', v_he2)), start=1):
    mask = (slab[p + '_tau'] > 0.1)
    bt = np.sqrt(slab[p + '_Bx'] ** 2 + slab[p + '_By'] ** 2)
    show(ax[i, 1], mask * slab[p + '_Bz'], rf'$B_\mathrm{{LOS}}^{{\mathrm{{He}},{i}}}$ [G]', cmap=ps.DIVERGING,
         symmetric=True, **kw)
    show(ax[i, 2], mask * bt, rf'$B_\mathrm{{trans}}^{{\mathrm{{He}},{i}}}$ [G]', cmap=ps.SEQ_BLUE, **kw)
    show(ax[i, 3], mask * v, rf'$v_{i}^\mathrm{{He}}$ [km s$^{{-1}}$]', cmap=ps.DIVERGING, symmetric=True, **kw)
    if p == 'He1':
        show(ax[i, 4], mask * slab[p + '_tau'], rf'$\tau_{i}$', cmap=ps.SEQ_BLUE, **kw)
    else:
        show(ax[i, 4], mask * slab[p + '_tau'], rf'$\tau_{i}$', cmap=ps.SEQ_BLUE, vmax=0.85, **kw)
for a in ax.ravel():
    mark_pixels(a)
for a in ax[-1]:
    a.set_xlabel('scan step')
for a in ax[:, 0]:
    a.set_ylabel('position along the slit [pixel]')
fig.savefig('figs/gris_chromosphere_blos_btrans.pdf', )

def spectrum(a, ix, iy, st):
    """Observed and synthetic Stokes parameter st of pixel (ix, iy), with the position of
    the red component of each He I slab in the observer's frame."""
    if st > 0:
        ps.zero_line(a)
    for p, ls in (('He1', '-'), ('He2', (0, (2, 2)))):
        a.axvline(LAMBDA_HE_RED * (1.0 + slab[p + '_v'][ix, iy] / C_KMS), color=ps.MUTED, ls=ls, lw=0.8, zorder=1)
    a.plot(wave_g, obs_g[ix, iy, st], color=ps.INK2, lw=0.9, alpha=0.85, label='observed')
    a.plot(wave_g, syn_g[ix, iy, st], color=ps.ORANGE, lw=1.6, label='synthetic')
    a.set_xlim(wave_g[0], wave_g[-1])
    a.set_xticks([10826, 10829, 10832])


# --------------------------------------------------------------------------- #
# Figure: telluric drift, chromospheric vs photospheric field, example fit
# --------------------------------------------------------------------------- #
# fig = pl.figure(figsize=(10, 8.4))
# gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 1, 1])

# a = fig.add_subplot(gs[0, 0])
# ps.zero_line(a)
# a.plot(np.arange(ny), drift_kms, color=ps.BLUE, label='telluric line')
# a.plot(np.arange(ny), vref_col - np.median(vref_col), color=ps.ORANGE, label='Si I, median along the slit')
# a.set_xlabel('scan step')
# a.set_ylabel(r'velocity shift [km s$^{-1}$]')
# a.legend()
# a.set_title('(a)  telluric drift')

# a = fig.add_subplot(gs[0, 1])
# lim = 1500
# a.hist2d(bz_ph.ravel(), bz_he.ravel(), bins=120, range=[[-lim, lim], [-lim, lim]], cmap=ps.SEQ_BLUE, cmin=1,
#          norm='log')
# x = np.array([-lim, lim])
# a.plot(x, x, color=ps.MUTED, ls=(0, (1, 2)), lw=1.2, label='1:1')
# a.plot(x, np.polyval(fit, x), color=ps.ORANGE, lw=1.6, label=rf'slope {fit[0]:.2f}')
# a.set_xlabel(r'$B_z^\mathrm{ph}$ ($\log\tau_{500}=-1$) [G]')
# a.set_ylabel(r'$B_z^\mathrm{He}$ [G]')
# a.set_aspect('equal')
# a.legend(loc='upper left')
# a.set_title('(b)  chromospheric vs photospheric field')

# for row, name in enumerate(('A', 'B')):
#     ix, iy = pixels[name]
#     for j, st in enumerate((0, 3)):
#         a = fig.add_subplot(gs[row + 1, j])
#         spectrum(a, ix, iy, st)
#         if row == 1:
#             a.set_xlabel(r'wavelength [$\AA$]')
#         a.set_ylabel('$I/I_c$' if st == 0 else f'${STOKES[st]}/I_c$')
#         a.set_title(f'({"cdef"[2 * row + j]})  pixel {name}, Stokes {STOKES[st]}')
#         if row == 0 and j == 0:
#             a.legend(loc='lower left')
# fig.savefig('figs/gris_analysis.pdf')

# --------------------------------------------------------------------------- #
# Figure: Stokes profiles of the marked pixels, one row per pixel
# --------------------------------------------------------------------------- #
fig, ax = pl.subplots(nrows=len(pixels), ncols=4, figsize=(12, 7), sharex=True, constrained_layout=True)
for row, (name, (ix, iy)) in enumerate(pixels.items()):
    for st in range(4):
        a = ax[row, st]
        spectrum(a, ix, iy, st)
        if row == 0:
            a.set_title(f'Stokes {STOKES[st]}', fontweight='bold')
        if st == 0:
            a.set_ylabel(f'pixel {name}  ({ix}, {iy})\n$I/I_c$')
        else:
            a.set_ylabel(f'${STOKES[st]}/I_c$')
        if row == len(pixels) - 1:
            a.set_xlabel(r'wavelength [$\AA$]')

        if st == 0:
                a.set_ylim([0.0, 1.0])
        if st == 1:
            a.set_ylim([-0.01, 0.01])
        if st == 2:
            a.set_ylim([-0.01, 0.01])
        if st == 3:
            a.set_ylim([-0.1, 0.1])
ax[0, 0].legend(loc='lower left')
fig.savefig('figs/gris_stokes.pdf')


# =========================================================================== #
# SST/CRISP, Ca II 8542
# =========================================================================== #
LAMBDA_8542 = 8542.09
LOGTAU_PHOT, LOGTAU_CHROM = -1.0, -4.5
BTRANS_MIN = 150.0          # B_trans [G] below which the azimuth is masked (spatially incoherent, noise dominated)

with h5py.File('inversions/inversion_result_caii8542.h5', 'r') as f:
    ca_logtau = f['logtau'][:]
    ca_wave = f['wavelength'][:]
    kp, kc = at(ca_logtau, LOGTAU_PHOT), at(ca_logtau, LOGTAU_CHROM)
    ca_T_phot = f['stratification/T'][:, :, kp]
    ca_T_chrom = f['stratification/T'][:, :, kc]
    ca_v_phot = f['stratification/vz'][:, :, kp]
    ca_v_chrom = f['stratification/vz'][:, :, kc]
    ca_strat = {c: f['stratification'][c][:, :, kc:kc + 1] for c in ('Bx', 'By', 'Bz')}
    ca_chi2 = f['chi2'][:]
    ca_obs_i = f['observations'][:, :, 0, :]
    ca_eps_i = stokes_i_error(ca_obs_i, f['synthetic'][:, :, 0, :])

ca_wing = ca_obs_i[:, :, 0]                                   # bluest point (-0.67 A), proxy of the continuum
ca_core = ca_obs_i[:, :, np.argmin(np.abs(ca_wave - LAMBDA_8542))]
_, ca_gamma = field(ca_strat, 0)
ca_b = np.sqrt(ca_strat['Bx'][:, :, 0]**2 + ca_strat['By'][:, :, 0]**2 + ca_strat['Bz'][:, :, 0]**2)
ca_bz = ca_strat['Bz'][:, :, 0]
ca_bperp = np.sqrt(ca_strat['Bx'][:, :, 0]**2 + ca_strat['By'][:, :, 0]**2)
# azimuth in the plane of the sky [deg], counterclockwise from +x, folded to [0, 180) (180 deg ambiguity)
ca_azi = np.degrees(np.arctan2(ca_strat['By'][:, :, 0], ca_strat['Bx'][:, :, 0])) % 180.0
ca_weak = ca_bperp < BTRANS_MIN
ca_v0 = np.median(ca_v_phot)                                  # mean photosphere taken at rest
ca_vlos = ca_v_chrom - ca_v0

print('== CRISP Ca II 8542')
print(f'  velocity zero point (median v at log tau={LOGTAU_PHOT}): {ca_v0:.2f} km/s')
print(f'  chi2 per point: median {np.median(ca_chi2):.2f}, 95% {np.percentile(ca_chi2, 95):.1f}')
print(f'  T({LOGTAU_PHOT}) median {np.median(ca_T_phot):.0f} K, T({LOGTAU_CHROM}) median {np.median(ca_T_chrom):.0f} K')
print(f'  eps_I median {np.median(ca_eps_i):.1f}%, 95% {np.percentile(ca_eps_i, 95):.1f}%')
print(f'  B_trans({LOGTAU_CHROM}) median {np.median(ca_bperp):.0f} G; azimuth masked (B_trans < {BTRANS_MIN:.0f} G) '
      f'in {100 * ca_weak.mean():.0f}% of the pixels')

# --------------------------------------------------------------------------- #
# Figure: Ca II 8542 maps
# --------------------------------------------------------------------------- #
tau_lbl = lambda x: rf'($\log\tau_{{500}}={x:g}$)'
# Bz with B_trans (first figure) or with the azimuth of the field (second figure)
for suffix in ('', '_azimuth'):
    fig, ax = pl.subplots(nrows=2, ncols=4, figsize=(17, 6.4), sharex=True, sharey=True, constrained_layout=True)
    show(ax[0, 0], ca_wing, rf'$I$ ({ca_wave[0]:.2f} $\AA$)')
    show(ax[0, 1], ca_core, rf'$I$ (line core, {LAMBDA_8542:.2f} $\AA$)')
    show(ax[0, 2], ca_T_phot, rf'$T$ {tau_lbl(LOGTAU_PHOT)} [K]', cmap=ps.SEQ_ORANGE.reversed())
    show(ax[0, 3], ca_T_chrom, rf'$T$ {tau_lbl(LOGTAU_CHROM)} [K]', cmap=ps.SEQ_ORANGE.reversed())
    show(ax[1, 0], ca_vlos, rf'$v$ {tau_lbl(LOGTAU_CHROM)} [km s$^{{-1}}$]', cmap=ps.DIVERGING, symmetric=True,
         pct=(2, 98))
    show(ax[1, 1], ca_bz, rf'$B_z$ {tau_lbl(LOGTAU_CHROM)} [G]', cmap=ps.DIVERGING, symmetric=True, pct=(1, 99))
    if suffix:
        show(ax[1, 2], ca_azi, rf'$\phi_B$ {tau_lbl(LOGTAU_CHROM)} [deg]', cmap=pl.get_cmap('twilight'), vmin=0,
             vmax=180, mask=ca_weak, mask_color='white')
    else:
        show(ax[1, 2], ca_bperp, rf'$B_\mathrm{{trans}}$ {tau_lbl(LOGTAU_CHROM)} [G]', cmap=ps.SEQ_BLUE, pct=(1, 99))
    show(ax[1, 3], ca_eps_i, r'$\epsilon_I$ [%]', cmap=ps.SEQ_VIOLET, vmin=0, vmax=np.percentile(ca_eps_i, 99))
    for a in ax[-1]:
        a.set_xlabel('x [pixel]')
    for a in ax[:, 0]:
        a.set_ylabel('y [pixel]')
    fig.savefig(f'figs/crisp_caii8542_maps{suffix}.pdf', )


