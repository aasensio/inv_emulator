import numpy as np
import matplotlib.pyplot as pl
import h5py

f_10830 = h5py.File('quality/quality_he10830.h5', 'r')
stokes_emulator_10830 = f_10830['stokes_emulator_10830'][:]
stokes_hazel_10830 = f_10830['stokes_hazel_10830'][:]
f_10830.close()

f_6301 = h5py.File('quality/quality_fe6301.h5', 'r')
stokes_emulator_6301 = f_6301['stokes_emulator_6301'][:]
stokes_hazel_6301 = f_6301['stokes_hazel_6301'][:]
f_6301.close()

pct_68 = np.percentile(stokes_emulator_10830 - stokes_hazel_10830, [50.0-68/2.0, 50.0+68/2.0], axis=0)
pct_95 = np.percentile(stokes_emulator_10830 - stokes_hazel_10830, [50.0-95/2.0, 50.0+95/2.0], axis=0)

fig, ax = pl.subplots(nrows=2, ncols=4, figsize=(11, 7), sharey='row', layout='constrained')
for i in range(4):
    ax[0, i].fill_between(np.arange(pct_95.shape[2]), pct_95[0, i ,:], pct_95[1, i ,:], color='C0', alpha=0.5)
    ax[0, i].fill_between(np.arange(pct_68.shape[2]), pct_68[0, i ,:], pct_68[1, i ,:], color='C0', alpha=0.5)

print('10830: mean=', np.mean(stokes_emulator_10830 - stokes_hazel_10830, axis=(0, 2)))
print('10830: 68% =', np.max(np.abs(pct_68), axis=(0, 2)), '95% =', np.max(np.abs(pct_95), axis=(0, 2)))

pct_68 = np.percentile(stokes_emulator_6301 - stokes_hazel_6301, [50.0-68/2.0, 50.0+68/2.0], axis=0)
pct_95 = np.percentile(stokes_emulator_6301 - stokes_hazel_6301, [50.0-95/2.0, 50.0+95/2.0], axis=0)

for i in range(4):
    ax[1, i].fill_between(np.arange(pct_95.shape[2]), pct_95[0, i ,:], pct_95[1, i ,:], color='C0', alpha=0.5)
    ax[1, i].fill_between(np.arange(pct_68.shape[2]), pct_68[0, i ,:], pct_68[1, i ,:], color='C0', alpha=0.5)

print('6301: mean=', np.mean(stokes_emulator_6301 - stokes_hazel_6301, axis=(0, 2)))
print('6301: 68% =', np.max(np.abs(pct_68), axis=(0, 2)), '95% =', np.max(np.abs(pct_95), axis=(0, 2)))