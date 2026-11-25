"""Synthesize Ca II 8542 for the first 2000 models of the test set with the
emulator (through SpectralSynthesizer, as in the inversions) and compare with
the Lightweaver database at mu=1. Produces quality/quality_caii8542.h5, read
by precision.py.

The database keeps vz(log tau=-1) as a Doppler shift, so its profiles are
compared in the observer's frame, with the emulator shifted by v_ref. Only the
wavelengths inside the shifted emulator grid for all the models are kept: the
inversions ignore the others.
"""
import h5py
import numpy as np
import torch
from astropy.io import fits

from modules.inversion import LineEmulator, SpectralSynthesizer

MODELS = '/scratch/Dropbox/GIT/DeepLearning/universal_stokes_flow/database/models/models_testing.h5'
STOKES = '/scratch1/datasets/caii_universal/synthetic_allmus_final_CaII_testing.fits'
N_MODELS = 2000
N_WAVE = 150
I_NORM = 2.59e-5            # as in caii8542.py
MU = 1.0

with fits.open(STOKES, memmap=True) as hdu:
    wave = np.array(hdu[1].data[:N_WAVE], dtype=np.float64)
    i_mu = int(np.argmin(np.abs(hdu[2].data - MU)))
    stokes_database = np.array(hdu[0].data[i_mu, :N_MODELS, :, :N_WAVE], dtype=np.float64) / I_NORM

with h5py.File(MODELS, 'r') as f:
    models = f['model'][:N_MODELS].astype(np.float64)

v_ref = models[:, 22, 4]
physical = models[:, :, [1, 4, 3, 5, 6, 7]].transpose(0, 2, 1).copy()   # T, vz, vmic, Bx, By, Bz
physical[:, 1] -= v_ref[:, None]

synth = SpectralSynthesizer([LineEmulator("weights/emulator_caii8542_pca_resmlp512_cosine_deriv.pt", wave)])
with torch.no_grad():
    stokes_emulator, valid = synth(torch.tensor(physical, dtype=torch.float32),
                                   torch.full((N_MODELS,), MU),
                                   torch.tensor(v_ref, dtype=torch.float32),
                                   torch.tensor(synth.relative_wavelengths(wave), dtype=torch.float32),
                                   None)

keep = valid.numpy().all(axis=0)
print(f"{keep.sum()} of {N_WAVE} wavelengths valid for all the models: {wave[keep][0]:.3f} - {wave[keep][-1]:.3f} A")

with h5py.File('quality/quality_caii8542.h5', 'w') as fout:
    fout.create_dataset('wavelength', data=wave[keep])
    fout.create_dataset('stokes_emulator_8542', data=stokes_emulator.numpy()[:, :, keep])
    fout.create_dataset('stokes_database_8542', data=stokes_database[:, :, keep])
