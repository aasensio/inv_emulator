"""Synthesize Si I 10827 + He I 10830 for one photospheric stratification and
one set of He I slab parameters with the emulators (SiHeSynthesizer) and
compare with Hazel2 (photosphere -> chromospheric slab)."""
import h5py
import matplotlib.pyplot as pl
import numpy as np
import torch


import hazel
from modules.hei import SiHeSynthesizer
from modules.hei_pca import HeIPCAEmulator
from modules.inversion import LineEmulator, SpectralSynthesizer
from tqdm import tqdm

MODELS = '/scratch/Dropbox/GIT/DeepLearning/universal_stokes_flow/database/models/models_testing.h5'
WAVE = np.linspace(10824.3, 10832.7, 421)
wave_fe = np.linspace(6300.8921, 6303.2671, 150)
MU = 1.0
HEIGHT = 3.0
LAMBDA_6301, LAMBDA_6302 = 6301.5012, 6302.4936

    
# Photospheric model (80, 8): logtau, T, Pe, vmic, v, Bx, By, Bz (v = 0 at log tau = -1 as in training)
fin = h5py.File(MODELS, 'r')

synth = SiHeSynthesizer(SpectralSynthesizer([LineEmulator('weights/emulator_si10827_pca_resmlp512_cosine_deriv.pt', np.linspace(10824.0, 10833.0, 350))]),
                        HeIPCAEmulator.load('weights/emulator_he10830_pca.pt'), step=0.02)

fe_synth = SpectralSynthesizer(
        [LineEmulator("weights/emulator_fe6301_pca_resmlp512_cosine_deriv.pt", wave_fe[wave_fe - LAMBDA_6301 < 0.5]),
        LineEmulator("weights/emulator_fe6302_pca_resmlp512_cosine_deriv.pt", wave_fe[wave_fe - LAMBDA_6302 > -0.5])],
        psf=None,            # e.g. {"fwhm": 0.025} (A) or (dlambda, profile) arrays
        telluric=None,
    )

# --- Hazel2 10830 ---
mod = hazel.Model(working_mode='synthesis', verbose=False)
mod.add_spectral({'Name': 'spec1', 'Wavelength': [WAVE[0], WAVE[-1], len(WAVE)], 'topology': 'ph1->ch1',
                'LOS': [np.degrees(np.arccos(MU)), 0.0, 90.0], 'Boundary condition': [1.0, 0.0, 0.0, 0.0]}, index=0)
mod.add_photosphere({'Name': 'ph1', 'Spectral region': 'spec1', 'Spectral lines': [300],
                    'Wavelength': [WAVE[0], WAVE[-1]]})
mod.add_chromosphere({'Name': 'ch1', 'Spectral region': 'spec1', 'Height': HEIGHT, 'Line': '10830',
                    'Wavelength': [WAVE[0], WAVE[-1]], 'Reference frame': 'vertical'})
mod.setup()



stokes_emulator_10830 = []
stokes_hazel_10830 = []


Bx_scale = 500.0
By_scale = 500.0
Bz_scale = 500.0
dv_range = [4.0, 18.0]

for loop in tqdm(range(2000)):
     
    model = fin['model'][loop].astype(np.float64)
        
    Bx = np.random.normal(loc=0, scale=500.0, size=1)[0]
    By = np.random.normal(loc=0, scale=500.0, size=1)[0]
    Bz = np.random.normal(loc=0, scale=500.0, size=1)[0]
    dv = np.random.uniform(4.0, 18.0, size=1)[0]
    a = np.random.uniform(0.0, 0.2, size=1)[0]
    tau = np.random.uniform(0.1, 2.0, size=1)[0]
    v = np.random.uniform(-5.0, 5.0, size=1)[0]
    
    # He I slab (field in G, Hazel's vertical frame; velocities in km/s; height in arcsec)
    he = dict(He_Bx=Bx, He_By=By, He_Bz=Bz, He_tau=tau, He_v=v, He_deltav=dv, He_a=a,
            He_beta=1.0, He_height=3.0)

    v_ref = model[22, 4]
    physical = model[:, [1, 4, 3, 5, 6, 7]].T.copy()   # T, vz, vmic, Bx, By, Bz
    physical[1] -= v_ref
    
    # He 10830
    with torch.no_grad():
        stokes_em, _ = synth(torch.tensor(physical[None], dtype=torch.float32), torch.tensor([MU]),
                            torch.tensor([v_ref], dtype=torch.float32),
                            torch.tensor(synth.relative_wavelengths(WAVE), dtype=torch.float32),
                            {k: torch.tensor([v], dtype=torch.float32) for k, v in he.items()})
    stokes = stokes_em[0].numpy()

    stokes_emulator_10830.append(stokes[None, :, :].copy())


    # He 10830    
    mod.atmospheres['ph1'].set_parameters(model, 1.0, 0.0)
    mod.atmospheres['ch1'].set_parameters([he[k] for k in ('He_Bx', 'He_By', 'He_Bz', 'He_tau', 'He_v', 'He_deltav',
                                                        'He_beta', 'He_a')], 1.0)
    mod.synthesize()
    stokes = mod.spectrum['spec1'].stokes

    stokes_hazel_10830.append(stokes[None, :, :].copy())
    

stokes_emulator_10830 = np.concatenate(stokes_emulator_10830, axis=0)
stokes_hazel_10830 = np.concatenate(stokes_hazel_10830, axis=0)

fin.close()

fout = h5py.File('quality/quality_he10830.h5', 'w')
fout.create_dataset('stokes_emulator_10830', data=stokes_emulator_10830)
fout.create_dataset('stokes_hazel_10830', data=stokes_hazel_10830)
fout.close()

# fig, ax = pl.subplots(2, 2, figsize=(11, 7), sharex=True, layout='constrained')

# LAMBDA_6301, LAMBDA_6302 = 6301.5012, 6302.4936

# for loop in range(1):

#     MODELS = '/scratch/Dropbox/GIT/DeepLearning/universal_stokes_flow/database/models/models_validation.h5'
#     wave_fe = np.linspace(6300.8921, 6303.2671, 150)
#     MU = 1.0

#     # Photospheric model (80, 8): logtau, T, Pe, vmic, v, Bx, By, Bz (v = 0 at log tau = -1 as in training)
#     with h5py.File(MODELS, 'r') as f:
#         model = f['model'][123].astype(np.float64)
#     model[:, 4] += loop - model[22, 4]          # 1 km/s at log tau = -1

#     # --- emulators ---
#     fe_synth = SpectralSynthesizer(
#         [LineEmulator("weights/emulator_fe6301_pca.pt", wave_fe[wave_fe - LAMBDA_6301 < 0.5]),
#         LineEmulator("weights/emulator_fe6302_pca.pt", wave_fe[wave_fe - LAMBDA_6302 > -0.5])],
#         psf=None,            # e.g. {"fwhm": 0.025} (A) or (dlambda, profile) arrays
#         telluric=None,
#     )

#     v_ref = model[22, 4]
#     physical = model[:, [1, 4, 3, 5, 6, 7]].T.copy()   # T, vz, vmic, Bx, By, Bz
#     physical[1] -= v_ref
#     with torch.no_grad():
#         stokes_em, _ = fe_synth(torch.tensor(physical[None], dtype=torch.float32), torch.tensor([MU]),
#                             torch.tensor([v_ref], dtype=torch.float32),
#                             torch.tensor(fe_synth.relative_wavelengths(wave_fe), dtype=torch.float32),
#                             None)
#     stokes_em = stokes_em[0].numpy()

#     # --- Hazel2 ---
#     mod = hazel.Model(working_mode='synthesis', verbose=False)
#     mod.add_spectral({'Name': 'spec1', 'Wavelength': [wave_fe[0], wave_fe[-1], len(wave_fe)], 'topology': 'ph1',
#                     'LOS': [np.degrees(np.arccos(MU)), 0.0, 90.0], 'Boundary condition': [1.0, 0.0, 0.0, 0.0]}, index=0)
#     mod.add_photosphere({'Name': 'ph1', 'Spectral region': 'spec1', 'Spectral lines': [200, 201],
#                         'Wavelength': [wave_fe[0], wave_fe[-1]]})
#     mod.setup()
#     mod.atmospheres['ph1'].set_parameters(model, 1.0, 0.0)

#     mod.synthesize()
#     stokes_hz = mod.spectrum['spec1'].stokes


#     # --- plot ---
#     for i, (a, lab) in enumerate(zip(ax.flat, 'IQUV')):
#         a.plot(wave_fe, stokes_hz[i], color='#2a78d6', lw=1.5, label='Hazel2')  
#         a.plot(wave_fe, stokes_em[i], color='#eb6834', lw=1.2, ls='--', label='emulators')
#         a.set_ylabel(lab if i == 0 else f'{lab}/I$_c$')        
#     for a in ax[1]:
#         a.set_xlabel('wavelength [Å]')
#     ax[0, 0].legend(frameon=False)

#     pl.show()
