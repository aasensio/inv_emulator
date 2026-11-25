# import hazel

# which = 'testing'

# if which == 'training':
#     filename = 'config_training.ini'

# if which == 'validation':
#     filename = 'config_validation.ini'

# if which == 'testing':
#     filename = 'config_testing.ini'

# iterator = hazel.Iterator(use_mpi=True)
# rank = iterator.get_rank()
# mod = hazel.Model(filename, rank=rank)
# iterator.use_model(model=mod)
# iterator.run_all_pixels()

import concurrent.futures
import h5py
import numpy as np
import hazel
from tqdm import tqdm

# Set verbose=False inside workers to prevent stdout collisions
mod = hazel.Model(working_mode='synthesis', verbose=False)
dict_spec = {
    'Name': 'spec1', 
    'Wavelength': [6300.8921, 6303.2671, 112], 
    'topology': 'ph1',
    'LOS': [0.0, 0.0, 90.0], 
    'Boundary condition': [1.0, 0.0, 0.0, 0.0]
}
mod.add_spectral(dict_spec, 0)

mod.add_photosphere({
    'Name': 'ph1', 
    'Spectral region': 'spec1', 
    'Spectral lines': [200, 201],
    'Wavelength': [6300.8921, 6303.2671]
})

mod.setup()

def synthesize_single(args):
    """
    Worker function to run synthesis on a single model index.
    """
    idx, model_i, ff_i, vmac_i, mu_i = args

    theta = np.arccos(mu_i) * 180.0 / np.pi  # Convert mu to theta in degrees
      
    mod.atmospheres['ph1'].set_parameters(model_i, ff_i, vmac_i)    
    mod.spectrum['spec1'].set_los([theta, 0., 90.])
    mod.synthesize()

    # Extract the synthesized Stokes profiles
    wvl, stokes = mod.spectrum['spec1'].wavelength_axis, mod.spectrum['spec1'].stokes
    return idx, wvl, stokes


if __name__ == '__main__':

    which = 'all'

    if which == 'all':
        which = ['testing', 'validation', 'training']
        # which = ['training']
    else:
        which = [which]

    mus = [1.0, 0.8, 0.6, 0.4, 0.2]
    
    for w in which:

        fout = h5py.File(f'stokes_{w}.h5', 'w')

        # 1. Load data into memory in the main process
        with h5py.File(f'../models/models_{w}.h5', 'r') as f:
            models = f['model'][:]  # Array of shape (N, ...)
            ffs = f['ff'][:]        # Array of shape (N, ...)
            vmacs = f['vmac'][:]    # Array of shape (N, ...)
        
        n_models = models.shape[0]
        
        print(f"Loaded {n_models} models for parallel synthesis.")

        all_stokes = []

        for i, mu in enumerate(mus):

            # Package tasks into argument tuples
            tasks = [(i, models[i], ffs[i], vmacs[i], mu) for i in range(n_models)]

            stokes_all = {}

            # 2. Run synthesis in parallel across available CPU cores
            with concurrent.futures.ProcessPoolExecutor() as executor:
                # Submit tasks to process pool
                futures = [executor.submit(synthesize_single, task) for task in tasks]

                # Retrieve results as they complete
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Synthesizing spectra", unit="pixel"):
                    idx, wvl, stokes = future.result()
                    stokes_all[idx] = stokes.copy()

            if i == 0:
                fout.create_dataset('wavelength', data=wvl, dtype=np.float32)
                fout.create_dataset('mu', data=np.array(mus), dtype=np.float32)
                fout.create_dataset('stokes', shape=(len(mus), n_models, 4, len(wvl)), dtype=np.float32)

            print(f"Writing final output for mu={mu}")

            fout['stokes'][i, :, :, :] = np.array([stokes_all[j] for j in range(n_models)])

            print(f"Done!")                       
        
        fout.close()