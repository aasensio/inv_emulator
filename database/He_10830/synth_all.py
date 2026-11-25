import numpy as np
import h5py
from tqdm import tqdm
import hazel
import sys
import multiprocessing as mp
    
def do_synthesis(models):

    model, geom = models
    
    Bx, By, Bz, dv, a = model
    h, theta = geom
        
    # One component    
    mod.spectrum['spec1'].set_los(np.array([theta, 0.0, 90.0]))
    mod.spectrum['spec1'].set_boundary(np.array([1.0, 0.0, 0.0, 0.0]))
    mod.atmospheres['ch1'].set_parameters([Bx, By, Bz, 1.0, 0.0, dv, 1.0, a, 1.0], 1.0)
    mod.atmospheres['ch1'].height = h
    
    mod.synthesize()

    wl = mod.spectrum['spec1'].wavelength_axis
    eps = mod.atmospheres['ch1'].eps
    eta = mod.atmospheres['ch1'].eta
    rho = mod.atmospheres['ch1'].rho

    return wl, eps, eta, rho

if __name__ == '__main__':
    tmp = ['validation', 'testing', 'training'] # 'validation
    
    n_lambda = 256
    n_cpus = 28

    for which in tmp:
    
        # Read the He I model parameters
        f = h5py.File(f'models_{which}.h5', 'r')    
        model_he = f['model_he'][:]    
        geom_he = f['geom_he'][:]    
        f.close()
        
        n_models = model_he.shape[0]
        
        model_he = [model_he[i, :].astype('float64') for i in range(n_models)]
        geom_he = [geom_he[i, :].astype('float64') for i in range(n_models)]    

        mod = hazel.Model(working_mode='synthesis')
        mod.add_spectral({'Name': 'spec1', 'Wavelength': [10823, 10834, n_lambda], 'topology': 'ch1', 'LOS': [0.0, 0.0, 90.0], 'Boundary condition': [1.0,0.0,0.0,0.0]}, index=0)
        mod.add_chromosphere({'Name': 'ch1', 'Spectral region': 'spec1', 'Height': 3.0, 'Line': '10830', 'Wavelength': [10823, 10834], 'Reference frame': 'vertical'})    
        mod.setup()

        mset_eps = np.zeros((n_models, 4, n_lambda), dtype=np.float32)
        mset_eta = np.zeros((n_models, 4, n_lambda), dtype=np.float32)
        mset_rho = np.zeros((n_models, 3, n_lambda), dtype=np.float32)

        mset_wvl = np.zeros(n_lambda, dtype=np.float32)

        with mp.Pool(processes=n_cpus) as pool:

            for index, result in tqdm(enumerate(pool.imap(do_synthesis, zip(model_he, geom_he))), total=n_models):

                wvl, eps, eta, rho = result            
                mset_eps[index:(index+1), :, :] = eps.astype('float32')
                mset_eta[index:(index+1), :, :] = eta.astype('float32')
                mset_rho[index:(index+1), :, :] = rho.astype('float32')
                if (index == 0):
                    mset_wvl[:] = wvl


        f = h5py.File(f'emissivity_absorption_{which}.h5', 'w')
        db_eps = f.create_dataset('eps', mset_eps.shape, dtype=np.float32)
        db_eta = f.create_dataset('eta', mset_eta.shape, dtype=np.float32)
        db_rho = f.create_dataset('rho', mset_rho.shape, dtype=np.float32)
        db_wvl = f.create_dataset('wvl', mset_wvl.shape, dtype=np.float32)
        db_eps[:] = mset_eps
        db_eta[:] = mset_eta
        db_rho[:] = mset_rho
        db_wvl[:] = mset_wvl
        f.close()