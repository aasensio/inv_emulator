import numpy as np
import h5py
from sample import GP
from tqdm import tqdm
import concurrent.futures
import os
import random

    
gp = GP(bc_type='clamped', root='models1d')

which = 'prueba'

def init_worker():
    """Re-seeds random generators using OS entropy per worker process."""
    # Seed Python's standard 'random' module
    random.seed()

    # Seed NumPy (creates a fresh seed from OS entropy for every process)
    np.random.seed(int.from_bytes(os.urandom(4), byteorder="little"))

# Define a top-level helper function so multiprocessing can easily pickle it
def worker_task(_):
    # Returns only the first element (model) from the tuple returned by sample_model
    model, length, index, eff = gp.sample_model(n_samples=1)
    return model


if which == 'training':
    n_models = 1000000

    print("Total number of training models: ", n_models)

    models = []
    with concurrent.futures.ProcessPoolExecutor(initializer=init_worker) as executor:
        # Submit all tasks to the pool
        futures = [executor.submit(worker_task, i) for i in range(n_models)]

        # as_completed yields futures as soon as they finish, giving a smooth progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=n_models):
            models.append(future.result())

    models = np.concatenate(models, axis=0)

    # logtau, T, Pe, vmic, v, Bx, By, Bz
    models = np.transpose(models[:, :, ::-1], (0, 2, 1))

    ff = np.ones((n_models,), dtype=np.float64)
    vmac = np.zeros((n_models,), dtype=np.float64)

    f = h5py.File('models_training.h5', 'w')
    db_model = f.create_dataset('model', models.shape, dtype=np.float32)
    db_ff = f.create_dataset('ff', ff.shape, dtype=np.float32)
    db_vmac = f.create_dataset('vmac', vmac.shape, dtype=np.float32)
    db_model[:] = models.astype('float32')
    db_ff[:] = ff.astype('float32')
    db_vmac[:] = vmac.astype('float32')
    f.close()

if which == 'validation':

    n_models = 10000

    print("Total number of validation models: ", n_models)

    models = []
    with concurrent.futures.ProcessPoolExecutor(initializer=init_worker) as executor:
        # Submit all tasks to the pool
        futures = [executor.submit(worker_task, i) for i in range(n_models)]

        # as_completed yields futures as soon as they finish, giving a smooth progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=n_models):
            models.append(future.result())

    models = np.concatenate(models, axis=0)

    # logtau, T, Pe, vmic, v, Bx, By, Bz
    models = np.transpose(models[:, :, ::-1], (0, 2, 1))

    ff = np.ones((n_models,), dtype=np.float64)
    vmac = np.zeros((n_models,), dtype=np.float64)

    f = h5py.File('models_validation.h5', 'w')
    db_model = f.create_dataset('model', models.shape, dtype=np.float32)
    db_ff = f.create_dataset('ff', ff.shape, dtype=np.float32)
    db_vmac = f.create_dataset('vmac', vmac.shape, dtype=np.float32)
    db_model[:] = models.astype('float32')
    db_ff[:] = ff.astype('float32')
    db_vmac[:] = vmac.astype('float32')
    f.close()

if which == 'testing':    
    n_models = 10000

    print("Total number of testing models: ", n_models)

    models = []
    # ProcessPoolExecutor leverages multiple CPU cores
    with concurrent.futures.ProcessPoolExecutor(initializer=init_worker) as executor:
        # Submit all tasks to the pool
        futures = [executor.submit(worker_task, i) for i in range(n_models)]

        # as_completed yields futures as soon as they finish, giving a smooth progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=n_models):
            models.append(future.result())

    models = np.concatenate(models, axis=0)

    # logtau, T, Pe, vmic, v, Bx, By, Bz
    models = np.transpose(models[:, :, ::-1], (0, 2, 1))

    ff = np.ones((n_models,), dtype=np.float64)
    vmac = np.zeros((n_models,), dtype=np.float64)

    f = h5py.File('models_testing.h5', 'w')
    db_model = f.create_dataset('model', models.shape, dtype=np.float32)
    db_ff = f.create_dataset('ff', ff.shape, dtype=np.float32)
    db_vmac = f.create_dataset('vmac', vmac.shape, dtype=np.float32)
    db_model[:] = models.astype('float32')
    db_ff[:] = ff.astype('float32')
    db_vmac[:] = vmac.astype('float32')
    f.close()

if which == 'prueba':    
    n_models = 1000

    print("Total number of testing models: ", n_models)

    models = []
    # ProcessPoolExecutor leverages multiple CPU cores
    with concurrent.futures.ProcessPoolExecutor(initializer=init_worker) as executor:
        # Submit all tasks to the pool
        futures = [executor.submit(worker_task, i) for i in range(n_models)]

        # as_completed yields futures as soon as they finish, giving a smooth progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=n_models):
            models.append(future.result())

    models = np.concatenate(models, axis=0)

    # logtau, T, Pe, vmic, v, Bx, By, Bz
    models = np.transpose(models[:, :, ::-1], (0, 2, 1))

    ff = np.ones((n_models,), dtype=np.float64)
    vmac = np.zeros((n_models,), dtype=np.float64)

    f = h5py.File('models_prueba.h5', 'w')
    db_model = f.create_dataset('model', models.shape, dtype=np.float32)
    db_ff = f.create_dataset('ff', ff.shape, dtype=np.float32)
    db_vmac = f.create_dataset('vmac', vmac.shape, dtype=np.float32)
    db_model[:] = models.astype('float32')
    db_ff[:] = ff.astype('float32')
    db_vmac[:] = vmac.astype('float32')
    f.close()
