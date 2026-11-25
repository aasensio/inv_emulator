import numpy as np
import h5py
from tqdm import tqdm

def hei_model():
    """
    Generate a random He I model.

    """

    # Sample the He I model parameters
    height_range = [0.1, 10.0]
    Bx_scale = 500.0
    By_scale = 500.0
    Bz_scale = 500.0
    dv_range = [4.0, 18.0]
    a_range = [0.0, 1.5]
    theta_range = [0.0, 89.9]

    h = np.random.uniform(height_range[0], height_range[1], size=1)
    Bx = np.random.normal(loc=0, scale=Bx_scale, size=1)
    By = np.random.normal(loc=0, scale=By_scale, size=1)
    Bz = np.random.normal(loc=0, scale=Bz_scale, size=1)
    dv = np.random.uniform(dv_range[0], dv_range[1], size=1)
    a = np.random.uniform(a_range[0], a_range[1], size=1)
    theta = np.random.uniform(theta_range[0], theta_range[1], size=1)

    model_he = np.concatenate([Bx[:, None], By[:, None], Bz[:, None], dv[:, None], a[:, None]], axis=1)

    geom_he = np.concatenate((h[:, None], theta[:, None]), axis=1)
    
    return model_he, geom_he


def generate(n_models, outfile):
    
    model_he = []
    geom_he = []
    
    for i in tqdm(range(n_models)):
    
        # First component He I model
        hei_model_data, geom_he_data = hei_model()
        model_he.append(hei_model_data)
        geom_he.append(geom_he_data)

                        
    # First component He I model
    # Bx, By, Bz, tau, v, dv, a
    model_he = np.concatenate(model_he, axis=0)
    geom_he = np.concatenate(geom_he, axis=0)

    
    f = h5py.File(outfile, 'w')    
    db_model_he = f.create_dataset('model_he', model_he.shape, dtype=np.float32)
    db_geom_he = f.create_dataset('geom_he', geom_he.shape, dtype=np.float32)    
    db_model_he[:] = model_he.astype('float32')
    db_geom_he[:] = geom_he.astype('float32')    
    f.close()

    print("Generated models saved to", outfile)

if __name__ == "__main__":

    tmp = ['validation', 'testing', 'training']
    
    for which in tmp:

        if which == 'training':
            n_models = 1000000
    
            print("Total number of training models: ", n_models)
    
            generate(n_models, 'models_training.h5')
    
        if which == 'validation':
    
            n_models = 10000
    
            print("Total number of validation models: ", n_models)
    
            generate(n_models, 'models_validation.h5')
    
        if which == 'testing':    
            n_models = 50000
    
            print("Total number of validation models: ", n_models)
    
            generate(n_models, 'models_testing.h5')