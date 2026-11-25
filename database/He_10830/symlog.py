import numpy as np

def symlog(x):
    return np.sign(x) * np.log1p(np.abs(x))

