import numpy as np
import matplotlib.pyplot as pl

pl.close('all')

x = np.linspace(-7.5, 1.5, 82)

d = x[None, :] - x[:, None]
ll = [15.0, 10, 5.0, 3.0]
sigma = np.linspace(3000.0, 1000, 82)

n = len(ll)

fig, ax = pl.subplots(nrows=2, ncols=2, figsize=(10, 10))

for i in range(n):

    cov = sigma[None, :] * sigma[:, None] * np.exp(-d**2 / ll[i]**2)

    samples = np.random.multivariate_normal(np.zeros(82), cov, 5)

    ax.flat[i].plot(samples.T)