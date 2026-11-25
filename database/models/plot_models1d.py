import numpy as np
import matplotlib.pyplot as pl
from sample import GP
import random

pl.close('all')

gp = GP(bc_type='clamped', root='models1d')

fig, ax = pl.subplots(nrows=5, ncols=5, figsize=(20, 20))
for i in range(25):
    ax.flat[i].plot(gp.models[i, :, 0], gp.models[i, :, 1])
    ax.flat[i].set_title(gp.model_name[i])

ww = np.zeros((len(gp.models), 25, 25))
fig, ax = pl.subplots(nrows=5, ncols=5, figsize=(20, 20))
for i in range(25):    
    for j in range(25):
        w = [random.gammavariate(0.1, 1.0) for _ in range(len(gp.models))]
        w = np.array(w)
        w /= np.sum(w)
        ww[:, j, i] = w
        ax.flat[i].plot(gp.models[0, :, 0], np.sum(w[:, None] * gp.models[:, :, 1], axis=0))
    ax.flat[i].set_ylim([0,20000])


ww = ww.reshape((len(gp.models), 25 * 25))
fig, ax = pl.subplots()
ax.plot(ww[:, 0:10])