import numpy as np
import matplotlib.pyplot as pl
import h5py
from symlog import symlog

if __name__ == "__main__":

    pl.close('all')

    tmp = h5py.File("emissivity_absorption_validation.h5", "r")

    which = 'epsilon'
    which = 'eta'
    which = 'rho'

    #############
    # Epsilon
    #############
    if which == 'epsilon':
        max_value = [3.6e-5, 2.4e-6, 2.4e-6, 1e-5]
        eps = tmp['eps'][:]

        locations = [50, 100, 150, 200, 250]
        factors = [5e2, 1e5, 1e5, 1e5]

        fig, ax = pl.subplots(nrows=2, ncols=2, figsize=(10, 10))

        loop = 0
        for i in range(2):
            for j in range(2):
                for k in locations:
                    ax[i, j].hist(eps[:, loop, k] / max_value[loop], bins=50, alpha=0.5, label=f'Location {k}')

                loop += 1

        fig, ax = pl.subplots(nrows=2, ncols=2, figsize=(10, 10))

        loop = 0
        for i in range(2):
            for j in range(2):
                for k in locations:
                    x = eps[:, loop, k] / max_value[loop]
                    ax[i, j].hist(symlog(factors[loop]*x), bins=50, alpha=0.5, label=f'Location {k}')

                loop += 1

    #############
    # Eta
    #############
    if which == 'eta':
        max_value = [1, 0.1, 0.1, 1.0]
        eta = tmp['eta'][:]

        locations = [50, 100, 150, 200, 250]
        factors = [5e1, 5e6, 5e6, 1e5]

        fig, ax = pl.subplots(nrows=2, ncols=2, figsize=(10, 10))

        loop = 0
        for i in range(2):
            for j in range(2):
                for k in locations:
                    ax[i, j].hist(eta[:, loop, k] / max_value[loop], bins=50, alpha=0.5, label=f'Location {k}')

                loop += 1

        fig, ax = pl.subplots(nrows=2, ncols=2, figsize=(10, 10))

        loop = 0
        for i in range(2):
            for j in range(2):
                for k in locations:
                    x = eta[:, loop, k] / max_value[loop]
                    ax[i, j].hist(symlog(factors[loop]*x), bins=50, alpha=0.5, label=f'Location {k}')

                loop += 1

    #############
    # Rho
    #############
    if which == 'rho':
        max_value = [0.1, 0.1,1.0]
        rho = tmp['rho'][:]

        locations = [50, 100, 150, 200, 250]
        factors = [5e4, 5e4, 1e3]

        fig, ax = pl.subplots(nrows=1, ncols=3, figsize=(15, 5))

        for loop in range(3):
            for k in locations:
                ax[loop].hist(rho[:, loop, k] / max_value[loop], bins=50, alpha=0.5, label=f'Location {k}')

        fig, ax = pl.subplots(nrows=1, ncols=3, figsize=(15, 5))
        
        for loop in range(3):
            for k in locations:
                x = rho[:, loop, k] / max_value[loop]
                ax[loop].hist(symlog(factors[loop]*x), bins=50, alpha=0.5, label=f'Location {k}')
                