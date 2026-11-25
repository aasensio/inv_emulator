import numpy as np
import matplotlib.pyplot as pl

bx = 100.0 * np.random.randn(100)
by = 100.0 * np.random.randn(100)

b = np.hypot(bx, by)
chi = np.arctan2(by, bx)

chi2 = np.copy(chi)

chi2[chi2 < 0] += np.pi

bx_new = b * np.cos(chi2)
by_new = b * np.sin(chi2)

fig, ax = pl.subplots(nrows=1, ncols=2, figsize=(10, 5))

ax[0].plot(bx)
ax[0].plot(bx_new)

ax[1].plot(by)
ax[1].plot(by_new)