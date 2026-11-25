import numpy as np
import matplotlib.pyplot as pl
import h5py
from scipy.interpolate import CubicSpline

def add_boundary_knots(spline):
    """
    Add knots infinitesimally to the left and right.

    Additional intervals are added to have zero 2nd and 3rd derivatives,
    and to maintain the first derivative from whatever boundary condition
    was selected. The spline is modified in place.

    https://docs.scipy.org/doc/scipy/tutorial/interpolate/extrapolation_examples.html

    """
    # determine the slope at the left edge
    leftx = spline.x[0]
    lefty = spline(leftx)
    leftslope = spline(leftx, nu=1)

    # add a new breakpoint just to the left and use the
    # known slope to construct the PPoly coefficients.
    leftxnext = np.nextafter(leftx, leftx - 1)
    leftynext = lefty + leftslope*(leftxnext - leftx)
    leftcoeffs = np.array([0, 0, leftslope, leftynext])
    spline.extend(leftcoeffs[..., None], np.r_[leftxnext])

    # repeat with additional knots to the right
    rightx = spline.x[-1]
    righty = spline(rightx)
    rightslope = spline(rightx,nu=1)
    rightxnext = np.nextafter(rightx, rightx + 1)
    rightynext = righty + rightslope * (rightxnext - rightx)
    rightcoeffs = np.array([0, 0, rightslope, rightynext])
    spline.extend(rightcoeffs[..., None], np.r_[rightxnext])

hsra = np.loadtxt('hsra.1d', skiprows=4)
hsra = hsra[::-1, :]  # Reverse the order of rows

xnew = np.linspace(-7.5, 1.5, 80)

natural = CubicSpline(hsra[:, 0], hsra[:, 1], bc_type='clamped')
add_boundary_knots(natural)

Tnew = natural(xnew)[::-1]

f = h5py.File('models_training.h5', 'r')
T = f['model'][:, :, 1]

d2 = np.sum((T - Tnew[None, :])**2, axis=-1)

print("Minimum distance:", np.min(d2))
print("Index of minimum distance:", np.argmin(d2))

fig, ax = pl.subplots()
ax.plot(xnew, Tnew, label='HSRA Model')
ax.plot(xnew, Tnew)
ax.plot(xnew, T[np.argmin(d2), :], label='Closest Model')