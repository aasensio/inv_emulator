import random
import numpy as np
import matplotlib.pyplot as pl
import glob
from scipy.interpolate import CubicSpline
from tqdm import tqdm

class GP(object):

    """
    A class to use Gaussian Processes (GP) to model semi-empirical solar models
    It has a database of 1D models and can generate new models by sampling from the GP. All models
    are reinterpolated to a common grid of 80 points in logtau between -7.5 and 1.5. The models that
    do not reach these values are extrapolated using cubic splines with clamped boundary conditions.

    Variables are allowed to vary in the following ranges (hardwired):
    - T: [2500, 150000]
    - vmic: [0, 5]
    - v: [-30, 30]
    - Bx: [-2500, 2500]
    - By: [-2500, 2500]
    - Bz: [-2500, 2500]
    
    The standard deviation of the GP is changed linearly between the logtau=-7.5 and logtau=1.5 with the following values (hardwired):
    - T: [3000, 1000]
    - vmic: [1.0, 1.0]
    - v: [3.0, 3.0]
    - Bx: [700, 700]
    - By: [700, 700]
    - Bz: [700, 700]
    
    Parameters:
        bc_type (str): Boundary condition type for cubic spline extrapolation ('natural'/'clamped'). Default is 'clamped'.
        
    """

    def __init__(self, n_depth=80, bc_type='clamped', root='.', min_sigma=None, max_sigma=None):

        self.files = glob.glob(f'{root}/*.1d')
        self.n_models = len(self.files)

        self.n_depth = n_depth

        # Reinterpolate models to a common grid and extrapolate in case it is needed
        self.xnew = np.linspace(-7.5, 1.5, n_depth)

        # logtau, T, vmic, v, Bx, By, Bz
        self.models = np.zeros((len(self.files), n_depth, 7))

        self.model_name = [None] * len(self.files)

        for i, file in enumerate(self.files):
            model = self.read_model(file)

            self.model_name[i] = file.split('/')[-1].split('.')[0]

            self.models[i, :, 0] = self.xnew

            # T
            natural = CubicSpline(model[:, 0], model[:, 1], bc_type=bc_type)
            self.add_boundary_knots(natural)
            self.models[i, :, 1] = natural(self.xnew)

            # vmic
            natural = CubicSpline(model[:, 0], model[:, 3], bc_type=bc_type)
            self.add_boundary_knots(natural)
            self.models[i, :, 2] = 0.0 * natural(self.xnew)

            # v
            natural = CubicSpline(model[:, 0], model[:, 4], bc_type=bc_type)
            self.add_boundary_knots(natural)
            self.models[i, :, 3] = natural(self.xnew)

            # Bx
            natural = CubicSpline(model[:, 0], model[:, 5], bc_type=bc_type)
            self.add_boundary_knots(natural)
            self.models[i, :, 4] = natural(self.xnew)

            # By
            natural = CubicSpline(model[:, 0], model[:, 6], bc_type=bc_type)
            self.add_boundary_knots(natural)
            self.models[i, :, 5] = natural(self.xnew)

            # Bz
            natural = CubicSpline(model[:, 0], model[:, 7], bc_type=bc_type)
            self.add_boundary_knots(natural)
            self.models[i, :, 6] = natural(self.xnew)

        self.n_models = len(self.models)
        
        self.vars = ['T', 'vmic', 'v', 'Bx', 'By', 'Bz']
        self.ranges = [[2500, 100000], [0, 5], [-30, 30], [-3500, 3500], [-3500, 3500], [-3500, 3500]]

        if min_sigma is not None:
            self.min_sigma = min_sigma
        else:
            self.min_sigma = [500, 2.0, 5.0, 1000, 1000, 1000]

        if max_sigma is not None:
            self.max_sigma = max_sigma
        else:
            self.max_sigma = [100, 2.0, 5.0, 1000, 1000, 1000]

        for i in range(6):
            print(f"{self.vars[i]}")
            print(f"   * Range - {self.ranges[i][0]} - {self.ranges[i][1]}")
            print(f"   * Sigma - [{self.min_sigma[i]}, {self.max_sigma[i]}]")

        self.sigma = [np.linspace(self.min_sigma[i], self.max_sigma[i], n_depth) for i in range(6)]
        
        self.d = self.xnew[None, :] - self.xnew[:, None]

        self.length_range = np.array([2.0, 25.0])
        self.length_range_log = np.log10(self.length_range)

        self.rng = np.random.default_rng()

    def gp_sample(self, mu, d, length, sigma, n_samples=1, ranges=None):
        """
        Generate a new model with a given correlation length and standard deviation.
        """
        cov = sigma[None, :] * sigma[:, None] * np.exp(-d**2 / length**2)
        cov += 1e-8 * np.eye(cov.shape[0]) * sigma**2
        
        samples = []

        iters = 0        

        for i in range(n_samples):
            ok = False
            while not ok:
                # sample = np.random.multivariate_normal(mu, cov)                
                try:
                    sample = self.rng.multivariate_normal(mu, cov, method='cholesky')
                    cholesky = True
                except:
                    sample = self.rng.multivariate_normal(mu, cov, method='eigh')
                    cholesky = False

                # Check if the sample is within the specified ranges
                ok = np.all((sample < ranges[1]) & (sample > ranges[0]))
                iters += 1
            
            samples.append(sample)
            
        samples = np.array(samples)
        efficiency = n_samples / iters

        return samples, efficiency


    def add_boundary_knots(self, spline):
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

    def read_model(self, filename):        
        tmp = np.loadtxt(filename, skiprows=4)
        return tmp[::-1, :]
    
    def sample_model(self, model=None, length=None, n_samples=1):
        """
        Samples new models from the Gaussian Process.

        Parameters:
            model (int, optional): Index of the model to sample from. If None, a random model is selected. Default is None.
            length (float, optional): Correlation length for the Gaussian Process. If None, a random length is selected from the log range. Default is None.
            n_samples (int): Number of samples to generate. Default is 1.

        Returns:
            tuple: A tuple containing:
                - out (numpy.ndarray): Array of sampled models with shape (n_samples, 7, n_depth): logtau, T, Pe, vmic, v, Bx, By, Bz
                - length (float): The correlation length used for sampling.
                - model (int): The index of the model used for sampling.
                - eff (float): The mean efficiency of the sampling process.

        Example:
            gp = GP(bc_type='natural')
            out, length, index, eff = gp.sample_model(n_samples=2)
        """

        if model is None:
            w = [random.gammavariate(0.01, 1.0) for _ in range(self.n_models)]
            w = np.array(w)
            w /= np.sum(w)
            # model = np.random.randint(0, len(self.models))

        if length is None:
            length = np.random.uniform(*self.length_range_log)
            length = 10.0**length
        
        out = np.zeros((n_samples, 7, self.n_depth))

        out[:, 0, :] = self.xnew
        
        eff = []
        
        for i in range(6):            
            mu = np.sum(w[:, None] * self.models[:, :, i+1], axis=0)

            sample, efficiency = self.gp_sample(mu, self.d, length, self.sigma[i], n_samples=n_samples, ranges=self.ranges[i])

            out[:, i+1, :] = sample

            eff.append(efficiency)

        eff = np.mean(eff)

        out = np.insert(out, 2, -np.ones(self.n_depth), axis=1)
        out[:, 2, :] = -1.0

        # Reduce the magnetic field strength by a random factor drawn from a beta distribution to 
        # ensure weak are common
        b_field_factor = np.random.beta(2.0, 3.0)
        out[:, 5:, :] *= b_field_factor

        # Reduce the velocity by a random factor drawn from a beta distribution to 
        # ensure low velocities are common
        v_factor = np.random.beta(2.0, 3.0)
        out[:, 3, :] *= v_factor

        # Reduce vmic by a random factor drawn from a beta distribution to 
        # ensure low velocities are common
        v_factor = np.random.beta(2.0, 1.0)
        out[:, 4, :] *= v_factor
                
        return out, length, model, eff

    def plot_model(self, model, index_model):

        fig, ax = pl.subplots(nrows=3, ncols=2, figsize=(10, 10))

        # logtau, T, Pe, vmic, v, Bx, By, Bz
        ind_plot = [1, 3, 4, 5, 6, 7]        

        n_models = model.shape[0]

        x = np.repeat(self.xnew[:, None], n_models, axis=1)

        for i in range(6):
            ax.flat[i].plot(x, model[:, ind_plot[i], :].T)
            ax.flat[i].plot(self.xnew, self.models[index_model, :, i+1], 'o')
            ax.flat[i].set_title(self.vars[i])

if __name__ == '__main__':
    gp = GP(bc_type='clamped')
    
    out, length, index, eff = gp.sample_model(n_samples=4)
    print(index, eff)

    pl.close('all')

    print(f'Efficiency: {eff:.2f} - Length: {length:.2f}')

    gp.plot_model(out, index)