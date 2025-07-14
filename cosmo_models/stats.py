import numpy as np
from tqdm import tqdm
import itertools
import logging
import matplotlib.pyplot as plt
# from codecarbon import EmissionsTracker
# tracker = EmissionsTracker()
# logging.getLogger("codecarbon").setLevel(logging.CRITICAL)
#todo: add codecarbon (uncomment this)

class Models:
    def __init__(self, **kwargs):
        for name in dir(self):
            if not name.startswith('_'):
                attr = getattr(self, name)
                if callable(attr):
                    print(f'- {name}(): {attr.__doc__}')

    def free_parameters(self, modelname):
        """
        Retrieves the number of extra parameters skipping the first two (self and x).
        :param modelname: Name of the function.
        :return: Number for the free parameters of a model.
        """
        method = globals().get(modelname)
        if method:
            return method.__code__.co_varnames[2:]
        else:
            raise KeyError("Trying to use a model that doesn't exist.")

    def get_args(self, argnames, **kwargs):
        values = list()
        for name in argnames:
            if name in kwargs:
                values.append(kwargs[name])
            else:
                raise ValueError("The model name and parameters do not agree.")
        return values

    def linear_model(self, x, **kwargs):
        a, b = self.get_args(('a', 'b'), **kwargs)
        return a*x + b

    # def quadratic_model(self, x, **kwargs):
    #     a, b, c = self.get_args(('a', 'b', 'c'), **kwargs)
    #     return a*x**2 + b*x + c

    def exponential_model(self, x, **kwargs):
        a, b = self.get_args(('a', 'b'), **kwargs)
        return a*np.exp(b*x)

    def power_law_model(self, x, **kwargs):
        a, b = self.get_args(('a', 'b'), **kwargs)
        return a*x**b

models = Models()

class Stats:
    def __init__(self, **kwargs):
        self.x = kwargs.get('x')
        self.y = kwargs.get('y')
        self.model = kwargs.get('model')
        self.models = models

    def get_model(self, modelname, x, **kwargs):
        # method = globals().get(modelname)
        # if method:
        #     return method(x, **kwargs)
        # else:
        #     raise KeyError("Trying to use a model that doesn't exist.")
        #todo: implement this
        return models.linear_model(x, **kwargs)

    def measure_perpendicular_distances(self, x, y, model="linear_model", **kwargs):
        """
        Returns a scalar or an array with the perpendicular distance(s) between data point(s) and a fit line.
        :param x: x-axis values (Eiso/Egamma)
        :param y: y-axis values (Epeak)
        :param model: string containing the name of the fit function to be used.
        :param kwargs: parameters for the model
        :return: scalar or an array with the perpendicular distance(s) between data point(s) and a fit line.
        """
        raise ValueError("This method is not implemented yet.")

    def chi_squared(self, x, y, xerr, yerr, extra_scatter, model, **kwargs):
        """
        Computes the chi squared by summing the square of the residuals (observed - model). Does not account for errors neiter axis.
        :param x: x-axis values (Eiso/Egamma)
        :param y: y-axis values (Epeak)
        :param model: string containing the name of the fit function to be used.
        :param kwargs: parameters for the model
        :return: value of chi_squared (without errors)
        """
        model = self.get_model(model, x, **kwargs)
        residuals = (y - model)**2
        variance = 0.0
        if xerr is None and yerr is None:
            variance = 1.0
        else:
            if xerr is not None:
                print("X-axis errors were given, but not used (not implemented yet.")
            if yerr is not None:
                variance += yerr**2
            if extra_scatter is not None:
                print("Extra scatter was given, but not used (not implemented yet.")

        return np.sum(residuals/variance)

    def reduced_chi_squared(self, x, y, xerr, yerr, extra_scatter, model, **kwargs):
        chi_squared = self.chi_squared(x, y, xerr, yerr, extra_scatter, model, **kwargs)
        dof = len(x) - models.free_parameters(model)
        return chi_squared / dof

    def bestfit_2d(self, surface, p1_space, p2_space):
        min_chi = np.min(surface)
        p1_idx, p2_idx = np.unravel_index(np.argmin(surface), surface.shape)
        p1_fit, p2_fit = p2_space[p1_idx], p1_space[p2_idx]
        return p1_fit, p2_fit, min_chi

    def fitter(self, p1_space, p2_space, x, y, xerr=None, yerr=None, extra_scatter=None, model="linear_model"):
        G = np.zeros([len(p1_space), len(p2_space)])

        for i, j in tqdm(itertools.product(range(len(p1_space)), range(len(p2_space))), total=len(p1_space)*len(p2_space), desc=f"Fitting Correlation (model={model})"):
            kwargs = {'a': p1_space[i], 'b': p2_space[j]}
            G[i,j] = self.reduced_chi_squared(x, y, xerr, yerr, extra_scatter, model, **kwargs)

        p1_fit, p2_fit, min_chi_squared = self.bestfit_2d(G, p1_space, p2_space)
        return p1_fit, p2_fit, min_chi_squared, G

class plotting:
    def __init__(self, **kwargs):
        self.x = kwargs.get('Om')
        self.y = kwargs.get('Ode')

    def create_mask(self, surface, flat=True, no_big_bang=True):
        # create a proper mask for unphysical regions
        # instead of a simple mask, this proper mask also ensures that there are no np.infs or NaNs
        mask = np.zeros_like(surface, dtype=bool)
        for i in range(len(self.x)):
            for j in range(len(self.y)):
                if self.x[i]+self.y[j]>1.2 or self.x[i]+self.y[j]<0.8: # approx flat-universe region
                    mask[i,j] = flat
                if (#y[j]>x[i]+1 or
                    self.y[j]>=self.x[i]**(1/2.32) + 1.0): # approx no big bang area
                    mask[i,j] = no_big_bang
                if surface[i,j]==np.inf or surface[i,j]==np.nan: mask[i,j] = True

        return np.ma.masked_where(mask, surface)

    def add_constraint_lines(self, plot):
        plot.plot(self.x, 0.5*self.x, linestyle='--', color='black', alpha=0.7)
        plot.annotate('accelerating', (1.5,0.80), rotation=26.35)
        plot.annotate('decelerating', (1.5,0.67), rotation=26.35)

        plot.plot(self.x, self.x**(1/2.32)+1, c='r', alpha=0.7)
        plot.annotate('NO BIG BANG', (0.05,1.55), rotation=45, color='red')

        plot.plot(self.x, 1-self.x, linestyle='--', color='gray', alpha=0.7)
        plot.annotate('open', (0.75,0.1), rotation=-45)
        plot.annotate('closed', (0.8,0.15), rotation=-45)

        return None

    def create_contour(self, xf, yf, surface, levels=None, cmap="plasma"):
        masked_surface = self.create_mask(surface, flat=False, no_big_bang=False)
        if levels==None:
            minimum = np.min(surface)
            levels = [minimum, minimum+2.3, minimum+4.61, minimum+9.21]

        plt.figure(figsize=(8,6))
        plt.contourf(self.x, self.y, masked_surface.T, levels=levels, cmap=cmap)
        plt.colorbar(label=f'$\chi_r^2$ surface')
        plt.scatter(xf, yf, marker='x', c='r', s=100, linewidths=1, label=f'Best fit: {xf, yf}')
        plt.scatter(0.3, 0.7, marker='x', c='black', linewidths=1, label=f'Standard Cosmology: {xf, yf}')

        plt.xlabel(r'$\Omega_m$'); plt.ylabel(r'$\Omega_{DE}$'); plt.legend(loc='upper right')
        plt.xlim(self.x[0], self.x[-1]); plt.ylim(self.y[0], self.y[-1]); plt.tight_layout()

        self.add_constraint_lines(plt.gca())

        return plt.gca()


