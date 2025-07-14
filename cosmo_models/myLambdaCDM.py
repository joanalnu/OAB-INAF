import numpy as np
from scipy.integrate import cumulative_trapezoid


class LambdaCDMClass:
    """
    Lambda-CDM cosmological model with default parameter access.
    """

    def __init__(self, parameters):
        # Extract parameters from the CosmologicalParametersClass instance
        # and set them as attributes of this class
        for key, value in parameters.__dict__.items():
            setattr(self, key, value)

        print("LCDM Initiated")

    def hubble_param(self, z):
        """
        Hubble Parameter from Lambda-CDM model
        H(z) = H0 * sqrt(Omega_m0*(1+z)^3 + Omega_r0*(1+z)^4 + Omega_Lambda0 + Omega_K0*(1+z)^2)
        :param z: redshift scalar or array
        :return: values for the hubble parameter in dependence of the parameters and redshift.
        """
        E_squared = (self.Omega_m0 * (1 + z) ** 3 +
                     self.Omega_r0 * (1 + z) ** 4 +
                     self.Omega_Lambda0 +
                     self.Omega_K0 * (1 + z) ** 2)
        return self.H0 * np.sqrt(E_squared)

    def integrand_scalar(self, z):
        z_array = np.linspace(0, z, num=1000)
        y_values = 1.0 / self.hubble_param(z_array)
        integral = cumulative_trapezoid(y_values, z_array, initial=0)[-1]
        return integral

    def comoving_distance(self, z):
        return (self.c/self.H0) * np.array([self.integrand_scalar(x_val) for x_val in z])

    def proper_distance(self, z):
        com_dist = self.comoving_distance(z)  # in km
        chi = (self.H0 / self.c) * com_dist  # dimensionless
        sqrt_ok = np.sqrt(np.abs(self.Omega_K0))

        if self.Omega_K0 < 0:
            return (self.c / self.H0) * np.sin(sqrt_ok * chi) / sqrt_ok
        elif self.Omega_K0 > 0:
            return (self.c / self.H0) * np.sinh(sqrt_ok * chi) / sqrt_ok
        else:
            return com_dist  # flat universe

    def angular_distance(self, z):
        return (1 / (1 + z)) * self.proper_distance(z)

    def luminosity_distance(self, z):
        return (1 + z) * self.proper_distance(z)