import numpy as np

class CosmologicalParametersClass:
    """
    Base class to store parameters for the cosmological models in this directory.
    """

    def __init__(self, **kwargs):
        # standard cosmological parameters
        self.c = 2.99792458e10  # cm/s
        self.H0 = 70.0 * 1e5 / 3.086e24  # s^-1 (fixed multiplication)
        self.Omega_m0 = 0.3  # Matter density parameter today
        self.Omega_r0 = 0.001  # Radiation density parameter today
        self.Omega_Lambda0 = 1 - self.Omega_m0 - self.Omega_r0  # Dark energy (flat universe)
        self.Omega_K0 = 0.0  # assuming flat Universe

        # Dark energy equation of state parameters (following DESI 2025)
        self.w0 = -0.7  # Present value of w
        self.wa = -1.0  # Evolution parameter for w(z) = w0 + wa * z/(1+z)

        # RS-II brane model parameters
        self.rho_c = 3 * (self.H0 * 3.086e19) ** 2 / (8 * np.pi * 6.674e-11)  # Critical density in SI
        self.lambda_brane = 1e-4  # Brane tension parameter (adjustable)
        self.lambda_5 = 0.0

        # Update if additional arguments are provided
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                # Recalculate dependent parameters
                if key in ['Omega_m0', 'Omega_r0']:
                    self.Omega_Lambda0 = 1 - self.Omega_m0 - self.Omega_r0
                elif key == 'H0':
                    self.rho_c = 3 * (self.H0 * 3.086e19) ** 2 / (8 * np.pi * 6.674e-11)

    def update_parameters(self, **kwargs):
        """Convenient method to update multiple parameters at once"""
        self.__init__(**{**self.__dict__, **kwargs})

    def get_all_parameters(self):
        """Return dictionary of all parameters"""
        return {key: value for key, value in self.__dict__.items()
                if not key.startswith('_')}

CosmologicalParameters = CosmologicalParametersClass