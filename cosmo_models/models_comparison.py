# original code from github.com/capibara3/cosmology/blob/main/brane-theory-sims
# this code is used in parameters.py and in myLambdaCDM.py






import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid


# ===========================
# 1. Parameter definition
# ===========================
class CosmologicalParameters:
    """
    Base class to store cosmological parameters with easy access methods.
    """

    def __init__(self, **kwargs):
        # Standard cosmological parameters
        self.H0 = 70.0*1e5/3.086e24 #s^-1 in cgs
        self.Omega_m0 = 0.3  # Matter density parameter today
        self.Omega_r0 = 0.0 # neglecting radiation density
        self.Omega_k0 = 0.0 # assuming flat universe k=0
        self.Omega_Lambda0 = 0.7

        # Dark energy equation of state parameters (following DESI 2025)
        self.w0 = -0.7  # Present value of w
        self.wa = -1.0  # Evolution parameter for w(z) = w0 + wa * z/(1+z)

        # Update with any provided parameters
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

# ===========================
# 2. Model definition
# ===========================
class LambdaCDM(CosmologicalParameters):
    """
    Lambda-CDM cosmological model with inherited parameter access.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def H(self, z):
        """
        Hubble parameter for Lambda-CDM model
        H(z) = H0 * sqrt(Omega_m0*(1+z)^3 + Omega_r0*(1+z)^4 + Omega_Lambda0)
        """
        E_squared = (self.Omega_m0 * (1 + z) ** 3 +
                     self.Omega_r0 * (1 + z) ** 4 +
                     self.Omega_Lambda0)
        return self.H0 * np.sqrt(E_squared)

class wowaCDM(CosmologicalParameters):
    """
    w0wa-CDM cosmological model with inherited parameter access.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def H(self, z):
        """
        Hubble parameter for w0wa-CDM model
        Dark energy density: rho_DE(z) = rho_DE0 * (1+z)^(3*(1+w0+wa)) * exp(-3*wa*z/(1+z))
        """

        def integrand(zp):
            return (1 + self.w0 + self.wa * zp / (1 + zp)) / (1 + zp)

        if isinstance(z, np.ndarray):
            integral_values = []
            for z_val in z:
                if z_val == 0:
                    integral_val = 0
                else:
                    n_points = max(100, int(z_val * 50))
                    z_points = np.linspace(0, z_val, n_points)
                    y_values = integrand(z_points)
                    integral_result = cumulative_trapezoid(y_values, z_points, initial=0)
                    integral_val = integral_result[-1]
                integral_values.append(integral_val)
            integral_values = np.array(integral_values)
        else:
            if z == 0:
                integral_values = 0
            else:
                n_points = max(100, int(z * 50))
                z_points = np.linspace(0, z, n_points)
                y_values = integrand(z_points)
                integral_result = cumulative_trapezoid(y_values, z_points, initial=0)
                integral_values = integral_result[-1]

        rho_DE_evolution = np.exp(3 * integral_values)
        E_squared = (self.Omega_m0 * (1 + z) ** 3 +
                     self.Omega_r0 * (1 + z) ** 4 +
                     self.Omega_Lambda0 * rho_DE_evolution)
        return self.H0 * np.sqrt(E_squared)




if __name__ == "__main__":
    # initialize models
    params = CosmologicalParameters
    LCDM = LambdaCDM()
    wwCDM = wowaCDM()

    z = np.linspace(.0, 8.0, 100)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(z, LCDM.H(z), 'b-', linewidth=2, label=r'$\Lambda\text{CDM}$')
    ax.plot(z, wwCDM.H(z), 'r--', linewidth=2, label=r'$w_0w_a\text{CDM}$')
    plt.yscale('log')

    ax.set_xlabel('Redshift z')
    ax.set_ylabel('H(z) [km/s/Mpc]')
    ax.set_title('Hubble Parameter Evolution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ADD INSET HERE
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # Create inset
    axins = inset_axes(ax, width="35%", height="35%", loc='lower right')

    # Plot zoomed region (0 < z < 1)
    z_inset = np.linspace(0.0, 1.0, 50)
    axins.plot(z_inset, LCDM.H(z_inset), 'b-', linewidth=2)
    axins.plot(z_inset, wwCDM.H(z_inset), 'r--', linewidth=2)

    # Set inset limits
    axins.set_xlim(0, 1)
    # Automatically adjust y-limits or set manually
    axins.set_ylim(min(LCDM.H(z_inset)), max(LCDM.H(z_inset)))

    # Style the inset
    axins.grid(True, alpha=0.3)
    axins.set_xlabel('z', fontsize=9)
    axins.set_ylabel('H(z)', fontsize=9)
    axins.tick_params(labelsize=8)

    # Optional: highlight the zoomed region on main plot
    ax.indicate_inset_zoom(axins, edgecolor="gray", alpha=0.7, linewidth=1)

    plt.show()