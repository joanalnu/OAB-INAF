import numpy as np
from scipy.integrate import quad

# Speed of light in km/s
c = 299792.458

def E_inv(a, Omega_m, Omega_Lambda, w0, wa, Omega_k, Omega_r):
    """Return 1/E(a) = H0 / H(a) for given scale factor a."""
    # CPL dark energy scaling
    w_a = w0 + wa * (1 - a)
    rho_de = Omega_Lambda * a**(-3 * (1 + w0 + wa)) * np.exp(3 * wa * (a - 1))
    
    # Dimensionless expansion rate
    E2 = Omega_r * a**(-4) + Omega_m * a**(-3) + Omega_k * a**(-2) + rho_de
    return 1.0 / np.sqrt(E2)


def comoving_distance(a, Omega_m, Omega_Lambda, w0, wa, Omega_k, Omega_r, H0):
    """Comoving distance χ(a) = c/H0 ∫ da' / (a'^2 E(a')) from a to 1."""
    integrand = lambda a_: E_inv(a_, Omega_m, Omega_Lambda, w0, wa, Omega_k, Omega_r) / (a_**2)
    chi, _ = quad(integrand, a, 1.0)
    return (c / H0) * chi


def luminosity_distance(input_array, Omega_m, Omega_Lambda, w0, wa, Omega_k, Omega_r, H0, input_type='z'):
    """
    Compute luminosity distance [Mpc] for an array of redshifts or scale factors.
    
    Parameters
    ----------
    input_array : array-like
        Array of redshifts (if input_type='z') or scale factors (if input_type='a').
    Omega_m, Omega_Lambda, w0, wa, Omega_k, Omega_r : floats
        Cosmological parameters.
    H0 : float
        Hubble constant [km/s/Mpc].
    input_type : {'z', 'a'}
        Type of input array.
    """
    input_array = np.asarray(input_array)
    if input_type == 'z':
        a_array = 1.0 / (1.0 + input_array)
    elif input_type == 'a':
        a_array = input_array
    else:
        raise ValueError("input_type must be 'z' or 'a'")
    
    dL = np.zeros_like(a_array)
    
    for i, a in enumerate(a_array):
        chi = comoving_distance(a, Omega_m, Omega_Lambda, w0, wa, Omega_k, Omega_r, H0)
        # Handle curvature
        sqrtOk = np.sqrt(np.abs(Omega_k))
        if Omega_k > 0:  # open universe
            D_M = (c / H0) / sqrtOk * np.sinh(sqrtOk * H0 * chi / c)
        elif Omega_k < 0:  # closed universe
            D_M = (c / H0) / sqrtOk * np.sin(sqrtOk * H0 * chi / c)
        else:  # flat universe
            D_M = chi
        dL[i] = (1.0 / a) * D_M  # d_L = (1/a) * D_M
    
    return dL


# Example usage:
if __name__ == "__main__":
    z = np.linspace(0, 2, 50)
    dL = luminosity_distance(
        z,
        Omega_m=0.3,
        Omega_Lambda=0.7,
        w0=-1.0,
        wa=0.2,
        Omega_k=0.0,
        Omega_r=0.0,
        H0=70.0, # where is H0 converted to SI (or cgs) units?
        input_type='z'
    )
    print("Luminosity distances [Mpc]:")
    print(dL)


    import matplotlib.pyplot as plt

    # Plot luminosity distance vs redshift
    plt.figure(figsize=(8, 5))
    plt.plot(z, dL, label=fr'$w_0={-1.0},\, w_a={0.2}$', color='C0', lw=2)

    plt.title("Luminosity Distance vs Redshift (Dynamical Dark Energy)", fontsize=14)
    plt.xlabel("Redshift  z", fontsize=12)
    plt.ylabel(r"Luminosity Distance  $d_L$ [Mpc]", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

