import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
from astropy.cosmology import w0waCDM
import astropy.units as u

# ------------------ Synthetic Data ------------------
np.random.seed(42)
N = 200  # More data points
z = np.random.uniform(0.1, 5.0, N)   # Wider redshift range

# True cosmology (fiducial)
Omega_m_true, w0_true, wa_true, H0_true, Ok_true = 0.3, -1.0, 0.0, 70.0, 0.0

# Proper cosmological distance calculator
def d_L(z, Omega_m, w0, wa, H0, Ok):
    """Proper luminosity distance calculation using astropy"""
    cosmo = w0waCDM(H0=H0, Om0=Omega_m, Ode0=1 - Omega_m - Ok, w0=w0, wa=wa)
    return cosmo.luminosity_distance(z).value  # Returns in Mpc

# True Amati relation
a_true, b_true, sigma_int_true = 0.45, 1.8, 0.15

# Calculate true Eiso and y values
Eiso_true = 1e52 * (d_L(z, Omega_m_true, w0_true, wa_true, H0_true, Ok_true)/1e28)**2
x_true = np.log10(Eiso_true)
y_true = a_true * x_true + b_true

# Add scatter and measurement error
sigma_meas = 0.05  # Better measurement precision
y_obs = y_true + np.random.normal(0, np.sqrt(sigma_meas**2 + sigma_int_true**2), N)

# ------------------ Likelihood ------------------
def log_likelihood(theta, x, y, yerr, z):
    a, b, sigma_int, Omega_m, w0, wa, H0, Ok = theta
    if sigma_int <= 0 or H0 <= 0 or Omega_m < 0 or Omega_m > 1:
        return -np.inf
    
    # Cosmology enters via Eiso -> x
    Eiso_model = 1e52 * (d_L(z, Omega_m, w0, wa, H0, Ok)/1e28)**2
    x_model = np.log10(Eiso_model)

    y_model = a * x_model + b
    var = yerr**2 + sigma_int**2
    return -0.5 * np.sum((y - y_model)**2/var + np.log(2*np.pi*var))

# Priors - much tighter based on external constraints
def log_prior(theta):
    a, b, sigma_int, Omega_m, w0, wa, H0, Ok = theta
    if (0.2 < a < 0.7 and 1.5 < b < 2.5 and 0.05 < sigma_int < 0.3 and
        0.2 < Omega_m < 0.4 and -1.5 < w0 < -0.5 and -1.0 < wa < 1.0 and
        65 < H0 < 75 and -0.1 < Ok < 0.1):  # Much tighter ranges!
        return 0.0
    return -np.inf

def log_posterior(theta, x, y, yerr, z):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr, z)

# ------------------ Run MCMC ------------------
ndim = 8
nwalkers = 32
initial = np.array([a_true, b_true, sigma_int_true, Omega_m_true, 
                   w0_true, wa_true, H0_true, Ok_true])  # Better initial guesses
pos = initial + 1e-3 * np.random.randn(nwalkers, ndim)  # Tighter spread

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(x_true, y_obs, sigma_meas*np.ones(N), z))
sampler.run_mcmc(pos, 2000, progress=True)  # Longer chain

# ------------------ Results ------------------
flat_samples = sampler.get_chain(discard=2000, thin=20, flat=True)  # More burn-in
labels = ["a", "b", "sigma_int", r"$\Omega_m$", r"$w_0$", r"$w_a$", r"$H_0$", r"$\Omega_k$"]

# Diagnostic: check parameter ranges and uncertainties
print("Parameter constraints:")
for i, label in enumerate(labels):
    samples = flat_samples[:, i]
    q = np.percentile(samples, [16, 50, 84])
    print(f"{label}: {q[1]:.3f} +{q[2]-q[1]:.3f} -{q[1]-q[0]:.3f}")

# Create corner plot with customized ranges
ranges = [(0.3, 0.6), (1.6, 2.0), (0.1, 0.2),        # a, b, sigma_int
          (0.25, 0.35), (-1.2, -0.8), (-0.5, 0.5),   # Ω_m, w0, wa
          (65, 75), (-0.05, 0.05)]                   # H0, Ω_k

fig = corner.corner(
    flat_samples, 
    labels=labels, 
    truths=[a_true, b_true, sigma_int_true, Omega_m_true, w0_true, wa_true, H0_true, Ok_true],
    smooth=1.0,
    bins=30,
    plot_density=True,
    fill_contours=True,
    range=ranges,  # Focus on constrained regions
    quantiles=[0.16, 0.5, 0.84],  # Show quantiles
    contour_kwargs=dict(colors=None),
    contourf_kwargs=dict(alpha=0.7),
    levels=[0.39, 0.86, 0.99]
)

plt.savefig("mcmc_improved_cosmology.png", dpi=300, bbox_inches='tight')
plt.show()

# ------------------ Amati-only corner plot ------------------
# Extract only the first three parameters (a, b, sigma_int)
selected_samples = flat_samples[:, :3]
selected_labels = labels[:3]
selected_truths = [a_true, b_true, sigma_int_true]

# Create corner plot for only a, b, sigma_int
fig = corner.corner(
    selected_samples, 
    labels=selected_labels, 
    truths=selected_truths,
    smooth=1.0,
    smooth1d=1.0,
    bins=30,
    plot_contours=True,
    plot_density=True,
    fill_contours=True,
    contour_kwargs=dict(colors=None),
    contourf_kwargs=dict(alpha=0.7),
    levels=[0.39, 0.86, 0.99]
)

plt.savefig("mcmc_amati_params_only.png", dpi=300, bbox_inches='tight')
plt.show()