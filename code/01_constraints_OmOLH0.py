import numpy as np
import pandas as pd
from astropy.cosmology import LambdaCDM
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tqdm import tqdm

# 1. Data Loading and Preparation
df = pd.read_csv("../data/table.csv")
z = df['z'].values

Epeak = np.log10(df['Epeak']).values
Epeak_err = df['Epeak_err'].values / (df['Epeak'].values * np.log(10))
Epeak_bc = Epeak - np.mean(Epeak)

original_Eiso = np.log10(df['Eiso']).values
original_Eiso_bc = original_Eiso - np.mean(original_Eiso)

# 2. Grid Setup (Reduced slightly for speed, increase to 25 if time permits)
N = 15
Om_grid = np.linspace(0.0, 1.5, N)
Ode_grid = np.linspace(0.0, 1.5, N)
H0_grid = np.linspace(60.0, 80.0, N)

# Parameters
factor = 1.30
extra_err = np.log10(factor)
standard_cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
standard_dl = standard_cosmo.luminosity_distance(z).value

# Storage
chi_volume = np.zeros((N, N, N))
best_params_grid = np.zeros((N, N, N, 2)) # stores [a, b]

def linear_model(x, a, b):
    return a * x + b

# 3. Computation Loop
for i in tqdm(range(N), desc="Scanning LCDM Volume"):
    for j in range(N):
        for k in range(N):
            cosmo = LambdaCDM(H0=H0_grid[k], Om0=Om_grid[i], Ode0=Ode_grid[j])
            dl = cosmo.luminosity_distance(z).value
            
            # Rescale Eiso based on new cosmology
            Eiso = 2 * np.log10(dl / standard_dl) + original_Eiso
            Eiso_bc = Eiso - np.mean(Eiso)
            
            # CRITICAL: Sum errors in quadrature
            total_err = np.sqrt(Epeak_err**2 + extra_err**2)
            
            try:
                popt, _ = curve_fit(linear_model, Eiso_bc, Epeak_bc, sigma=total_err, p0=[0.5, 0.0])
                residuals = Epeak_bc - linear_model(Eiso_bc, *popt)
                chi_volume[i, j, k] = np.sum((residuals / total_err)**2)
                best_params_grid[i, j, k] = popt
                dof = len(Eiso) - 2 # not longer used
            except:
                chi_volume[i, j, k] = 1e6

# 4. Results and Flatness Analy
min_chi = np.nanmin(chi_volume)
min_idx = np.unravel_index(np.argmin(chi_volume), chi_volume.shape)
best_Om, best_Ode, best_H0 = Om_grid[min_idx[0]], Ode_grid[min_idx[1]], H0_grid[min_idx[2]]

print(f"\n--- LCDM Best Fit ---")
print(f"Om: {best_Om:.3f}, Ode: {best_Ode:.3f}, H0: {best_H0:.2f}")
print(f"min chi: {min_chi:.3f}")

# Flatness Metric: Variance of Chi2 across each dimension
h0_slice_variance = np.var(chi_volume[min_idx[0], min_idx[1], :])
Om_slice_variance = np.var(chi_volume[:, min_idx[1], min_idx[2]])
Ode_slice_variance = np.var(chi_volume[min_idx[0], :, min_idx[2]])
print(f"H0 Surface Flatness (Variance): {h0_slice_variance:.2e}")
print(f"Om Surface Flatness (Variance): {Om_slice_variance:.2e}")
print(f"Ode Surface Flatness (Variance): {Ode_slice_variance:.2e}")

# 5. Plotting for Paper
# Figure A: Amati Fit
plt.figure(figsize=(8,6))
best_a, best_b = best_params_grid[min_idx]
x_range = np.linspace(-1, 1, 100)
plt.errorbar(original_Eiso_bc, Epeak_bc, yerr=Epeak_err, fmt='o', alpha=0.3, label='GRB Data')
plt.plot(x_range, linear_model(x_range, best_a, best_b), 'r-', label=f'Best fit (slope={best_a:.2f})')
plt.fill_between(x_range, linear_model(x_range, best_a, best_b) - extra_err, 
                 linear_model(x_range, best_a, best_b) + extra_err, color='red', alpha=0.1)
plt.title("Amati Relation (Best Fit Cosmology)")
plt.savefig("../figures/paper_amati_fit.png")

# Figure B: Density Contours (Slice at best H0)
plt.figure(figsize=(8,6))
chi_slice = chi_volume[:, :, min_idx[2]]
min_chi = np.min(chi_slice)
plt.contourf(Om_grid, Ode_grid, chi_slice.T, levels=20, cmap='viridis_r')
plt.contour(Om_grid, Ode_grid, chi_slice.T, levels=[min_chi+2.3, min_chi+4.61], colors='white')
plt.xlabel(r'$\Omega_m$'); plt.ylabel(r'$\Omega_\Lambda$')
plt.title(f"LCDM Constraints (H0 = {best_H0:.1f})")
plt.savefig("../figures/paper_lcdm_grb_contours.png")

np.save("../data/chi_volume_grb_lcdm.npy", chi_volume)

