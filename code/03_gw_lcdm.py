import h5py
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import LambdaCDM
from scipy.optimize import curve_fit
from tqdm import tqdm

# Read GW data
with h5py.File('../data/nsns_population_joan.hdf5', 'r') as table:
    #print(table.keys())
    dl = table['dL'][()][0:999]
    z = table['z'][()][0:999]
    # dl_err = table['deldL'][()][0:999]

dl_err = 0.01 * dl # fix error 5% of dL value (Note: 0.01 is 1%, update to 0.05 if 5% is intended)

# Grid setup
N = 15

Om_grid = np.linspace(0.1, 2.0, N)
Ode_grid = np.linspace(0.1, 2.0, N)
H0_grid = np.linspace(60.0, 80.0, N) # Adjusted to match curve_fit bounds and typical H0 ranges

chi_volume = np.zeros([N, N, N])
mask = np.zeros([N, N, N], dtype=bool)

factor = 1
extra = np.log10(factor)
total_err = dl_err + extra # Standardized variable name

def auto_test_model(x, H0, Om, Ode):
    cosmo = LambdaCDM(H0=H0, Om0=Om, Ode0=Ode) # Fixed variable name
    return cosmo.luminosity_distance(x).value

# Initial Quick Fit
popt, pcov = curve_fit(auto_test_model, z, dl, p0=[70.0, 1.0, 1.0], bounds=([60.0, 0.0, 0.0], [80.0, 2.0, 2.0]))
H0_auto_test, Om_auto_test, Ode_auto_test = popt
print(f'H0 test: {H0_auto_test:.3f}')
print(f'Om test: {Om_auto_test:.3f}')
print(f'Ode test: {Ode_auto_test:.3f}')

# Grid Search
for i in tqdm(range(N), desc="Scanning LCDM Volume"):
    for j in range(N):
        for k in range(N):
            cosmo = LambdaCDM(H0=H0_grid[i], Om0=Om_grid[j], Ode0=Ode_grid[k])
            model = cosmo.luminosity_distance(z).value
            
            if np.isnan(model).any():
                mask[i,j,k] = True
                chi_volume[i,j,k] = 1e6
                continue

            residuals = dl - model
            dof = len(z) - 3 # 3 parameters (H0, Om, Ode)
            chi_volume[i,j,k] = np.sum((residuals/total_err)**2) / dof

masked_chi_volume = np.ma.array(chi_volume, mask=mask)

min_chi = np.min(masked_chi_volume)
min_idx = np.unravel_index(np.argmin(masked_chi_volume), masked_chi_volume.shape)
best_H0, best_Om, best_Ode = H0_grid[min_idx[0]], Om_grid[min_idx[1]], Ode_grid[min_idx[2]]

print(f'\n--- LCDM Best Fit (GW) ---')
print(f'H0: {best_H0:.3f}, Om: {best_Om:.3f}, Ode: {best_Ode:.3f}')
print(f'min chi: {min_chi:.3f}')

# Flatness Metric: Variance of Chi2 across each dimension
h0_slice_variance = np.var(masked_chi_volume[:, min_idx[1], min_idx[2]])
Om_slice_variance = np.var(masked_chi_volume[min_idx[0], : , min_idx[2]]) # Fixed typo
Ode_slice_variance = np.var(masked_chi_volume[min_idx[0], min_idx[1], :])

print(f"H0 Surface Flatness (Variance): {h0_slice_variance:.2e}")
print(f"Om Surface Flatness (Variance): {Om_slice_variance:.2e}")
print(f"Ode Surface Flatness (Variance): {Ode_slice_variance:.2e}")

# ==========================================
# Plotting
# ==========================================

# Figure A: Hubble Diagram
plt.figure(figsize=(8,6))
z_sorted_idx = np.argsort(z)
z_sorted = z[z_sorted_idx]
dl_sorted = dl[z_sorted_idx]

# Calculate best fit model for plotting
best_cosmo = LambdaCDM(H0=best_H0, Om0=best_Om, Ode0=best_Ode)
model_dl = best_cosmo.luminosity_distance(z_sorted).value

plt.errorbar(z, dl, yerr=total_err, fmt='o', alpha=0.3, label='GW Data')
plt.plot(z_sorted, model_dl, 'r-', linewidth=2, 
         label=f'Best fit\n($H_0$={best_H0:.1f}, $\\Omega_m$={best_Om:.2f}, $\\Omega_\\Lambda$={best_Ode:.2f})')

plt.xlabel('Redshift $z$')
plt.ylabel('Luminosity Distance $d_L$ (Mpc)')
plt.title("GW Hubble Diagram (Best Fit Cosmology)")
plt.legend()
plt.savefig("../figures/paper_gw_hubble_diagram.png")


# Figure B: Density Contours (Slice at best H0)
plt.figure(figsize=(8,6))

# The array is structured as [H0, Om, Ode], so we slice at min_idx[0]
chi_slice = masked_chi_volume[min_idx[0], :, :]
min_chi_slice = np.min(chi_slice)

# Generate contours
plt.contourf(Om_grid, Ode_grid, chi_slice.T, levels=20, cmap='viridis_r')
plt.contour(Om_grid, Ode_grid, chi_slice.T, levels=[min_chi_slice+2.3, min_chi_slice+4.61], colors='white')

plt.xlabel(r'$\Omega_m$')
plt.ylabel(r'$\Omega_\Lambda$')
plt.title(f"LCDM Constraints from GW ($H_0$ = {best_H0:.1f})")
plt.savefig("../figures/paper_lcdm_gw_contours.png")

# Save Volume Data
np.save("../data/chi_volume_gw_lcdm.npy", chi_volume)
