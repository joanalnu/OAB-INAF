import h5py
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import w0waCDM
from scipy.optimize import curve_fit
from tqdm import tqdm

# ==========================================
# 1. Read GW Data
# ==========================================
with h5py.File('../data/nsns_population_joan.hdf5', 'r') as table:
    print(table.keys())
    dl = table['dL'][()][0:99] # 999
    z = table['z'][()][0:99] # 999

# Define errors (keeping the logic from 03_gw_lcdm)
dl_err = 0.01 * dl  # 1% error (Update to 0.05 if 5% is intended as per previous comment)
factor = 1
extra = np.log10(factor)
total_err = dl_err + extra

# ==========================================
# 2. 4D Grid Setup
# ==========================================
N = 8 # 12

# Order: Om, w0, wa, H0
Om_grid = np.linspace(0.1, 0.6, N)
w0_grid = np.linspace(-2.5, 0.5, N)
wa_grid = np.linspace(-3.0, 2.0, N)
H0_grid = np.linspace(65, 75, N)

chi_hypervolume = np.zeros([N, N, N, N])
mask = np.zeros([N, N, N, N], dtype=bool)

# Optional: Quick initial fit via curve_fit
def auto_test_model_w0wa(x, H0, Om, w0, wa):
    # Flatness assumption implemented here (Ode0 = 1 - Om0)
    cosmo = w0waCDM(H0=H0, Om0=Om, Ode0=1.0-Om, w0=w0, wa=wa)
    return cosmo.luminosity_distance(x).value

try:
    popt, pcov = curve_fit(
        auto_test_model_w0wa, z, dl, 
        p0=[70.0, 0.3, -1.0, 0.0], 
        bounds=([60.0, 0.0, -3.0, -5.0], [80.0, 1.0, 1.0, 5.0])
    )
    print(f"Curve_fit initial guess:")
    print(f"H0: {popt[0]:.2f}, Om: {popt[1]:.2f}, w0: {popt[2]:.2f}, wa: {popt[3]:.2f}\n")
except Exception as e:
    print(f"Initial fit failed: {e}\n")

# ==========================================
# 3. Computation Loop
# ==========================================
for i in tqdm(range(N), desc="Scanning w0wa Volume"):
    for j in range(N):
        for k in range(N):
            for l in range(N):
                # Apply flatness assumption: Ode0 = 1 - Om
                cosmo = w0waCDM(H0=H0_grid[l], Om0=Om_grid[i], Ode0=1-Om_grid[i], 
                                w0=w0_grid[j], wa=wa_grid[k])
                
                model = cosmo.luminosity_distance(z).value
                
                # Check for unphysical cosmology issues (NaNs)
                if np.isnan(model).any():
                    mask[i, j, k, l] = True
                    chi_hypervolume[i, j, k, l] = 1e6
                    continue

                residuals = dl - model
                dof = len(z) - 4 # 4 parameters (H0, Om, w0, wa)
                chi_hypervolume[i, j, k, l] = np.sum((residuals/total_err)**2) / dof

# ==========================================
# 4. Results & Contrast
# ==========================================
masked_chi_hypervolume = np.ma.array(chi_hypervolume, mask=mask)

min_chi = np.min(masked_chi_hypervolume)
min_idx = np.unravel_index(np.argmin(masked_chi_hypervolume), masked_chi_hypervolume.shape)

best_Om = Om_grid[min_idx[0]]
best_w0 = w0_grid[min_idx[1]]
best_wa = wa_grid[min_idx[2]]
best_H0 = H0_grid[min_idx[3]]

print(f"\n--- w0waCDM Best Fit (GW Data) ---")
print(f"Om: {best_Om:.3f}")
print(f"w0: {best_w0:.3f}")
print(f"wa: {best_wa:.3f}")
print(f"H0: {best_H0:.3f}")
print(f"min chi: {min_chi:.3f}")

# Flatness Check (Contrast)
valid_chi = masked_chi_hypervolume[masked_chi_hypervolume < 1e5]
if len(valid_chi) > 0:
    chi_max = np.max(valid_chi)
    print(f"Chi2 Range (Min/Max): {min_chi:.2f} / {chi_max:.2f}")
    print(f"Relative Contrast: {(chi_max - min_chi)/min_chi:.4f}")

# ==========================================
# 5. Plotting
# ==========================================

# Figure A: Hubble Diagram
plt.figure(figsize=(8,6))
z_sorted_idx = np.argsort(z)
z_sorted = z[z_sorted_idx]
dl_sorted = dl[z_sorted_idx]

# Calculate best fit model for plotting
best_cosmo = w0waCDM(H0=best_H0, Om0=best_Om, Ode0=1.0-best_Om, w0=best_w0, wa=best_wa)
model_dl = best_cosmo.luminosity_distance(z_sorted).value

plt.errorbar(z, dl, yerr=total_err, fmt='o', alpha=0.3, label='GW Data')
plt.plot(z_sorted, model_dl, 'r-', linewidth=2, 
         label=f'Best fit\n($\\Omega_m$={best_Om:.2f}, $w_0$={best_w0:.2f}, $w_a$={best_wa:.2f}, $H_0$={best_H0:.1f})')

plt.xlabel('Redshift $z$')
plt.ylabel('Luminosity Distance $d_L$ (Mpc)')
plt.title("GW Hubble Diagram ($w_0w_a$CDM)")
plt.legend()
plt.savefig("../figures/paper_gw_w0wa_hubble.png")


# Figure B: w0 vs wa Contour (Sliced at best H0 and Om)
plt.figure(figsize=(8,6))

# Slicing the hypervolume at the indices of best Om and best H0
w0wa_slice = masked_chi_hypervolume[min_idx[0], :, :, min_idx[3]]
min_chi_slice = np.min(w0wa_slice)

plt.contourf(w0_grid, wa_grid, w0wa_slice.T, levels=20, cmap='magma_r')
plt.colorbar(label=r'$\chi^2_{\nu}$')
plt.contour(w0_grid, wa_grid, w0wa_slice.T, levels=[min_chi_slice+2.3, min_chi_slice+4.61], colors='white', linestyles=['solid', 'dashed'])

plt.xlabel(r'$w_0$')
plt.ylabel(r'$w_a$')
plt.title(f"Dark Energy Evolution Constraints from GW\n(Fixed $\\Omega_m$={best_Om:.2f}, $H_0$={best_H0:.1f})")
plt.savefig("../figures/paper_gw_w0wa_contours.png")

# Save Data
np.save("../data/chi_hypervolume_gw_w0wa.npy", chi_hypervolume)
