#!/usr/bin/env python3
"""
Corrected version of your SGRB χ²-surface script.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import LambdaCDM
from scipy.optimize import curve_fit
from tqdm import tqdm
import os

# === CONFIG ===
DATA_PATH = '../data/nsns_population_joan.hdf5'
OUT_DIR = '../figures'
OUT_CHI_NPY = '../data/SGRB_chi_surface.npy'
OUT_MASK_NPY = '../data/SGRB_mask.npy'
OUT_FIG = os.path.join(OUT_DIR, 'SGRBX2.png')

# Make sure output folder exists
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUT_CHI_NPY), exist_ok=True)

# === 1) Read data and robustly remove NaNs / Infs ===
with h5py.File(DATA_PATH, 'r') as table:
    print("Keys in HDF5:", list(table.keys()))
    z = np.array(table['z'][()])                # redshift
    Epeak = np.log10(np.array(table['Epeak'][()]))   # log10(Epeak)
    Eiso_log = np.log10(np.array(table['Eiso'][()])) # log10(Eiso) original
    dL = np.array(table['dL'][()])              # luminosity distance (if present)

# Create mask for rows with any NaN/Inf
row_mask = (
    np.isnan(z) | np.isnan(Epeak) | np.isnan(Eiso_log) | np.isnan(dL) |
    np.isinf(z) | np.isinf(Epeak) | np.isinf(Eiso_log) | np.isinf(dL)
)

# Apply mask
z = z[~row_mask]
Epeak = Epeak[~row_mask]
Eiso_log = Eiso_log[~row_mask]
dL = dL[~row_mask]

# Optional slice (you used z = z[50:] etc. in the original)
# Uncomment if you want to drop first 50 entries:
# slice_start = 50
# z = z[slice_start:]
# Epeak = Epeak[slice_start:]
# Eiso_log = Eiso_log[slice_start:]
# dL = dL[slice_start:]

# Quick check
print(f"Loaded {len(z)} valid entries after masking.")

# === 2) Quick visualization (barycenter) ===
plt.figure(figsize=(6,6))
plt.scatter(Eiso_log, Epeak, s=6, label='data')
mean_x = np.mean(Eiso_log)
mean_y = np.mean(Epeak)
plt.scatter(mean_x, mean_y, marker='x', c='red', s=100,
            label=f'barycenter ({mean_x:.2f}, {mean_y:.2f})')
plt.xlabel('log10(Eiso)')
plt.ylabel('log10(Epeak)')
plt.legend()
plt.tight_layout()
plt.show()

# === 3) Single-cosmology linear fit (chi-squared surface example) ===
# center data (barycentric)
Eiso_bc = Eiso_log - np.mean(Eiso_log)
Epeak_bc = Epeak - np.mean(Epeak)

def chi_squared_simple(x, y, a, b):
    model = a * x + b
    return np.sum((y - model) ** 2)

a_grid = np.linspace(0.0, 1.0, 100)
b_grid = np.linspace(-1.0, 1.0, 100)
chi_surface_simple = np.zeros((len(a_grid), len(b_grid)))

for i, ai in enumerate(a_grid):
    for j, bj in enumerate(b_grid):
        chi_surface_simple[i, j] = chi_squared_simple(Eiso_bc, Epeak_bc, ai, bj)

idx_flat = np.argmin(chi_surface_simple)
idx2 = np.unravel_index(idx_flat, chi_surface_simple.shape)
a_best_simple = a_grid[idx2[0]]
b_best_simple = b_grid[idx2[1]]
print("Simple best-fit a,b:", a_best_simple, b_best_simple)

plt.figure(figsize=(6,5))
plt.contourf(a_grid, b_grid, chi_surface_simple.T, levels=50, cmap='plasma', alpha=0.9)
plt.scatter(a_best_simple, b_best_simple, c='white', marker='x', s=80, label='best fit')
plt.xlabel('a'); plt.ylabel('b')
plt.title('χ² surface (simple grid search)')
plt.colorbar(label='χ²')
plt.legend()
plt.tight_layout()
plt.show()

# === 4) Full χ² surface scanning over cosmologies ===
# parameter ranges
Om_vals = np.linspace(0.0, 2.0, 50)
Ode_vals = np.linspace(0.0, 2.0, 50)

# standard cosmology reference (for dL scaling)
standard_cosmo = LambdaCDM(H0=70.0, Om0=0.3, Ode0=0.7)
# astropy returns Quantity; get numeric values for direct np ops
standard_dl = standard_cosmo.luminosity_distance(z).value  # in Mpc (numeric)

# Prepare outputs
chi_surface = np.full((len(Om_vals), len(Ode_vals)), np.nan, dtype=float)
a_fit = np.full_like(chi_surface, np.nan)
b_fit = np.full_like(chi_surface, np.nan)
bad_mask = np.zeros_like(chi_surface, dtype=bool)

def linear_model(x, p1, p2):
    return p1 * x + p2

# Scan
for i in tqdm(range(len(Om_vals)), desc="Calculating χ² surface (Om loop)"):
    for j in range(len(Ode_vals)):
        cosmo = LambdaCDM(H0=70.0, Om0=Om_vals[i], Ode0=Ode_vals[j])
        # numeric luminosity distance for this cosmology
        dl_q = cosmo.luminosity_distance(z)   # Quantity
        if np.any(~np.isfinite(dl_q.value)):
            bad_mask[i, j] = True
            continue
        dl = dl_q.value

        # Shift in log10(Eiso) due to different dL:  Eiso_new_log = Eiso_log + 2*log10(dl/standard_dl)
        # both dl and standard_dl are numeric arrays (same z points)
        with np.errstate(divide='ignore', invalid='ignore'):
            delta_log = 2.0 * np.log10(dl / standard_dl)
        if np.any(~np.isfinite(delta_log)):
            bad_mask[i, j] = True
            continue

        Eiso_modified = Eiso_log + delta_log
        Eiso_mod_bc = Eiso_modified - np.mean(Eiso_modified)

        # ensure arrays are finite and same length
        if (np.isnan(Eiso_mod_bc).any() or np.isinf(Eiso_mod_bc).any() or
                np.isnan(Epeak_bc).any() or np.isinf(Epeak_bc).any()):
            bad_mask[i, j] = True
            continue

        # Fit linear model Epeak_bc = p1 * Eiso_mod_bc + p2
        try:
            popt, pcov = curve_fit(linear_model, Eiso_mod_bc, Epeak_bc,
                                   p0=[0.5, 0.0],
                                   bounds=([-10.0, -10.0], [10.0, 10.0]))
        except Exception:
            # fitting failed for this cosmology
            bad_mask[i, j] = True
            continue

        # compute residuals and chi^2 (here we use unweighted sum of squares; replace with a proper error model if available)
        residuals = Epeak_bc - linear_model(Eiso_mod_bc, *popt)
        chi2 = np.sum(residuals ** 2)
        chi_surface[i, j] = chi2
        a_fit[i, j], b_fit[i, j] = popt

# Mask the bad points
masked_chi = np.ma.array(chi_surface, mask=bad_mask)

# Find best-fit cosmology
if np.all(np.isnan(masked_chi)):
    raise RuntimeError("All χ² values are NaN — something went wrong (check input data and cosmology grid).")

min_chi = np.nanmin(masked_chi)
min_idx = np.unravel_index(np.nanargmin(masked_chi), masked_chi.shape)
Om_best = Om_vals[min_idx[0]]
Ode_best = Ode_vals[min_idx[1]]
a_best = a_fit[min_idx]
b_best = b_fit[min_idx]

print(f"Best fit: Om={Om_best:.4f}, Ode={Ode_best:.4f}, χ²={min_chi:.4e}")
print(f"Corresponding linear fit: a={a_best:.4f}, b={b_best:.4f}")

# === 5) Plot χ² surface and confidence contours ===
plt.figure(figsize=(7,6))
# filled chi2
cf = plt.contourf(Om_vals, Ode_vals, masked_chi.T, levels=60, cmap='viridis')
cbar = plt.colorbar(cf)
cbar.set_label('χ² (sum of squares)')

# overlay confidence-level contours around min_chi
# For two parameters, approximate Δχ² for 1σ, 2σ, 3σ are: 2.30, 6.18, 11.83
levels = [min_chi + 2.30, min_chi + 6.18, min_chi + 11.83]
# Only draw levels that are finite and greater than min_chi
valid_levels = [lv for lv in levels if np.isfinite(lv) and lv > min_chi]
if valid_levels:
    CS = plt.contour(Om_vals, Ode_vals, masked_chi.T, levels=valid_levels, colors='red', linestyles=['solid','dashed','dotted'], linewidths=1.2)
    plt.clabel(CS, inline=True, fontsize=8, fmt="%.2f")

# best fit and standard LCDM marker
plt.scatter(Om_best, Ode_best, c='white', marker='x', s=80, label=f'Best fit: Om={Om_best:.3f}, Ode={Ode_best:.3f}')
plt.scatter(0.3, 0.7, c='black', marker='x', s=80, label='Standard LCDM (0.3, 0.7)')

plt.xlabel('Om'); plt.ylabel('Ode')
plt.title(r'$\chi^2$ surface (sum of squared residuals)')
plt.xlim(Om_vals.min(), Om_vals.max())
plt.ylim(Ode_vals.min(), Ode_vals.max())
plt.legend(loc='upper right')
plt.tight_layout()

# save then show
plt.savefig(OUT_FIG, dpi=300)
print("Saved figure to", OUT_FIG)
plt.show()

# === 6) Save results ===
np.save(OUT_CHI_NPY, chi_surface)
np.save(OUT_MASK_NPY, bad_mask)
print("Saved chi surface and mask arrays.")
