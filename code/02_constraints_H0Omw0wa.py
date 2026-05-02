import numpy as np
import pandas as pd
from astropy.cosmology import w0waCDM
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tqdm import tqdm

# 1. Setup
df = pd.read_csv("../data/table.csv")
z = df['z'].values
Epeak = np.log10(df['Epeak']).values
Epeak_err = df['Epeak_err'].values / (df['Epeak'].values * np.log(10))
Epeak_bc = Epeak - np.mean(Epeak)
original_Eiso = np.log10(df['Eiso']).values
original_Eiso_bc = original_Eiso - np.mean(original_Eiso)

# 2. 4D Grid Setup
N = 12 # Keep N small for 4D (12^4 = 20,736 iterations)
Om_grid = np.linspace(0.1, 0.6, N)
w0_grid = np.linspace(-2.5, 0.5, N)
wa_grid = np.linspace(-3.0, 2.0, N) # Refined physical range
H0_grid = np.linspace(65, 75, N)

extra_err = np.log10(1.30)
standard_cosmo = w0waCDM(H0=70, Om0=0.3, Ode0=0.7, w0=-1.0, wa=0.0)
standard_dl = standard_cosmo.luminosity_distance(z).value

chi_hypervolume = np.zeros((N, N, N, N))

# 3. Computation Loop
for i in tqdm(range(N), desc="Scanning w0wa Volume"):
    for j in range(N):
        for k in range(N):
            for l in range(N):
                # Flatness assumption: Ode = 1 - Om
                cosmo = w0waCDM(H0=H0_grid[l], Om0=Om_grid[i], Ode0=1-Om_grid[i], 
                                w0=w0_grid[j], wa=wa_grid[k])
                dl = cosmo.luminosity_distance(z).value
                
                Eiso = 2 * np.log10(dl / standard_dl) + original_Eiso
                Eiso_bc = Eiso - np.mean(Eiso)
                total_err = np.sqrt(Epeak_err**2 + extra_err**2)
                
                try:
                    popt, _ = curve_fit(lambda x, a, b: a*x+b, Eiso_bc, Epeak_bc, sigma=total_err)
                    residuals = Epeak_bc - (popt[0]*Eiso_bc + popt[1])
                    chi_hypervolume[i, j, k, l] = np.sum((residuals / total_err)**2)
                except:
                    chi_hypervolume[i, j, k, l] = 1e6

# 4. Results
min_idx = np.unravel_index(np.argmin(chi_hypervolume), chi_hypervolume.shape)
print(f"\n--- w0waCDM Best Fit ---")
print(f"Om: {Om_grid[min_idx[0]]:.2f}, w0: {w0_grid[min_idx[1]]:.2f}, wa: {wa_grid[min_idx[2]]:.2f}")

# Flatness Check (Contrast)
chi_min = np.min(chi_hypervolume)
chi_max = np.max(chi_hypervolume[chi_hypervolume < 1e5])
print(f"Chi2 Range (Min/Max): {chi_min:.2f} / {chi_max:.2f}")
print(f"Relative Contrast: {(chi_max - chi_min)/chi_min:.4f}")

# 5. Paper Figure: w0 vs wa (at best Om and H0)
plt.figure(figsize=(8,6))
w0wa_slice = chi_hypervolume[min_idx[0], :, :, min_idx[3]]
plt.contourf(w0_grid, wa_grid, w0wa_slice.T, levels=20, cmap='magma_r')
plt.colorbar(label=r'$\chi^2$')
plt.xlabel(r'$w_0$'); plt.ylabel(r'$w_a$')
plt.title(r"Dark Energy Evolution Constraints ($\Omega_m$ and $H_0$ fixed at best-fit)")
plt.savefig("paper_w0wa_contours.png")
