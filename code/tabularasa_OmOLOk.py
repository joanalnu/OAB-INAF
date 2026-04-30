# this is a coopy of tabularasa_OmOLH0.py at Nov 19, 2025; instead of H0, here we compute Ok
# this is the same as tabularassa.py, but here we not only fit (Om, OL) but also Ok using 3D chi squared volume
# the data used is Giancarlo et al. 2008 (LGRBs)

import numpy as np
import pandas as pd
from astropy.cosmology import LambdaCDM
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tqdm import tqdm

df = pd.read_csv("../data/table.csv")
z = df['z']

Epeak = np.log10(df['Epeak'])
Epeak_err = df['Epeak_err'] / (df['Epeak'] * np.log(10))
Epeak_bc = Epeak - np.mean(Epeak)

original_Eiso = np.log10(df['Eiso'])
original_Eiso_err = df['Eiso_err'] / (df['Eiso'] * np.log(10))
original_Eiso_bc = original_Eiso - np.mean(original_Eiso)

Om = np.linspace(0.0, 2.0, 25)
Ode = np.linspace(0.0, 2.0, 25)
Ok = np.linspace(0.0, 2.0, 25)

factor = 1.30
extra_err = np.log10(factor)

standard_cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7, Ok0=0.0)
standard_dl = standard_cosmo.luminosity_distance(z)

chi_surface = np.zeros([len(Om), len(Ode), len(Ok)])
a = np.zeros([len(Om), len(Ode), len(Ok)])
b = np.zeros([len(Om), len(Ode), len(Ok)])
mask = np.zeros([len(Om), len(Ode), len(Ok)])

def model(x, p1, p2):
    return p1*x + p2


for i in tqdm(range(len(Om)), desc="Calculating χ² surface", leave=False):
    for j in range(len(Ode)):
        for k in range(len(Ok)):
            # instead of using astropy's LambdaCDM we use our dynamical DE model
            
            cosmo = LambdaCDM(H0=70.0, Om0=Om[i], Ode0=Ode[j], Ok0=Ok[k])
            dl = cosmo.luminosity_distance(z)

            Eiso = 2*np.log10(dl/standard_dl) + original_Eiso
            Eiso_bc = Eiso - (np.sum(Eiso)/len(Eiso))

            if np.isnan(Eiso).any():
                a[i, j, k], b[i, j, k] = -1e4, -1e4
                chi_surface[i, j, k] = 1e4
                mask[i,j,k] = True
                continue

            total_err = Epeak_err + extra_err

            popt, pcov = curve_fit(model, Eiso_bc, Epeak_bc, sigma=total_err, p0=[0.5, 0.0], bounds=([-10.0, -10.0], [10.0, 10.0]))
            a[i,j,k], b[i,j,k] = popt
            residuals = Epeak_bc - model(Eiso_bc, *popt)
            dof = len(Eiso_bc)-2
            chi_surface[i, j, k] = np.sum((residuals / total_err) ** 2)#/dof

masked_chi_surface = np.ma.array(chi_surface, mask=mask)

min_chi = np.nanmin(masked_chi_surface)
i1, i2, i3 = np.unravel_index(np.nanargmin(masked_chi_surface), masked_chi_surface.shape)
Om_fit, Ode_fit, Ok_fit = Om[i1], Ode[i2], Ok[i3]

print(r"$(\\Omega_m, \\Omega_\\Lambda, H_0)$ Fit Results")
print(f"Om = {Om_fit}")
print(f"Ode = {Ode_fit}")
print(f"Ok = {Ok_fit}")

# Plotting

# Om-Ode plot for best hubble constant
cutting_index = Ok_fit
plt.figure()
plt.contourf(Om, Ode, masked_chi_surface[:, :, cutting_index].T, levels=50)
plt.xlabel('Om')
plt.ylabel('Ode')
plt.title(f'Chi2 slice at Ok = {Ok_fit:.2f} (best fit)')
plt.colorbar(label=r"$\\chi^2$")
plt.show()

# 3D visualisation of the chi2 volume (Om, Ode, Ok)-grid (just for fun)
A, B, C = np.meshgrid(Om, Ode, Ok, indexing='ij')
a_flat, b_flat, c_flat = A.flatten(), B.flatten(), C.flatten() # flatten everything for 3D plotting
masked_chi_flat = masked_chi_surface.flatten()

chi2_norm = (masked_chi_flat - np.min(masked_chi_flat) / (np.max(masked_chi_flat) - np.min(masked_chi_flat)))

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(a_flat, b_flat, c_flat, c=chi2_norm, cmap='viridis', s=8, alpha=0.6)
ax.set_xlabel('Om'); ax.set_ylabel('Ode'); ax.set_zlabel('Ok')
ax.set_titled("3D Chi2 volume visualisation")
fig.colorbar(p,label='chi2_norm')
plt.show()

# corner plot (Om, Ode, Ok)
# import corner
# figure = corner.corner(chi2_norm)