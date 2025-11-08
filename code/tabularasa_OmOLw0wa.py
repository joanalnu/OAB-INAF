# this is the same as tabularasa.py, but here we not only fit (Om, OL) but also (w0, wa) using 4D chi squared volume
# the data used is Giancarlo et al. 2008 (LGRBs)
# we use a fixed H0 of 67.5, results from tabularasa_OmOLH0.py

import numpy as np
import pandas as pd
from astropy.cosmology import w0waCDM
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
w0 = np.linspace(-2.0, 1.0, 25)
wa = np.linspace(-2.0, 1.0, 25)

factor = 1.30
extra_err = np.log10(factor)

standard_cosmo = w0waCDM(H0=67.5, Om0=0.3, Ode0=0.7)
standard_dl = standard_cosmo.luminosity_distance(z)

chi_surface = np.zeros([len(Om), len(Ode), len(w0), len(wa)])
a = np.zeros([len(Om), len(Ode), len(w0), len(wa)])
b = np.zeros([len(Om), len(Ode), len(w0), len(wa)])
mask = np.zeros([len(Om), len(Ode), len(w0), len(wa)])

def model(x, p1, p2):
    return p1*x + p2


for i in tqdm(range(len(Om)), desc="Calculating χ² surface", leave=False):
    for j in range(len(Ode)):
        for k in range(len(w0)):
            for l in range(len(wa)):
                # instead of using astropy.cosmology.LambdaCDM we here use astropy.cosmology.w0waCDM (w0wzCDM is as correct and as far as I know )
                cosmo = w0waCDM(H0=67.5, Om0=Om[i], Ode0=Ode[j], w0=w0[k], wa=wa[l])
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
                a[i,j,k,l], b[i,j,k,l] = popt
                residuals = Epeak_bc - model(Eiso_bc, *popt)
                dof = len(Eiso_bc)-2
                chi_surface[i, j, k, l] = np.sum((residuals / total_err) ** 2)#/dof

masked_chi_surface = np.ma.array(chi_surface, mask=mask)

min_chi = np.nanmin(masked_chi_surface)
i1, i2, i3, i4 = np.unravel_index(np.nanargmin(masked_chi_surface), masked_chi_surface.shape)
Om_fit, Ode_fit, w0_fit, wa_fit = Om[i1], Ode[i2], w0[i3], wa[i4]

print(r"$(\\Omega_m, \\Omega_\\Lambda, H_0)$ Fit Results")
print(f"Om = {Om_fit}")
print(f"Ode = {Ode_fit}")
print(f"w0 = {w0_fit}")
print(f"wa = {wa_fit}")

# Plotting

# w0-wa plot for best Om-Ode fits
plt.figure()
plt.contourf(w0, wa, masked_chi_surface[Om_fit, Ode_fit, :, :].T, levels=50)
plt.xlabel('w0')
plt.ylabel('wa')
plt.title(f'(w0, wa) Chi2 slice at Om={Om_fit:.2f}, Ode={Ode_fit:.2f} (best fit)')
plt.colorbar(label=r"$\\chi^2$")
plt.show()

# corner plot (Om, Ode, H0)
# import corner
# figure = corner.corner(chi2_norm)