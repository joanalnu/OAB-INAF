# The initial intention of this script was to extend the chi2 surface analysis to a 4D hypervolume fitting the parameters (Om, Ode, w0, wa). However, this is computationally very expensive. Since I obtained very similar results for Om and Ode both in the (Om, Ode) and (Om, Ode, H0) fits I'll be using fixed values corresponding to the (Om, Ode, H0) fit results and only fitting (w0, wa) here, in order to get insight about what this data could tell us about dynamical dark energy (or perhaps only constant).

# Therefore the code structure is identical to tabularasa.py, only fitting (Om, Ode) with astropy.cosmology.w0waCDM instead of (Om, Ode) using astropy.cosmology.LambdaCDM

# the data used is Giancarlo et al. 2008 (LGRBs)

# we are using Om = fixed_Om, Ode=fixed_Ode, H0=fixed_H0
fixed_Om = 0.3
fixed_Ode = 0.7
fixed_H0 = 70.0 # km/s/Mpc

import numpy as np
import pandas as pd
from astropy.cosmology import w0waCDM
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tqdm import tqdm

df = pd.read_csv("../data/table.csv") # when running from mother directory (oab-inaf) change to only 'data/table.csv'
z = df['z']

Epeak = np.log10(df['Epeak'])
Epeak_err = df['Epeak_err'] / (df['Epeak'] * np.log(10))
Epeak_bc = Epeak - np.mean(Epeak)

original_Eiso = np.log10(df['Eiso'])
original_Eiso_err = df['Eiso_err'] / (df['Eiso'] * np.log(10))
original_Eiso_bc = original_Eiso - np.mean(original_Eiso)

w0 = np.linspace(-10.0, 10.0, 50)
wa = np.linspace(-150.0, 10.0, 50)

factor = 1.30
extra_err = np.log10(factor)

standard_cosmo = w0waCDM(H0=fixed_H0, Om0=fixed_Om, Ode0=fixed_Ode, w0=-1.0, wa=0.0)
standard_dl = standard_cosmo.luminosity_distance(z)

chi_surface = np.zeros([len(w0), len(wa)])
a = np.zeros([len(w0), len(wa)])
b = np.zeros([len(w0), len(wa)])
mask = np.zeros([len(w0), len(wa)])


def model(x, p1, p2):
    return p1*x + p2


for i in tqdm(range(len(w0)), desc="Calculating χ² surface", leave=True):
    for j in range(len(wa)):
        # instead of using astropy.cosmology.LambdaCDM we here use astropy.cosmology.w0waCDM (w0wzCDM is as correct and as far as I know )
        cosmo = w0waCDM(H0=fixed_H0, Om0=fixed_Om, Ode0=fixed_Ode, w0=w0[i], wa=wa[j])
        dl = cosmo.luminosity_distance(z)

        Eiso = 2*np.log10(dl/standard_dl) + original_Eiso
        Eiso_bc = Eiso - (np.sum(Eiso)/len(Eiso))

        if np.isnan(Eiso).any():
            a[i, j], b[i, j] = -1e4, -1e4
            chi_surface[i, j] = 1e4
            mask[i,j] = True
            continue

        total_err = Epeak_err + extra_err

        popt, pcov = curve_fit(model, Eiso_bc, Epeak_bc, sigma=total_err, p0=[0.5, 0.0], bounds=([0.1, -10.0], [10.0, 10.0]))
        a[i,j], b[i,j] = popt
        residuals = Epeak_bc - model(Eiso_bc, *popt)
        dof = len(Eiso_bc)-2
        chi_surface[i, j] = np.sum((residuals / total_err) ** 2)#/dof

masked_chi_surface = np.ma.array(chi_surface, mask=mask)

min_chi = np.nanmin(masked_chi_surface)
i1, i2, = np.unravel_index(np.nanargmin(masked_chi_surface), masked_chi_surface.shape)
w0_fit, wa_fit = w0[i1], wa[i2]

print(r"$(w_0, w_a)$ Fit Results")
print(f"w0 = {w0_fit}")
print(f"wa = {wa_fit}")

# Plotting

# w0-wa chi2 surface plot
plt.figure()
plt.contourf(w0, wa, masked_chi_surface.T, levels=50)
plt.xlabel('w0')
plt.ylabel('wa')
plt.title(f'(w0, wa) Chi2 slice at Om=fixed_Om Ode=fixed_Ode H0=fixed_H0')
plt.colorbar(label="chi-2 figure")
plt.show()

# corner plot (Om, Ode, H0)
# import corner
# figure = corner.corner(chi2_norm)