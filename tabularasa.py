import numpy as np
import pandas as pd
from astropy.cosmology import LambdaCDM
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

df = pd.read_csv("table.csv")
z = df['z']

# Proper error propagation for logarithmic quantities
Epeak = np.log10(df['Epeak'])
Epeak_err = df['Epeak_err'] / (df['Epeak'] * np.log(10))
Epeak_bc = Epeak - np.mean(Epeak)

extra = np.log10(3)

original_Eiso = np.log10(df['Eiso'])
original_Eiso_err = df['Eiso_err'] / (df['Eiso'] * np.log(10))
original_Eiso_bc = original_Eiso - np.mean(original_Eiso)

# More physically motivated parameter space
Om = np.linspace(0.0, 2.0, 100)
Ode = np.linspace(0.0, 2.0, 100)

standard_cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
standard_dl = standard_cosmo.luminosity_distance(z)

chi_surface = np.zeros([len(Om), len(Ode)])
a = np.zeros([len(Om), len(Ode)])
b = np.zeros([len(Om), len(Ode)])
mask = np.zeros([len(Om), len(Ode)])

def model(x, p1, p2):
    return p1*x + p2

# for i in range(len(Om)):
#     for j in range(len(Ode)):
    cosmo = LambdaCDM(H0=70, Om0=Om[i], Ode0=Ode[j])
    dl = cosmo.luminosity_distance(z)

    Eiso = 2*np.log10(dl/standard_dl) + original_Eiso
    Eiso_bc = Eiso - (np.sum(Eiso)/len(Eiso))

    if np.isnan(Eiso).any():
        a[i, j], b[i, j] = -1e4, -1e4
        chi_surface[i, j] = 1e4
        mask[i,j] = True
        continue

    total_err = Epeak_err + extra

    popt, pcov = curve_fit(model, Eiso_bc, Epeak_bc, sigma=total_err, p0=[0.5, 0.0], bounds=([-10.0, -10.0], [10.0, 10.0]))
    a[i,j], b[i,j] = popt
    residuals = Epeak_bc - model(Eiso_bc, *popt)
    chi_surface[i,j] = np.sum((residuals/(Epeak_err+extra)**2))
    #chi_surface[i,j] = np.sum((residuals/Epeak_err)**2)

masked_chi_surface = np.ma.array(chi_surface, mask=mask)

min_chi = np.nanmin(masked_chi_surface)
i1, i2 = np.unravel_index(np.nanargmin(masked_chi_surface), masked_chi_surface.shape)
Om_fit, Ode_fit = Om[i1], Ode[i2]

# Plotting
plt.figure()
plt.contourf(Om, Ode, masked_chi_surface.T, levels=50)
plt.contour(Om, Ode, masked_chi_surface.T, levels=[min_chi+2.3, min_chi+4.61, min_chi+6.17], colors='r', alpha=[1.0, 0.75, 0.5])
plt.scatter(Om_fit, Ode_fit, c='r', s=50, marker='x', label=f'Best fit: Om={Om_fit:.2f}, Ode={Ode_fit:.2f}, {min_chi}')
plt.xlabel('Om')
plt.ylabel('Ode')
plt.title('chi2 surface')
plt.legend()
plt.show()