import numpy as np
import pandas as pd

# read data
df = pd.read_csv('./table.csv')

z = df['z']
Epeak = np.log10(df['Epeak'])
Epeak_err = df['Epeak_err'] / (df['Epeak'] * np.log(10))
ycm = np.sum(Epeak) / len(Epeak)
Epeak_bc = Epeak - ycm

original_Eiso = np.log10(df['Eiso'])
original_Eiso_err = df['Eiso_err'] / (df['Eiso'] * np.log(10))
original_xcm = np.sum(original_Eiso) / len(original_Eiso)
Eiso_err = original_Eiso_err  # cosmology invariant

# define the parameter space
H0 = 70.0 * 1e5 / 3.086e24 # s^-1
Om = np.linspace(0.0, 2.0, 15)
Ode = np.linspace(0.0, 2.0, 15)
Ok = 0.0
Or = 0.0

from parameters import CosmologicalParametersClass
from myLambdaCDM import LambdaCDMClass

# Option 1: Create a single model with specific parameter values
# params = CosmologicalParametersClass(H0=H0, Omega_m0=0.3, Omega_Lambda0=0.7, Omega_K0=Ok, Omega_r0=Or)
# LCDM = LambdaCDMClass(params)
# print(LCDM.__dict__)

# computing standard luminosity distance (dl(70, O.3, 0.7, 0.0, 0.0))
standard_params = CosmologicalParametersClass(H0=(70.0 * 1e5 / 3.086e24), Omega_m0=0.3, Omega_Lambda0=0.7, Omega_K0=0.0, Omega_r0=0.0)
standard_dl = LambdaCDMClass(standard_params).luminosity_distance(z)

from GRB import GRBClass
# initialize EpeakGRB to obtain Epeak barycenter and don't have to compute it each cycle
Epeak_bc = GRBClass().bc(Epeak)

from stats import *
stats = Stats()

# define the parameter space for the Epeak-Eiso correlation
m_grid = np.linspace(.1,.9,50)
k_grid = np.linspace(-1., 1., 50)

chi_surface = np.zeros([len(Om), len(Ode)]) # this determines the fitting parameters

# fitting over a parameter grid
# for i, j in tqdm(itertools.product(range(len(Om)), range(len(Ode))), total=len(Om) * len(Ode), desc="Cosmological fit"):
for i in range(len(Om)):
    for j in range(len(Ode)):
        # create a universe
        params = CosmologicalParametersClass(H0=H0, Omega_m0=Om[i], Omega_Lambda0=Ode[j], Omega_k0=Ok, Omega_r0=Or)

        LCDMUniverse = LambdaCDMClass(params)

        # now compute fit with LCMDUniverse.luminosity_distance(z)
        GRB = GRBClass(z=z, Epeak=Epeak, Eiso=original_Eiso, LCDM=LCDMUniverse)
        Eiso = GRB.isotropic_equivalent_energy(z=z, DL0=standard_dl)

        # check that there are not infinite or NaN values
        # can happen with Om=0.0 Ode=0.0
        if np.any(~np.isfinite(Eiso)):
            tqdm.write(f'{i} {j}\tOm={Om[i]}, Ode={Ode[j]}, Eiso contains infinity on NaN')
            continue

        m, k, chi_surface[i,j], correlation_chi_squared = stats.fitter(m_grid, k_grid, GRB.bc(Eiso), Epeak_bc, xerr=None, yerr=Epeak_err, extra_scatter=None, model="linear_model")


mininum, Om_fit, Ode_fit = stats.bestfit_2d(chi_surface, Om, Ode) # this is determined by the fitting parameters
plotting = plotting(Om=Om, Ode=Ode)
plot = plotting.create_contour()



