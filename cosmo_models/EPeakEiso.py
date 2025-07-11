import numpy as np
import pandas as pd

# read data
df = pd.read_csv('../table.csv')

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
H0 = 70.0
Om_values = np.linspace(0.0, 2.0, 15)
Ode_values = np.linspace(0.0, 2.0, 15)
Ok = 0.0
Or = 0.0

from parameters import CosmologicalParametersClass
from LambdaCDM import LambdaCDMClass

# Option 1: Create a single model with specific parameter values
params = CosmologicalParametersClass(H0=H0, Omega_m0=0.3, Omega_Lambda0=0.7, Omega_K0=Ok, Omega_r0=Or)
LCDM = LambdaCDMClass(params)
print(LCDM.__dict__)

# Option 2: If you want to iterate over parameter combinations
models = []
for Om in Om_values:
    for Ode in Ode_values:
        # Create parameters for this combination
        params = CosmologicalParametersClass(H0=H0, Omega_m0=Om, Omega_Lambda0=Ode, Omega_K0=Ok, Omega_r0=Or)
        # Create model instance
        model = LambdaCDMClass(params)
        models.append(model)

# Example usage with the first model
print(f"First model Omega_m0: {models[0].Omega_m0}")
print(f"First model H0: {models[0].H0}")

# Test the hubble parameter calculation
z_test = 1.0
H_z = models[0].hubble_param(z_test)
print(f"H(z=1) = {H_z}")

# Test the integrand calculation
integral_result = models[0].integrand_scalar(z_test)
print(f"Integrand at z=1: {integral_result}")