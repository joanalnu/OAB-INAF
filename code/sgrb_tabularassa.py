# 3. barycenter correction
# 4. fit for single Om, Ode
# 5. fit for varying Om, Ode


import h5py
import numpy as np
import matplotlib.pyplot as plt


# 1. reading data
with h5py.File('../data/nsns_population_joan.hdf5', 'r') as table:
    z = table['z'][()]
    Epeak = np.log10(table['Epeak'][()])
    Epeak_err = 0.1 * Epeak # 10% errors on Epeak
    linear_err = 0.05*table['Epeak'][()]
    Epeak_err = linear_err / (table['Epeak'][()] + np.log(10))

    original_Eiso = np.log10(table['Eiso'][()])

nan_mask1 = np.isnan(original_Eiso)
nan_mask2 = np.isnan(Epeak)
combined_nan_mask = nan_mask1 | nan_mask2

inf_mask1 = np.isinf(original_Eiso)
inf_mask2 = np.isinf(Epeak)
combined_inf_mask = inf_mask1 | inf_mask2

outliner_mask = [val<40 for val in original_Eiso]
# for i, x in enumerate(outliner_mask):
#     if x==True:
#         print(i,x)

total_combined_mask = combined_nan_mask | combined_inf_mask | outliner_mask
inverse_mask = ~total_combined_mask

z = z[inverse_mask]
original_Eiso = original_Eiso[inverse_mask]
Epeak = Epeak[inverse_mask]
Epeak_err = Epeak_err[inverse_mask]

sorting_indices = np.argsort(original_Eiso)

z = z[sorting_indices]
original_Eiso = original_Eiso[sorting_indices]
Epeak = Epeak[sorting_indices]
Epeak_err = Epeak_err[sorting_indices]

# cut data down to max. 100 points
# z = z[:1000]
# original_Eiso = original_Eiso[:1000]
# Epeak = Epeak[:1000]
# Epeak_err = Epeak_err[:1000]



# barycenter correection
xcm, ycm = np.mean(original_Eiso), np.mean(Epeak)
original_Eiso_bc = original_Eiso - xcm
Epeak_bc = Epeak - ycm

# data preview
plt.scatter(original_Eiso_bc, Epeak_bc, s=3); plt.xlabel('Eiso'); plt.xlabel('Epeak')
plt.close()


# compute standard cosmo fit

def gof(a, b, x, y, sigy):
    predicted = a*x+b
    chi = np.sum((y - predicted)/sigy)
    return chi**2

a = np.linspace(-1.0, 1.0, 20)
b = np.linspace(-20.0, 20.0, 20)
chi_squared_surface = np.zeros([len(a), len(b)])

for i in range(len(a)):
    for j in range(len(b)):
        chi_squared_surface[i,j] = gof(a[i], b[j], original_Eiso_bc, Epeak_bc, Epeak_err)

plt.contourf(a, b, chi_squared_surface)
plt.xlabel('a (slope)'); plt.ylabel('b (intercept)'); plt.show()