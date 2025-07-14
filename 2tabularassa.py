import numpy as np
import pandas as pd
from astropy.cosmology import LambdaCDM
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from tqdm import tqdm  # Import the progress bar library

# Read data
df = pd.read_csv("table.csv")
z = df['z']

# Proper error propagation for logarithmic quantities
Epeak = np.log10(df['Epeak'])
Epeak_err = df['Epeak_err'] / (df['Epeak'] * np.log(10))
Epeak_bc = Epeak - np.mean(Epeak)

original_Eiso = np.log10(df['Eiso'])
original_Eiso_err = df['Eiso_err'] / (df['Eiso'] * np.log(10))
original_Eiso_bc = original_Eiso - np.mean(original_Eiso)

# Cosmology parameter space
Om = np.linspace(0.0, 2.0, 50)
Ode = np.linspace(0.0, 2.0, 50)

standard_cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
standard_dl = standard_cosmo.luminosity_distance(z)

# Range of extra values to animate over
extra_values = np.linspace(np.log10(1), np.log10(10), 25)  # Adjust range and number of frames as needed

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))
plt.xlabel('Om')
plt.ylabel('Ode')


# Model function
def model(x, p1, p2):
    return p1 * x + p2


def calculate_chi_surface(extra):
    chi_surface = np.zeros([len(Om), len(Ode)])
    a = np.zeros([len(Om), len(Ode)])
    b = np.zeros([len(Om), len(Ode)])
    mask = np.zeros([len(Om), len(Ode)], dtype=bool)

    # Add progress bar for the chi-square surface calculation
    for i in tqdm(range(len(Om)), desc="Calculating χ² surface", leave=False):
        for j in range(len(Ode)):
            cosmo = LambdaCDM(H0=70, Om0=Om[i], Ode0=Ode[j])
            dl = cosmo.luminosity_distance(z)

            Eiso = 2 * np.log10(dl / standard_dl) + original_Eiso
            Eiso_bc = Eiso - (np.sum(Eiso) / len(Eiso))

            if np.isnan(Eiso).any():
                a[i, j], b[i, j] = -1e4, -1e4
                chi_surface[i, j] = 1e4
                mask[i, j] = True
                continue

            popt, pcov = curve_fit(model, Eiso_bc, Epeak_bc, sigma=Epeak_err + extra, p0=[0.5, 0.0], bounds=([0.0, -1.0], [1.0, 1.0]))
            a[i, j], b[i, j] = popt
            residuals = Epeak_bc - model(Eiso_bc, *popt)
            chi_surface[i, j] = np.sum((residuals / (Epeak_err + extra)) ** 2)

    masked_chi_surface = np.ma.array(chi_surface, mask=mask)
    return masked_chi_surface


# Initialize with first frame
print("Initializing animation...")
current_extra = extra_values[0]
masked_chi_surface = calculate_chi_surface(current_extra)
min_chi = np.nanmin(masked_chi_surface)
i1, i2 = np.unravel_index(np.nanargmin(masked_chi_surface), masked_chi_surface.shape)
Om_fit, Ode_fit = Om[i1], Ode[i2]

# Create initial plot
contourf = ax.contourf(Om, Ode, masked_chi_surface.T, levels=50, cmap=cm.viridis)
contour = ax.contour(Om, Ode, masked_chi_surface.T,
                     levels=[min_chi + 2.3, min_chi + 4.61, min_chi + 6.17],
                     colors='r', alpha=[1.0, 0.75, 0.5])
scatter = ax.scatter(Om_fit, Ode_fit, c='r', s=50, marker='x')
title = ax.set_title(f'χ² surface (extra = {current_extra:.2f})')
cbar = fig.colorbar(contourf)
cbar.set_label('χ²')


def update(frame):
    global contourf, contour, scatter, title

    ax.clear()

    # Calculate new surface
    current_extra = extra_values[frame]
    masked_chi_surface = calculate_chi_surface(current_extra)
    min_chi = np.nanmin(masked_chi_surface)
    i1, i2 = np.unravel_index(np.nanargmin(masked_chi_surface), masked_chi_surface.shape)
    Om_fit, Ode_fit = Om[i1], Ode[i2]

    # Update plot
    contourf = ax.contourf(Om, Ode, masked_chi_surface.T, levels=50, cmap=cm.viridis)
    contour = ax.contour(Om, Ode, masked_chi_surface.T,
                        levels=[min_chi + 2.3, min_chi + 4.61, min_chi + 6.17],
                        colors='r', alpha=[1.0, 0.75, 0.5], linestyles=['-', '--', '..'])
    scatter = ax.scatter(Om_fit, Ode_fit, c='r', s=50, marker='x',
                        label=f'Best fit: Om={Om_fit:.2f}, Ode={Ode_fit:.2f}')
    title.set_text(f'χ² surface (extra = {current_extra:.2f})')
    title = ax.set_title(f'χ² surface (extra = {current_extra:.2f}, factor {(10**current_extra):.3f})')

    return contourf, contour, scatter, title

# Create animation with progress bar
print("Creating animation...")
with tqdm(total=len(extra_values), desc="Rendering frames") as pbar:
    def update_with_progress(frame):
        result = update(frame)
        pbar.update(1)  # Update progress bar
        return result


    ani = animation.FuncAnimation(fig, update_with_progress, frames=len(extra_values),
                                  interval=100, blit=False)

    # Save as MP4
    print("Saving animation...")
    writer = animation.FFMpegWriter(fps=10, bitrate=5000)
    ani.save('chi_surface_animation.mp4', writer=writer)

plt.close()
print("Animation saved as chi_surface_animation.mp4")