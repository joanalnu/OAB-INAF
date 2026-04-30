import numpy as np
import matplotlib.pyplot as plt

# --- Example data ---
x_data = np.linspace(0, 10, 50)
true_params = [2.0, 1.0, 0.5]
y_true = true_params[0] * np.exp(-true_params[1] * x_data) + true_params[2]
noise = 0.1 * np.random.randn(len(x_data))
y_obs = y_true + noise
sigma = np.full_like(y_obs, 0.1)

# --- Model function ---
def model(x, a, b, c):
    return a * np.exp(-b * x) + c

# --- Parameter grids ---
a_vals = np.linspace(1.0, 3.0, 30)
b_vals = np.linspace(0.5, 1.5, 30)
c_vals = np.linspace(0.0, 1.0, 30)

# --- Chi-squared volume computation ---
chi2_volume = np.zeros((len(a_vals), len(b_vals), len(c_vals)))

for i, a in enumerate(a_vals):
    for j, b in enumerate(b_vals):
        for k, c in enumerate(c_vals):
            y_model = model(x_data, a, b, c)
            chi2 = np.sum(((y_obs - y_model) / sigma) ** 2)
            chi2_volume[i, j, k] = chi2

# --- Example: visualize a slice of the volume ---
# Fix one parameter (e.g., c = c_vals[15]) and show chi²(a,b)
c_index = 15
plt.figure()
plt.contourf(a_vals, b_vals, chi2_volume[:, :, c_index].T, levels=50)
plt.xlabel("a")
plt.ylabel("b")
plt.title(f"Chi² slice at c = {c_vals[c_index]:.2f}")
plt.colorbar(label="χ²")
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D plots)

# Assuming you have chi2_volume, a_vals, b_vals, c_vals from earlier
A, B, C = np.meshgrid(a_vals, b_vals, c_vals, indexing='ij')

# Flatten everything for 3D plotting
a_flat = A.flatten()
b_flat = B.flatten()
c_flat = C.flatten()
chi2_flat = chi2_volume.flatten()

# Optional: normalize chi² for better color mapping
chi2_norm = (chi2_flat - np.min(chi2_flat)) / (np.max(chi2_flat) - np.min(chi2_flat))

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with χ² as color
p = ax.scatter(a_flat, b_flat, c_flat, c=chi2_norm, cmap='viridis', s=8, alpha=0.6)

ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_zlabel('c')
ax.set_title('3D Chi-squared Volume Visualization')

fig.colorbar(p, label='Normalized χ²')
plt.show()






import plotly.graph_objects as go
import numpy as np

A, B, C = np.meshgrid(a_vals, b_vals, c_vals, indexing='ij')

fig = go.Figure(data=go.Isosurface(
    x=A.flatten(),
    y=B.flatten(),
    z=C.flatten(),
    value=chi2_volume.flatten(),
    isomin=np.min(chi2_volume),
    isomax=np.min(chi2_volume) + 5,  # adjust range for clarity
    surface_count=4,
    caps=dict(x_show=False, y_show=False, z_show=False),
    colorscale='Viridis'
))

fig.update_layout(
    scene=dict(
        xaxis_title='a',
        yaxis_title='b',
        zaxis_title='c'
    ),
    title='Chi-squared 3D Isosurface Visualization'
)
fig.show()








min_idx = np.unravel_index(np.argmin(chi2_volume), chi2_volume.shape)
best_a, best_b, best_c = a_vals[min_idx[0]], b_vals[min_idx[1]], c_vals[min_idx[2]]
print(f"Best fit: a={best_a:.3f}, b={best_b:.3f}, c={best_c:.3f}, χ²={chi2_volume[min_idx]:.3f}")

ax.scatter(best_a, best_b, best_c, color='red', s=60, label='Min χ²')
ax.legend()
