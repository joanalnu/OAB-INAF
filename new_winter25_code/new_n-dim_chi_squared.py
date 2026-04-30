import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Example model (modify this!)
def model(x, params):
    # Example: exponential decay + polynomial offset
    a, b, c, d = params  # can have N parameters
    return a * np.exp(-b * x) + c * x + d

# --- Example data ---
np.random.seed(0)
x_data = np.linspace(0, 10, 50)
true_params = [2.0, 1.0, 0.2, 0.5]
y_true = model(x_data, true_params)
sigma = 0.1
y_obs = y_true + sigma * np.random.randn(len(x_data))

# --- Define parameter grids ---
param_ranges = {
    "a": np.linspace(1.5, 2.5, 10),
    "b": np.linspace(0.8, 1.2, 10),
    "c": np.linspace(0.0, 0.4, 10),
    "d": np.linspace(0.3, 0.7, 10)
}

param_names = list(param_ranges.keys())
param_grids = [param_ranges[p] for p in param_names]

# --- Compute chi-squared over all parameter combinations ---
results = []

for combo in itertools.product(*param_grids):
    y_model = model(x_data, combo)
    chi2 = np.sum(((y_obs - y_model) / sigma) ** 2)
    results.append(list(combo) + [chi2])

# Convert to dataframe for easy visualization
df = pd.DataFrame(results, columns=param_names + ['chi2'])

# --- Find best fit ---
best_idx = df['chi2'].idxmin()
best_fit = df.loc[best_idx, param_names].to_dict()
print(f"Best-fit parameters: {best_fit}")
print(f"Minimum chi²: {df.loc[best_idx, 'chi2']:.3f}")

# --- Pairwise visualization (corner-like plot) ---
sns.pairplot(df.sample(1000), vars=param_names, hue='chi2', palette='viridis', diag_kind='kde')
plt.suptitle("Pairwise Parameter Relationships Colored by χ²", y=1.02)
plt.show()













# If you go beyond ~4 parameters, a full grid becomes computationally huge (e.g., 10 points per parameter → 10⁴ combinations).
# You can randomly sample parameter space instead:


# Random sampling version (for many parameters)
# n_samples = 5000
# samples = np.array([
#     np.random.uniform(low=min(r), high=max(r), size=n_samples)
#     for r in param_grids
# ]).T

# chi2_list = []
# for s in samples:
#     y_model = model(x_data, s)
#     chi2_list.append(np.sum(((y_obs - y_model) / sigma) ** 2))

# df = pd.DataFrame(samples, columns=param_names)
# df['chi2'] = chi2_list

# sns.pairplot(df, vars=param_names, hue='chi2', palette='viridis', diag_kind='kde')
# plt.show()
