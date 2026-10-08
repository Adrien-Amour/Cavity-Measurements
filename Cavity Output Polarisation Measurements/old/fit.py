import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load CSV
df = pd.read_csv("test.csv", sep=',')

# Calculate efficiencies and errors
df['Eff_Ch3'] = 100 * df['Channel 3 Raw Counts'] / df['Valid Runs']
df['Eff_Ch4'] = 100 * df['Channel 4 Raw Counts'] / df['Valid Runs']
df['Err_Ch3'] = np.sqrt(df['Eff_Ch3'] * (1 - df['Eff_Ch3']) / df['Valid Runs'])
df['Err_Ch4'] = np.sqrt(df['Eff_Ch4'] * (1 - df['Eff_Ch4']) / df['Valid Runs'])

# Normalise efficiencies and errors to max = 1
df['Eff_Ch3_norm'] = df['Eff_Ch3'] / df['Eff_Ch3'].max()
df['Eff_Ch4_norm'] = df['Eff_Ch4'] / df['Eff_Ch4'].max()
df['Err_Ch3_norm'] = df['Err_Ch3'] / df['Eff_Ch3'].max()
df['Err_Ch4_norm'] = df['Err_Ch4'] / df['Eff_Ch4'].max()

# Sine fit function
def sin_func(x, A, phi, offset):
    return A * np.sin(2*np.deg2rad(x) + phi) + offset

# Fit Channel 3 (normalised)
popt3, _ = curve_fit(
    sin_func,
    df['Angle (deg)'],
    df['Eff_Ch3_norm'],
    sigma=df['Err_Ch3_norm'],
    absolute_sigma=True,
    p0=[1, 0, 0.5]
)
fit3 = sin_func(df['Angle (deg)'], *popt3)

# Fit Channel 4 (normalised)
popt4, _ = curve_fit(
    sin_func,
    df['Angle (deg)'],
    df['Eff_Ch4_norm'],
    sigma=df['Err_Ch4_norm'],
    absolute_sigma=True,
    p0=[1, 0, 0.5]
)
fit4 = sin_func(df['Angle (deg)'], *popt4)

# Plot
plt.figure(figsize=(8, 5))
plt.errorbar(df['Angle (deg)'], df['Eff_Ch3_norm'], yerr=df['Err_Ch3_norm'], fmt='o', label='Channel 3', capsize=3)
plt.errorbar(df['Angle (deg)'], df['Eff_Ch4_norm'], yerr=df['Err_Ch4_norm'], fmt='o', label='Channel 4', capsize=3)
plt.plot(df['Angle (deg)'], fit3, label='Sine Fit Ch3', color='C0', linestyle='--')
plt.plot(df['Angle (deg)'], fit4, label='Sine Fit Ch4', color='C1', linestyle='--')
plt.xlabel("Angle (deg)")
plt.ylabel("Normalised Efficiency")
plt.title("Normalised Efficiency vs Angle with Sine Fit")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print fit parameters
print("Channel 3 fit params (A, phi, offset):", popt3)
print("Channel 4 fit params (A, phi, offset):", popt4)