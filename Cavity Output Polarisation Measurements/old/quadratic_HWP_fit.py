import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load CSV
df = pd.read_csv("hwp_channel_max_quadratic.csv")  # Change to your actual filename

# Normalise y1 and y2 by their respective rates
df['y1_norm'] = df['y1'] / df['rate1']
df['y2_norm'] = df['y2'] / df['rate2']

# Normalise all to max = 1
df['y1_norm'] = df['y1_norm'] / df['y1_norm'].max()
df['y2_norm'] = df['y2_norm'] / df['y2_norm'].max()
df['rate1_norm'] = df['rate1'] / df['rate1'].max()
df['rate2_norm'] = df['rate2'] / df['rate2'].max()

# Quadratic function
def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

# Fit quadratic to (x1, y1_norm)
popt1, _ = curve_fit(quadratic, df['x1'], df['y1_norm'])
a1, b1, c1 = popt1
min_x1 = -b1 / (2 * a1)
min_y1 = quadratic(min_x1, *popt1)

# Fit quadratic to (x2, y2_norm)
popt2, _ = curve_fit(quadratic, df['x2'], df['y2_norm'])
a2, b2, c2 = popt2
min_x2 = -b2 / (2 * a2)
min_y2 = quadratic(min_x2, *popt2)

print(f"Minima for (x1, y1_norm): x = {min_x1}, y = {min_y1}")
print(f"Minima for (x2, y2_norm): x = {min_x2}, y = {min_y2}")

# Plotting
plt.figure(figsize=(8, 5))
plt.scatter(df['x1'], df['y1_norm'], color='C0', label='y1_norm data')
plt.scatter(df['x2'], df['y2_norm'], color='C1', label='y2_norm data')

# Smooth x for fitted curves
x1_fit = np.linspace(df['x1'].min(), df['x1'].max(), 200)
x2_fit = np.linspace(df['x2'].min(), df['x2'].max(), 200)
plt.plot(x1_fit, quadratic(x1_fit, *popt1), color='C0', linestyle='--', label='y1_norm fit')
plt.plot(x2_fit, quadratic(x2_fit, *popt2), color='C1', linestyle='--', label='y2_norm fit')

plt.scatter([min_x1], [min_y1], color='C0', marker='x', s=80, label='y1 minimum')
plt.scatter([min_x2], [min_y2], color='C1', marker='x', s=80, label='y2 minimum')

plt.xlabel('x')
plt.ylabel('Normalised y')
plt.title('Quadratic Fit to Normalised Data')
plt.show()