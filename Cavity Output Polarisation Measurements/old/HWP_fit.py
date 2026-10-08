import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load CSV
df = pd.read_csv("polariser_and_hwp_April_26.csv", sep=",")

pulse_length_s = 15e-6  # 15 microseconds

# Rates (Hz)
df["Rate_Ch3"] = df["Channel 3 Raw Counts"] / (df["Valid Runs"] * pulse_length_s)
df["Rate_Ch4"] = df["Channel 4 Raw Counts"] / (df["Valid Runs"] * pulse_length_s)

# Poisson errors -> rate errors (Hz)
df["Err_Ch3"] = np.sqrt(df["Channel 3 Raw Counts"]) / (df["Valid Runs"] * pulse_length_s)
df["Err_Ch4"] = np.sqrt(df["Channel 4 Raw Counts"]) / (df["Valid Runs"] * pulse_length_s)

def sin_func(x_deg, A, phi, offset):
    return A * np.sin(4 * np.deg2rad(x_deg) + phi) + offset

x = df["Angle (deg)"].to_numpy()

# ----- Fit Ch3 (raw) -----
y3 = df["Rate_Ch3"].to_numpy()
s3 = df["Err_Ch3"].to_numpy()
popt3, _ = curve_fit(
    sin_func, x, y3,
    sigma=s3, absolute_sigma=True,
    p0=[0.5 * (y3.max() - y3.min()), 0.0, y3.mean()],
    bounds=([0.0, -np.pi, -np.inf], [np.inf, np.pi, np.inf]),
)
fit3 = sin_func(x, *popt3)

# ----- Fit Ch4 (raw) -----
y4 = df["Rate_Ch4"].to_numpy()
s4 = df["Err_Ch4"].to_numpy()
popt4, _ = curve_fit(
    sin_func, x, y4,
    sigma=s4, absolute_sigma=True,
    p0=[0.5 * (y4.max() - y4.min()), 0.0, y4.mean()],
    bounds=([0.0, -np.pi, -np.inf], [np.inf, np.pi, np.inf]),
)
fit4 = sin_func(x, *popt4)

print("popt3:", popt3)
print("popt4:", popt4)

# Maxima positions (same formula as before)
maxima_ch3 = (np.rad2deg((np.pi / 2) - popt3[1]) % 360) / 4
maxima_ch4 = (np.rad2deg((np.pi / 2) - popt4[1]) % 360) / 4
print(f"Maxima position for Channel 3: {maxima_ch3:.2f} degrees")
print(f"Maxima position for Channel 4: {maxima_ch4:.2f} degrees")
angles_23 = [23.4, 23.5, 23.6]
angles_68 = [68.4, 68.5, 68.6]

df_angle_01 = df.copy()
df_angle_01["Angle01"] = df_angle_01["Angle (deg)"].round(1)

def mean_rates_at(angles):
    sub = df_angle_01[df_angle_01["Angle01"].isin(angles)]
    if sub.empty:
        raise ValueError(f"No rows found at angles {angles}. Check your CSV angle spacing.")
    return {
        "n_rows": len(sub),
        "ch3_mean_hz": sub["Rate_Ch3"].mean(),
        "ch4_mean_hz": sub["Rate_Ch4"].mean(),
    }

print("Mean @ 23.4/23.5/23.6:", mean_rates_at(angles_23))
print("Mean @ 68.4/68.5/68.6:", mean_rates_at(angles_68))
angles_23 = [23.4, 23.5, 23.6]
angles_68 = [68.4, 68.5, 68.6]

sub23 = df_angle_01[df_angle_01["Angle01"].isin(angles_23)]
sub68 = df_angle_01[df_angle_01["Angle01"].isin(angles_68)]

ch3_mean_23 = sub23["Rate_Ch3"].mean()
ch4_mean_23 = sub23["Rate_Ch4"].mean()
ch3_mean_68 = sub68["Rate_Ch3"].mean()
ch4_mean_68 = sub68["Rate_Ch4"].mean()
print(f"Mean @ 23.4/23.5/23.6: Ch3={ch3_mean_23:.2f} Hz, Ch4={ch4_mean_23:.2f} Hz")
print(f"Mean @ 68.4/68.5/68.6: Ch3={ch3_mean_68:.2f} Hz, Ch4={ch4_mean_68:.2f} Hz")
coupling_efficiency_ratio = ch3_mean_23 / ch4_mean_68
print(f"Coupling efficiency ratio (Ch3@23.5 / Ch4@68.5): {coupling_efficiency_ratio:.4f}")
extinction_ratio = ch3_mean_23 / (coupling_efficiency_ratio * ch4_mean_23)
print(f"Extinction ratio (Ch3@23.5 / (Coupling * Ch4@23.5)): {extinction_ratio:.4f}")
extinction_ratio = coupling_efficiency_ratio*ch4_mean_68 / ( ch4_mean_23)
print(f"Extinction ratio (Coupling * Ch4@68.5 / Ch4@23.5): {extinction_ratio:.4f}")
# ----- Plot raw rates + fits -----
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 7))

ax1.errorbar(x, y3, yerr=s3, fmt="o", capsize=3, label="Ch3 Rate")
ax1.plot(x, fit3, "--", label="Ch3 Fit")
ax1.set_ylabel("Rate (Hz)")
ax1.legend()
ax1.grid(True)

ax2.errorbar(x, y4, yerr=s4, fmt="o", capsize=3, label="Ch4 Rate")
ax2.plot(x, fit4, "--", label="Ch4 Fit")
ax2.set_xlabel("Angle (deg)")
ax2.set_ylabel("Rate (Hz)")
ax2.legend()
ax2.grid(True)

plt.suptitle("Rate vs Angle with sine fit (unnormalised)")
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
 
# Load CSV
df = pd.read_csv("HWP_Scan_June_2026.csv", sep=',')
 
# Define pulse length in seconds
pulse_length_s = 15e-6  # 15 microseconds

# Calculate rates in Hz
df['Rate_Ch3'] = df['Channel 3 Raw Counts'] / (df['Valid Runs'] * pulse_length_s)
df['Rate_Ch4'] = df['Channel 4 Raw Counts'] / (df['Valid Runs'] * pulse_length_s)

# Calculate errors assuming Poisson statistics on counts, propagate to rate
df['Err_Ch3'] = np.sqrt(df['Channel 3 Raw Counts']) / (df['Valid Runs'] * pulse_length_s)
df['Err_Ch4'] = np.sqrt(df['Channel 4 Raw Counts']) / (df['Valid Runs'] * pulse_length_s)

# Normalise rates and errors
df['Norm_Rate_Ch3'] = df['Rate_Ch3'] / df['Rate_Ch3'].max()
df['Norm_Rate_Ch4'] = df['Rate_Ch4'] / df['Rate_Ch4'].max()
df['Norm_Err_Ch3'] = df['Err_Ch3'] / df['Rate_Ch3'].max()
df['Norm_Err_Ch4'] = df['Err_Ch4'] / df['Rate_Ch4'].max()

# Sine fit function remains the same
def sin_func(x, A, phi, offset):
    return A * np.sin(4*np.deg2rad(x) + phi) + offset

# Fit Channel 3 (normalised)
popt3, _ = curve_fit(
    sin_func,
    df['Angle (deg)'],
    df['Norm_Rate_Ch3'],
    sigma=df['Norm_Err_Ch3'],
    absolute_sigma=True,
    p0=[1, 0, df['Norm_Rate_Ch3'].mean()],
    bounds=([0, -np.pi, 0], [np.inf, np.pi, np.inf])
)
fit3 = sin_func(df['Angle (deg)'], *popt3)

# Fit Channel 4 (normalised)
popt4, _ = curve_fit(
    sin_func,
    df['Angle (deg)'],
    df['Norm_Rate_Ch4'],
    sigma=df['Norm_Err_Ch4'],
    absolute_sigma=True,
    p0=[1, 0, df['Norm_Rate_Ch4'].mean()],
    bounds=([0, -np.pi, 0], [np.inf, np.pi, np.inf])
)
fit4 = sin_func(df['Angle (deg)'], *popt4)
print(popt3)
print(popt4)
a3 = np.rad2deg(0.25 * (np.arcsin((0.5 - popt3[2]) / popt3[0]) - popt3[1]))+90
a4 = np.rad2deg(0.25 * (np.arcsin((0.5 - popt4[2]) / popt4[0]) - popt4[1]))



print(popt3)
print(popt4)
print(a3)
print(a4)
print(sin_func(a3, *popt3))
print(sin_func(a4, *popt4))

# Calculate maxima positions for Channel 3
maxima_ch3 = (np.rad2deg((np.pi / 2) - popt3[1]) % 360) / 4

# Calculate maxima positions for Channel 4
maxima_ch4 = (np.rad2deg((np.pi / 2) - popt4[1]) % 360) / 4

print(f"Maxima position for Channel 3: {maxima_ch3:.2f} degrees")
print(f"Maxima position for Channel 4: {maxima_ch4:.2f} degrees")

# Plot normalised data and fits
plt.figure()
plt.errorbar(df['Angle (deg)'], df['Norm_Rate_Ch3'], yerr=df['Norm_Err_Ch3'], fmt='o', label='Channel 3 Normalised', capsize=3)
plt.errorbar(df['Angle (deg)'], df['Norm_Rate_Ch4'], yerr=df['Norm_Err_Ch4'], fmt='o', label='Channel 4 Normalised', capsize=3)
plt.plot(df['Angle (deg)'], fit3, label='Sine Fit Ch3 (Norm)', color='C0', linestyle='--')
plt.plot(df['Angle (deg)'], fit4, label='Sine Fit Ch4 (Norm)', color='C1', linestyle='--')
#plt.axvline(x=a3, color='C0', linestyle='dashed', label='a3')
#plt.axvline(x=a4, color='C1', linestyle='dashed', label='a4')
plt.xlabel("Angle (deg)")
plt.ylabel("Normalised Rate")
plt.title("Normalised Rate vs Angle with Sine Fit")
plt.legend()
plt.grid(True)
plt.show()