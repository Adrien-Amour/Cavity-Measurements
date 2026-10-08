import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Load data
# =============================================================================

filename = "HWP_Scan_Circular_June_2026.csv"

df = pd.read_csv(filename)

print(df.columns)

# Adjust these names if necessary
angle = df["Angle (deg)"]
ch3 = df["Channel 3 Avg Count Rate"]
ch4 = df["Channel 4 Avg Count Rate"]

# =============================================================================
# Error bars
# =============================================================================
#
# Assumptions:
# - Each saved point is the average of 20 measurements
# - Each measurement integrates for 0.2 s
# - Counts are Poisson distributed
#
# If the recorded quantity is a count rate R:
#
# counts per shot = R * 0.2
#
# total counts accumulated across all 20 shots:
#
# N_tot = 20 * R * 0.2 = 4R
#
# uncertainty on the mean rate:
#
# sigma_R = sqrt(N_tot)/(20*0.2)
#         = sqrt(4R)/4
#         = sqrt(R)/2
#
# =============================================================================

sigma_ch3 = np.sqrt(ch3) / 2
sigma_ch4 = np.sqrt(ch4) / 2

# =============================================================================
# Ratio and propagated uncertainty
# =============================================================================

ratio = ch3 / ch4

sigma_ratio = ratio * np.sqrt(
    (sigma_ch3 / ch3) ** 2 +
    (sigma_ch4 / ch4) ** 2
)

# =============================================================================
# Plot channel 3
# =============================================================================

plt.figure(figsize=(6,4))
plt.errorbar(
    angle,
    ch3,
    yerr=sigma_ch3,
    fmt='o-',
    capsize=3
)
plt.xlabel("Angle (deg)")
plt.ylabel("Channel 3 rate (s$^{-1}$)")
plt.title("Channel 3")
plt.grid(True)
plt.tight_layout()

# =============================================================================
# Plot channel 4
# =============================================================================

plt.figure(figsize=(6,4))
plt.errorbar(
    angle,
    ch4,
    yerr=sigma_ch4,
    fmt='o-',
    capsize=3
)
plt.xlabel("Angle (deg)")
plt.ylabel("Channel 4 rate (s$^{-1}$)")
plt.title("Channel 4")
plt.grid(True)
plt.tight_layout()

# =============================================================================
# Plot ratio
# =============================================================================

plt.figure(figsize=(6,4))
plt.errorbar(
    angle,
    ratio,
    yerr=sigma_ratio,
    fmt='o-',
    capsize=3
)
plt.xlabel("Angle (deg)")
plt.ylabel("Channel 3 / Channel 4")
plt.title("Rate ratio")
plt.grid(True)
plt.tight_layout()

# =============================================================================
# Fidelity analysis
# =============================================================================

R_max = ratio.max()
R_min = ratio.min()

r = R_max / R_min

# Formula from your notes
sin_delta = (1 - np.sqrt(r)) / (1 + np.sqrt(r))

delta = np.arcsin(sin_delta)

fidelity = 0.5 * (1 + np.cos(delta))

print()
print("===== Fidelity analysis =====")
print(f"R_max      = {R_max:.8f}")
print(f"R_min      = {R_min:.8f}")
print(f"r          = {r:.8f}")
print(f"sin(delta) = {sin_delta:.8e}")
print(f"delta      = {delta:.8e} rad")
print(f"delta      = {np.degrees(delta):.6f} deg")
print(f"Fidelity   = {fidelity:.10f}")

plt.show()



from scipy.optimize import curve_fit

theta0 = 24.2  # deg offset

def model(theta, eta_ratio, sin_delta):
    x = np.deg2rad(4 * (theta - theta0))
    return eta_ratio * (1 - sin_delta * np.cos(x)) / (1 + sin_delta * np.cos(x))

popt, pcov = curve_fit(model, angle, ratio, p0=[1.0, 0.1], sigma=sigma_ratio, absolute_sigma=True)

eta_ratio_fit, sin_delta_fit = popt
eta_err, sin_delta_err = np.sqrt(np.diag(pcov))

delta_fit = np.arcsin(sin_delta_fit)

print("eta_H/eta_V =", eta_ratio_fit, "+/-", eta_err)
print("delta (deg) =", np.degrees(delta_fit), "+/-", np.degrees(sin_delta_err/np.sqrt(1-sin_delta_fit**2)))
print(f"Fideltiy = {0.5 * (1 + np.cos(delta_fit)):.10f}")

theta_fit = np.linspace(angle.min(), angle.max(), 500)
ratio_fit = model(theta_fit, *popt)

plt.figure(figsize=(6,4))

plt.errorbar(
    angle,
    ratio/eta_ratio_fit,
    yerr=sigma_ratio,
    fmt='o',
    capsize=3,
    label="data"
)

plt.plot(
    theta_fit,
    ratio_fit/eta_ratio_fit,
    '-',
    label="fit"
)

plt.xlabel("Angle (deg)")
plt.ylabel("I_H / I_V")
plt.title("Fit to polarisation model")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

