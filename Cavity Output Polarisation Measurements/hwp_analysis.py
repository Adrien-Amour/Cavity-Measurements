import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =============================================================================
# Load data
# =============================================================================

df = pd.read_csv("HWP_Scan_axis1_June_2026.csv")

angle = df["Angle (deg)"].to_numpy()

ch3 = df["Channel 3 Avg Count Rate"].to_numpy()
ch4 = df["Channel 4 Avg Count Rate"].to_numpy()

# =============================================================================
# Physical model: HWP + PBS
# =============================================================================

def hwp_model(theta_deg, A, phi_deg, C):
    return A * np.cos(
        np.deg2rad(2 * theta_deg + phi_deg)
    )**2 + C

# =============================================================================
# Fit Channel 3
# =============================================================================

p0_3 = [
    np.max(ch3) - np.min(ch3),  # A
    0,                          # phi
    np.min(ch3)                 # C
]

popt3, pcov3 = curve_fit(
    hwp_model,
    angle,
    ch3,
    p0=p0_3
)

A3, phi3, C3 = popt3

# =============================================================================
# Fit Channel 4
# =============================================================================

p0_4 = [
    np.max(ch4) - np.min(ch4),
    90,
    np.min(ch4)
]

popt4, pcov4 = curve_fit(
    hwp_model,
    angle,
    ch4,
    p0=p0_4
)

A4, phi4, C4 = popt4

# =============================================================================
# Generate smooth fits
# =============================================================================

theta_fit = np.linspace(
    angle.min(),
    angle.max(),
    1000
)

fit3 = hwp_model(theta_fit, *popt3)
fit4 = hwp_model(theta_fit, *popt4)

# =============================================================================
# Normalise for plotting only
# =============================================================================

ch3_norm = ch3 / np.max(ch3)
ch4_norm = ch4 / np.max(ch4)

fit3_norm = fit3 / np.max(ch3)
fit4_norm = fit4 / np.max(ch4)

# =============================================================================
# Plot
# =============================================================================

plt.figure(figsize=(8,5))

plt.plot(
    angle,
    ch3_norm,
    'o',
    label='Channel 3'
)

plt.plot(
    theta_fit,
    fit3_norm,
    '-',
    linewidth=2,
    label='Channel 3 fit'
)

plt.plot(
    angle,
    ch4_norm,
    's',
    label='Channel 4'
)

plt.plot(
    theta_fit,
    fit4_norm,
    '-',
    linewidth=2,
    label='Channel 4 fit'
)

plt.xlabel("HWP angle (deg)")
plt.ylabel("Normalised count rate")
plt.title("HWP Polarisation Scan")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()

# =============================================================================
# Extinction ratios
# =============================================================================

ER3_fit = (A3 + C3) / C3
ER4_fit = (A4 + C4) / C4

ER3_data = np.max(ch3) / np.min(ch3)
ER4_data = np.max(ch4) / np.min(ch4)

print("\n==============================")
print("Channel 3")
print("==============================")
print(f"A      = {A3:.3f}")
print(f"phi    = {phi3:.3f} deg")
print(f"C      = {C3:.3f}")
print(f"ER fit = {ER3_fit:.1f}")
print(f"ER raw = {ER3_data:.1f}")

print("\n==============================")
print("Channel 4")
print("==============================")
print(f"A      = {A4:.3f}")
print(f"phi    = {phi4:.3f} deg")
print(f"C      = {C4:.3f}")
print(f"ER fit = {ER4_fit:.1f}")
print(f"ER raw = {ER4_data:.1f}")

# =============================================================================
# Visibility
# =============================================================================

V3 = A3 / (A3 + 2*C3)
V4 = A4 / (A4 + 2*C4)

print("\nVisibility")
print(f"Channel 3: {V3:.6f}")
print(f"Channel 4: {V4:.6f}")