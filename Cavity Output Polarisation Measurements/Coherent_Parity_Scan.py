import time
import numpy as np
import configparser
from pathlib import Path

from adriq.Optomechanics import *
from adriq.Counters import *
from adriq.Servers import Client
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Helper functions
# ============================================================
samples_per_basis = 50
poll_dt = 0.05
settle_time = 2.0

ch3_name = "single_photon_chan3"
ch4_name = "single_photon_chan4"

last_time = None


def acquire_counts(samples=samples_per_basis):
    global last_time

    fresh_ch3 = []
    fresh_ch4 = []

    while len(fresh_ch3) < samples or len(fresh_ch4) < samples:

        times, counts = qutau_reader.get_counts()

        if not times:
            time.sleep(poll_dt)
            continue

        current_time = times[-1]

        if current_time == last_time:
            time.sleep(poll_dt)
            continue

        last_time = current_time

        ch3_hist = counts.get(ch3_name, [])
        ch4_hist = counts.get(ch4_name, [])

        if ch3_hist:
            fresh_ch3.append(ch3_hist[-1])

        if ch4_hist:
            fresh_ch4.append(ch4_hist[-1])

        time.sleep(poll_dt)


    return np.array(fresh_ch3), np.array(fresh_ch4)


def measure_fraction(samples=samples_per_basis):

    time.sleep(settle_time)

    fresh_ch3, fresh_ch4 = acquire_counts(samples)

    f_ch3 = fresh_ch3 / ((fresh_ch3) + (fresh_ch4))
    f_ch4 = fresh_ch4 / ((fresh_ch3) + (fresh_ch4))

    return f_ch3, f_ch4, fresh_ch3, fresh_ch4



def infer_population(polarisation_analyser, phi, samples=samples_per_basis):
    """
    setting_1 : basis state 1 -> ch3
    setting_2 : basis state 1 -> ch4    
    """
    polarisation_analyser.set_equator_basis(phi, invert=False)
    
    f1_all, f2_all, ch3, ch4 = measure_fraction(samples)


    f1 = np.mean(f1_all)
    f2 = np.mean(f2_all)
    f1_err = np.std(f1_all) / np.sqrt(np.size(f1_all))
    f2_err = np.std(f2_all) / np.sqrt(np.size(f2_all))


    print(f1_all, f2_all)
    print(
        f"Measurement 1:"
        f" f_ch3={f1:.6f}+/-{f1_err:.6f},"
        f" f_ch4={f2:.6f}+/-{f2_err:.6f},"
        f" ch3_avg={np.mean(ch3):.2f},"
        f" ch4_avg={np.mean(ch4):.2f}"
    )
    polarisation_analyser.set_equator_basis(phi, invert=True)

    f3_all, f4_all, ch3, ch4 = measure_fraction(samples)

    f3 = np.mean(f3_all)
    f4 = np.mean(f4_all)
    f3_err = np.std(f3_all) / np.sqrt(np.size(f3_all))
    f4_err = np.std(f4_all) / np.sqrt(np.size(f4_all))



    print(
        f"Measurement 2:"
        f" f_ch3={f3:.6f}+/-{f3_err:.6f},"
        f" f_ch4={f4:.6f}+/-{f4_err:.6f},"
        f" ch3_avg={np.mean(ch3):.2f},"
        f" ch4_avg={np.mean(ch4):.2f}"
    )

    population_ratio = np.sqrt((f1 * f4) / (f2 * f3))
    population_ratio_err = 0.5 * population_ratio * np.sqrt(f1_err**2 / (f1**2*f4**2) + f4_err**2 / (f4**2*f1**2))
    eta_ratio = np.sqrt((f1 * f3) / (f2 * f4))

    P1 = population_ratio / (1 + population_ratio)
    P2 = 1 / (1 + population_ratio)

    # Mapped R and R_err directly to population_ratio variables to avoid NameErrors
    R = population_ratio
    R_err = population_ratio_err

    print(f"\033[96mPopulation Ratio R: {R:.4f} +/- {R_err:.4f}\033[0m")
    exp_val = (R - 1) / (R + 1) if (R + 1) > 0 else 0.0
    exp_err = (2 * R_err / (R + 1)**2) if (R + 1) > 0 else 0.0
    print(f"P_0 = {P1:.6f}")
    print(f"P_1 = {P2:.6f}")
    print(f"eta_ch3 / eta_ch4 (self-inferred) = {eta_ratio:.6f}")

    print()

    return exp_val, exp_err, P1, P2, eta_ratio



phi_values = np.arange(-22.5, 202.5, 22.5)
samples_per_phi = 20

parity_values = []
parity_errors = []
eta_ratios = []

polarisation_analyser = Client(Polarisation_Analyser)
qutau_reader = Client(QuTau_Reader)
polarisation_analyser.set_equator_basis(40, invert=False)

# for phi in phi_values:
#     exp_val, exp_err, P1, P2, eta_ratio = infer_population(polarisation_analyser, phi)

#     parity_values.append(exp_val)
#     parity_errors.append(exp_err)
#     eta_ratios.append(eta_ratio)
# print("Phi values:", phi_values)
# print("Eta ratios:", eta_ratios)


# plt.figure(figsize=(8, 5))

# plt.errorbar(
#     phi_values,
#     parity_values,
#     yerr=parity_errors,
#     fmt="o-",
#     capsize=4,
# )

# plt.axhline(0, color="black", linewidth=0.8)
# plt.xlabel("phi (degrees)")
# plt.ylabel("P0 - P1")
# plt.title("Parity versus phi")
# plt.xticks(phi_values)
# plt.grid(alpha=0.3)
# plt.tight_layout()

# from scipy.optimize import curve_fit

# def sinusoid(phi, A, phase):
#     return A * np.sin(2*np.pi*phi/360 + phase)

# popt, _ = curve_fit(sinusoid, phi_values, parity_values, sigma=parity_errors)

# phi_fit = np.linspace(0, 360, 500)

# plt.plot(phi_fit, sinusoid(phi_fit, *popt), "--", label="Sinusoidal fit")

# # Zero crossings in degrees
# phi_zeros = np.array([
#     (-popt[2]) * 360 / (2*np.pi),
#     (np.pi - popt[2]) * 360 / (2*np.pi)
# ]) % 360

# # Pick the one between 0 and 180
# phi_zero = phi_zeros[(phi_zeros >= 0) & (phi_zeros <= 180)][0]

# print(phi_zero)
phi_zero = 73
eta_ratios_at_phi_zero = []
exp_vals_at_phi_zero = []
for _ in range(10):
    exp_val, exp_err, P1, P2, eta_ratio = infer_population(polarisation_analyser, phi_zero, samples=30)
    eta_ratios_at_phi_zero.append(eta_ratio)
    exp_vals_at_phi_zero.append(exp_val)

print(eta_ratios_at_phi_zero)
print(np.mean(eta_ratios_at_phi_zero), np.std(eta_ratios_at_phi_zero))
print(exp_vals_at_phi_zero)
print(np.mean(exp_vals_at_phi_zero), np.std(exp_vals_at_phi_zero))
plt.show()
