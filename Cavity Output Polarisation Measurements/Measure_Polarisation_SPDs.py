import time
import numpy as np
import configparser
from pathlib import Path

from adriq.Optomechanics import *
from adriq.Counters import *
from adriq.Servers import Client




# ============================================================
# Hardware
# ============================================================

polarisation_analyser = Client(Polarisation_Analyser)
qutau_reader = Client(QuTau_Reader)

samples_per_basis = 50
poll_dt = 0.05
settle_time = 2.0

ch3_name = "single_photon_chan3"
ch4_name = "single_photon_chan4"

last_time = None

# Known input Stokes vector used only to calibrate detector imbalance.
# Convention in this script: X=H-V, Y=D-A, Z=L-R.
known_stokes = {"X": 0.7861876, "Y": 0.23023728, "Z": 0.55618367}
detector_calibration = {}
count_rates = {}


# ============================================================
# Helper functions
# ============================================================

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


def measure_fraction():

    time.sleep(settle_time)

    fresh_ch3, fresh_ch4 = acquire_counts()

    f_ch3 = fresh_ch3 / ((fresh_ch3) + (fresh_ch4))
    f_ch4 = fresh_ch4 / ((fresh_ch3) + (fresh_ch4))

    return f_ch3, f_ch4, fresh_ch3, fresh_ch4


def infer_population(polarisation_analyser, basis):
    """
    setting_1 : basis state 1 -> ch3
    setting_2 : basis state 1 -> ch4
    """
    if basis == "X":
        polarisation_analyser.set_X_basis()

    elif basis == "Y":
        polarisation_analyser.set_Y_basis()
    
    elif basis == "Z":
        polarisation_analyser.set_Z_basis()

    else:
        raise ValueError(
            f"Unknown basis '{basis}'. "
            "Expected 'X', 'Y' or 'Z'."
        )
    
    f1_all, f2_all, ch3, ch4 = measure_fraction()


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
    if basis == "X":
        polarisation_analyser.set_X_basis(invert=True)

    elif basis == "Y":
        polarisation_analyser.set_Y_basis(invert=True)

    elif basis == "Z":
        polarisation_analyser.set_Z_basis(invert=True)

    f3_all, f4_all, ch3, ch4 = measure_fraction()

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
    if basis == "X":
        label_1, label_2 = "P_H", "P_V"
    elif basis == "Y":
        label_1, label_2 = "P_D", "P_A"
    elif basis == "Z":
        label_1, label_2 = "P_L", "P_R"
    print(f"{label_1} = {P1:.6f}")
    print(f"{label_2} = {P2:.6f}")
    print(f"eta_ch3 / eta_ch4 (self-inferred) = {eta_ratio:.6f}")
    print(f"eta_ch3 / eta_ch4 from known Stokes, {basis}  = {eta_setting_1:.6f}")
    print(f"eta_ch3 / eta_ch4 from known Stokes, {basis}' = {eta_setting_2:.6f}")
    print()

    return exp_val, exp_err, P1, P2, eta_ratio

print("Measuring in the X basis...")
S1,S1_err,P_H, P_V, eta_HV = infer_population(
    polarisation_analyser,
    "X"
)
print("Measuring in the Y basis...")
S2,S2_err,P_D, P_A, eta_DA = infer_population(
    polarisation_analyser,
    "Y"
)

print("Measuring in the Z basis...")
S3,S3_err,P_L, P_R, eta_LR = infer_population(
    polarisation_analyser,
    "Z"
)


stokes = np.array([S1, S2, S3])

print("========================================")
print("Normalised Stokes vector")
print("========================================")
print(f"S1 = {S1:.6f}+/- {S1_err:.6f}")
print(f"S2 = {S2:.6f}+/- {S2_err:.6f}")
print(f"S3 = {S3:.6f}+/- {S3_err:.6f}")
print()
print("S =", stokes)
print()
print(f"eta_HV = {eta_HV:.6f}")
print(f"eta_DA = {eta_DA:.6f}")
print(f"eta_LR = {eta_LR:.6f}")

print("\nKnown-Stokes detector calibration (eta_ch3 / eta_ch4)")
for basis in ("X", "Y", "Z"):
    eta_1, eta_2 = detector_calibration[basis]
    print(f"{basis}: {eta_1:.6f}    {basis}': {eta_2:.6f}")


print("\nAverage counts per setting")
labels = {"X": ("H", "V"), "Y": ("D", "A"), "Z": ("L", "R")}
 
for basis in ("X", "Y", "Z"):
    ch3, ch4 = count_rates[basis]
    ch3_i, ch4_i = count_rates[basis + "'"]
    p1, p2 = labels[basis]
 
    print(f"{p1}: ch3/ch4 = {ch3/ch4_i:.4f}")
    print(f"{p2}: ch3/ch4 = {ch3_i/ch4:.4f}")

Plot_Stokes_Vector(stokes)

settings = ["X", "X'", "Y", "Y'", "Z", "Z'"]
ch3 = [count_rates[s][0] for s in settings]
ch4 = [count_rates[s][1] for s in settings]
 
x = np.arange(len(settings))
plt.figure()
plt.bar(x - 0.2, ch3, 0.4, label="ch3")
plt.bar(x + 0.2, ch4, 0.4, label="ch4")
plt.xticks(x, settings)
plt.ylabel("Average counts")
plt.legend()
plt.show()