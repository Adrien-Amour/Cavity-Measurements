from adriq.Counters import *
from adriq.ad9910 import *
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
from tqdm import tqdm
import csv
from pathlib import Path
from datetime import datetime

lasers = create_laser_objects(
    r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\dds_config.cfg",
    include_lasers=[
        "854 Cav"
    ]
)
cav_laser = lasers[0]
cav_laser.apply_general_settings()
time.sleep(2.0)  # Wait for the waveplate to settle
qutau_reader = Client(QuTau_Reader)

raw_counts_csv = "HWP_Scan_axis2_June_2026.csv"

samples_per_detuning = 20
poll_dt = 0.05
settle_time = 0.2

ch3_name = "single_photon_chan3"
ch4_name = "single_photon_chan4"

detuning_list = []
ch3_avg_list = []
ch4_avg_list = []
ch3_err_list = []
ch4_err_list = []
detunings = np.arange(-2, 2.05, 0.1)
amplitude = 0.25
profile = 0
cav_laser.update_detuning(detunings[0], amplitude, profile)
qutau_reader.start_counting()

try:
    with open(raw_counts_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Detuning",
            "Channel 3 Avg Count Rate",
            "Channel 4 Avg Count Rate",
        ])

        for detuning in tqdm(detunings, desc="Scanning Detunings", unit="step"):
            cav_laser.update_detuning(detuning, amplitude,profile)
            time.sleep(settle_time)

            fresh_ch3 = []
            fresh_ch4 = []
            last_time = None

            while len(fresh_ch3) < samples_per_detuning or len(fresh_ch4) < samples_per_detuning:
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

            ch3_avg = float(np.mean(fresh_ch3[:samples_per_detuning]))
            ch4_avg = float(np.mean(fresh_ch4[:samples_per_detuning]))
            ch3_var = float(np.var(fresh_ch3[:samples_per_detuning]))
            ch4_var = float(np.var(fresh_ch4[:samples_per_detuning]))
            ch3_err = np.sqrt(ch3_var / samples_per_detuning)
            ch4_err = np.sqrt(ch4_var / samples_per_detuning)

            writer.writerow([detuning, ch3_avg, ch4_avg])
            file.flush()

            detuning_list.append(detuning)
            ch3_avg_list.append(ch3_avg)
            ch4_avg_list.append(ch4_avg)
            ch3_err_list.append(ch3_err)
            ch4_err_list.append(ch4_err)

finally:
    qutau_reader.stop_counting()

plt.figure(figsize=(8, 5))
plt.plot(detuning_list, ch3_avg_list, "o-", label="Channel 3")
plt.plot(detuning_list, ch4_avg_list, "o-", label="Channel 4")
plt.xlabel("Detuning")
plt.ylabel("Average Count Rate")
plt.title("QuTau Count Rate vs Detuning")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


total_counts = np.array(ch3_avg_list) + np.array(ch4_avg_list)
total_counts_err = np.sqrt(np.array(ch3_err_list)**2 + np.array(ch4_err_list)**2)
detuning = np.array(detuning_list)

# Voigt model
def voigt(x, A, x0, sigma, gamma):
    # sigma = Gaussian std dev
    # gamma = Lorentzian HWHM
    offset = 0 #dark counts are negligible
    return A * voigt_profile(x - x0, sigma, gamma) 

# Initial guesses
A0 = (total_counts.max() - total_counts.min()) * np.pi * 100
x00 = detuning[np.argmax(total_counts)]
sigma0 = 0.100/2  # MHz
gamma0 = 0.52/2  # kHz (250 kHz FWHM)
offset0 = total_counts.min()

p0 = [A0, 0, sigma0, gamma0]

# Bounds
bounds = (
    [0, detuning.min(), 0,   0.51/2],
    [np.inf, detuning.max(), 1, 0.53/2]
)

popt, pcov = curve_fit(
    voigt,
    detuning,
    total_counts,
    p0=p0,
    bounds=bounds,
    sigma=total_counts_err,
    absolute_sigma=True,
)

A, x0, sigma, gamma = popt

gaussian_fwhm = 2*np.sqrt(2*np.log(2))*sigma
lorentzian_fwhm = 2*gamma

voigt_fwhm = (
    0.5346 * lorentzian_fwhm
    + np.sqrt(0.2166 * lorentzian_fwhm**2 + gaussian_fwhm**2)
)
perr = np.sqrt(np.diag(pcov))
A_err, x0_err, sigma_err, gamma_err = perr

print(f"Gaussian FWHM : {gaussian_fwhm:.1f}+/-{sigma_err:.1f} MHz")
print(f"Lorentzian FWHM: {lorentzian_fwhm:.1f}+/-{gamma_err:.1f} MHz")
print(f"Voigt FWHM     : {voigt_fwhm:.1f} MHz")
print(f"Centre: {x0:.2f}+/-{x0_err:.2f} kHz")

# Plot
xfit = np.linspace(detuning.min(), detuning.max(), 1000)

plt.figure(figsize=(8,5))
plt.errorbar(
    detuning,
    total_counts,
    yerr=total_counts_err,
    fmt="o",
    capsize=3,
    label="Data"
)

plt.plot(xfit, voigt(xfit, *popt), "-", lw=2, label="Voigt fit")
plt.xlabel("Detuning (kHz)")
plt.ylabel("Total Count Rate")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()





Path("cavity_transmission_scans").mkdir(exist_ok=True)

raw_counts_csv = Path("cavity_transmission_scans") / f"cavity_transmission_scan_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv"
with open(raw_counts_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Detuning (MHz)",
        "Average Counts",
        "Average Count Error"
    ])

    writer.writerows(zip(
        detuning,
        total_counts,
        total_counts_err
    ))