import numpy as np
import pyqtgraph as pg
import plotly.graph_objects as go
from pyqtgraph.Qt import QtGui, QtWidgets
from adriq.experiment import *
from adriq.tdc_functions import filter_trailing_zeros, compute_time_diffs, filter_runs
from adriq.RedLabs_Dac import Redlabs_DAC

import csv
import os
from tqdm import tqdm
import time
import webbrowser

redlabds_dac = Client(Redlabs_DAC)


# Start timing the script
start_time = time.time()


cavity_resonance =0
# Create the dictionary of DDS instances
calib_directory = r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment Files and VIs\AOM calibration VI\Calibration_Files"

def gaussian(amplitude, mu, sigma): #Sigma is Half Width 1/e height (multiply by 2*sqrt(ln(2)) to get FWHM ~1.665)
    return lambda t: amplitude * np.exp(-((t - mu)**2) / (sigma**2))

dds_dict = load_dds_dict("ram", r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\dds_config.cfg")

pulse_sequencer = Pulse_Sequencer()

exp_sequence = Experiment_Builder(dds_dict, pulse_sequencer, ram_step=0.04, N_Cycles=5E4)

exp_sequence.set_detunings(detuning_dict={"850 SP1": 0, "850 SP2": 0, "854 SP1": 0, "854 SP2": 0, "397a": -48, "854 Cav": cavity_resonance, "397c": -18, "866": -18, "866 OP": 30, "850 RP": 0})

exp_sequence.set_trapping_parameters(
    trapping_detuning_dict={"397c": -50},
    trapping_amplitude_dict={"397c": 0.8}
)
# Create the cooling section (cool for 6 useconds)
exp_sequence.create_cooling_section(length=6, amplitude_dict={"397c": 0.3, "854 SP1": 0.1, "850 RP": 0.2})

# Create a Gaussian output function for DDS 397a

# Create op section 
exp_sequence.create_section(name="Cooling Cav", duration=12, dds_functions={
    "397c": lambda t: 0.2,
    "850 SP1" : lambda t: 0.2,
    "854 Cav": lambda t: 0.05
}, pmt_gate_high=True) #First stage of Optical Pumping

exp_sequence.build_ram_arrays()

# Plot the amplitude arrays
#exp_sequence.plot_amplitude_arrays()
exp_sequence.flash()

exp_runner = Experiment_Runner(
    dds_dict,
    pulse_sequencer,
    timeout=100,
    pmt_threshold=1500,
    expected_fluorescence= 6000,
    pulse_expected_fluorescence= 1500,
    sp_threshold=None,
    load_timeout = 50,
    cavity_lock=True
)

exp_runner.experiment_trap_depth = 1

exp_runner.measure_expected_fluorescence()

Piezo_Voltages = np.arange(-4,4.1,0.5)
Piezo_Voltages = np.arange(-5, 5.1, 0.5)
fluorescence_results = []
valid_pulses = []

file_name = file_name()

for v in tqdm(Piezo_Voltages, desc="Scanning Piezo Voltages", unit="step"):
    redlabds_dac.set_piezo_voltage(v)
    time.sleep(1)  # Allow the piezo to settle

    exp_runner.clear_channels()
    exp_runner.start_experiment(N=10)

    if exp_runner.N_Valid_Pulses == 0:
        fluorescence_rate = 0
    else:
        total_counts = exp_runner.get_counts_in_window("signal-f", lower_cutoff=6, upper_cutoff=18)
        if isinstance(total_counts, dict) and 'pmt_counts_chan' in total_counts:
            fluorescence_rate = total_counts['pmt_counts_chan'] / (exp_runner.N_Valid_Pulses*12E-6)
        else:
            fluorescence_rate = total_counts / (exp_runner.N_Valid_Pulses*12E-6)

    fluorescence_results.append(fluorescence_rate)
    valid_pulses.append(exp_runner.N_Valid_Pulses)

    print(f"Piezo Voltage: {v}, Fluorescence: {fluorescence_rate}")

    # Keep your save_data logic, but now save fluorescence_rate
    save_data(
        file_name,
        voltage=v,
        fluorescence=fluorescence_rate,
        valid_pulses=exp_runner.N_Valid_Pulses
    )

# Save fluorescence vs. voltage data to a CSV file
csv_filename = "fluorescence_vs_voltage.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Piezo Voltage (V)", "Fluorescence Rate (cps)", "Valid Pulses"])
    for voltage, fluorescence, pulses in zip(Piezo_Voltages, fluorescence_results, valid_pulses):
        writer.writerow([voltage, fluorescence, pulses])

print(f"Fluorescence vs. Voltage data saved to {csv_filename}")

# Plot fluorescence vs. voltage
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(Piezo_Voltages, fluorescence_results, marker="o", linestyle="-", color="b", label="Fluorescence Rate")
plt.title("Fluorescence vs. Piezo Voltage")
plt.xlabel("Piezo Voltage (V)")
plt.ylabel("Fluorescence Rate (cps)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("fluorescence_vs_voltage.png")
plt.show()