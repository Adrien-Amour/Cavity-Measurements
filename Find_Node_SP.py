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


two_photon_resonance = -6.1
# Create the dictionary of DDS instances
calib_directory = r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment Files and VIs\AOM calibration VI\Calibration_Files"

def gaussian(amplitude, mu, sigma): #Sigma is Half Width 1/e height (multiply by 2*sqrt(ln(2)) to get FWHM ~1.665)
    return lambda t: amplitude * np.exp(-((t - mu)**2) / (sigma**2))

dds_dict = load_dds_dict("ram", r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\dds_config.cfg")

pulse_sequencer = Pulse_Sequencer()

exp_sequence = Experiment_Builder(dds_dict, pulse_sequencer, ram_step=0.04, N_Cycles=5E4)

exp_sequence.set_detunings(detuning_dict={"850 SP1": two_photon_resonance, "850 SP2": 0, "854 SP1": 0, "854 SP2": 0, "397a": -48, "854 Cav": 0, "397c": -18, "866S": -18, "866 OP": 30, "850 RP": 0})


cooling_length = 20
exp_sequence.load_cooling(length=cooling_length)
exp_sequence.load_trapping()
# Create a Gaussian output function for DDS 397a

exp_sequence.load_section("pump_to_stretch")
exp_sequence.create_section(name="wait", duration=2, dds_functions={})


# Create the single photon section

exp_sequence.create_section(name="Single Photon", duration=12, dds_functions={
    "850 SP1": gaussian(amplitude=0.25, mu=6, sigma=1),
}, pmt_gate_high=False,coincidence_detector=True, coincidence_low=1, coincidence_high=1)


exp_sequence.build_ram_arrays()

# Plot the amplitude arrays
#exp_sequence.plot_amplitude_arrays()
exp_sequence.flash()

exp_runner = Experiment_Runner(
    dds_dict,
    pulse_sequencer,
    timeout=100,
    pmt_threshold=2500,
    expected_fluorescence= 6000,
    pulse_expected_fluorescence= 2500,
    sp_threshold=None,
    load_timeout = 100,
    cavity_lock=False
)

exp_runner.experiment_trap_depth = 0.8

exp_runner.measure_expected_fluorescence()

Piezo_Voltages = np.arange(-7,7.1, 1)
# Piezo_Voltages = np.arange(0.8,2,0.2)

# Initialize required lists
efficiencies = []
efficiencies_2 = []
detunings = []
channel_3_counts = []
channel_4_counts = []
valid_pulses = []
efficiency_results = []


# Perform the experiment and collect fluorescence data
efficiency_errors = []
efficiency_errors_2 = []

file_name = generate_file_name()
shifts = [-0.2,-0.1,0,0.1,0.2]
shifts = [0]
for v in tqdm(Piezo_Voltages, desc="Scanning Piezo Voltages", unit="step"):
    redlabds_dac.set_piezo_voltage(v,rate=1)
    time.sleep(1)  # Allow the piezo to settle
    efficiencies_v = []
    efficiencies_2_v = []
    for shift in shifts:
        if len(shifts) > 1:
            #only need to overwrite the detuning if we are doing multiple shifts
            exp_sequence.edit_detunings({"850 SP1": two_photon_resonance + shift})
            exp_sequence.build_ram_arrays()
            exp_sequence.flash()

        exp_runner.clear_channels()
        exp_runner.start_experiment(N=5)

        total_counts = exp_runner.get_counts_in_window("signal-sp", lower_cutoff=34.3, upper_cutoff=46.3)

        # Extract counts for channels
        counts_channel_4 = total_counts.get('single_photon_chan4', 0)
        counts_channel_3 = total_counts.get('single_photon_chan3', 0)

        print(f"Counts Channel 4: {counts_channel_4}, Counts Channel 3: {counts_channel_3}")

        # Calculate efficiencies
        if exp_runner.N_Valid_Pulses == 0:
            efficiency = 0
            efficiency_2 = 0
            error = 0
            error_2 = 0
        else:
            efficiencies_v.append((counts_channel_4 / exp_runner.N_Valid_Pulses) * 100)
            efficiencies_2_v.append((counts_channel_3 / exp_runner.N_Valid_Pulses) * 100)

            # Calculate standard errors
            p_4 = counts_channel_4 / exp_runner.N_Valid_Pulses
            p_3 = counts_channel_3 / exp_runner.N_Valid_Pulses
            error = (100 * (p_4 * (1 - p_4) / exp_runner.N_Valid_Pulses) ** 0.5)
            error_2 = (100 * (p_3 * (1 - p_3) / exp_runner.N_Valid_Pulses) ** 0.5)

    # Append results to lists
    efficiencies.append(max(efficiencies_v))
    efficiencies_2.append(max(efficiencies_2_v))
    efficiency_errors.append(error)
    efficiency_errors_2.append(error_2)
    channel_3_counts.append(counts_channel_3)
    channel_4_counts.append(counts_channel_4)
    total_counts
    valid_pulses.append(exp_runner.N_Valid_Pulses)
    detunings.append(v)  # Assuming detuning corresponds to piezo voltage

    print(f"Channel 3 Efficiency: {max(efficiencies_v):.2f}% ± {error_2:.2f}%, Channel 4 Efficiency: {max(efficiencies_2_v):.2f}% ± {error:.2f}%")

    save_data(
            file_name,
            voltage=v,
            channel_3_counts=counts_channel_3,
            channel_4_counts=counts_channel_4,
            valid_pulses=exp_runner.N_Valid_Pulses,
            channel_3=max(efficiencies_v),
            channel_4=max(efficiencies_2_v)
        )



# Plot fluorescence vs. voltage
plt.figure(figsize=(8, 6))
plt.plot(Piezo_Voltages, efficiencies_2, marker="o", linestyle="-", color="b", label="Fluorescence Rate")
plt.plot(Piezo_Voltages, efficiencies, marker="o", linestyle="-", color="r", label="Fluorescence Rate")

plt.xlabel("Piezo Voltage (V)")
plt.ylabel("Single Photon Efficiency")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("fluorescence_vs_voltage.png")  # Save the plot as a PNG file
plt.show()