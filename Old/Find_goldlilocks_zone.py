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


two_photon_resonance = -5.3
# Create the dictionary of DDS instances
calib_directory = r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment Files and VIs\AOM calibration VI\Calibration_Files"

def gaussian(amplitude, mu, sigma): #Sigma is Half Width 1/e height (multiply by 2*sqrt(ln(2)) to get FWHM ~1.665)
    return lambda t: amplitude * np.exp(-((t - mu)**2) / (sigma**2))

dds_dict = load_dds_dict("ram", r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\dds_config.cfg")

pulse_sequencer = Pulse_Sequencer()

exp_sequence = Experiment_Builder(dds_dict, pulse_sequencer, ram_step=0.04, N_Cycles=5E4)

exp_sequence.set_detunings(detuning_dict={"850 SP1": two_photon_resonance, "850 SP2": 0, "854 SP1": 0, "854 SP2": 0, "397a": -48, "854 Cav": 0, "397c": -18, "866": -18, "866 OP": 30, "850 RP": 0})

exp_sequence.set_trapping_parameters(
    trapping_detuning_dict={"397c": -50},
    trapping_amplitude_dict={"397c": 0.3}
)
# Create the cooling section (cool for 6 useconds)
exp_sequence.create_cooling_section(length=6, amplitude_dict={"397c": 0.07, "854 SP1": 0.1, "850 RP": 0.2})

# Create a Gaussian output function for DDS 397a

# Create op section 
exp_sequence.create_section(name="Optical Pumping", duration=12, dds_functions={
    "397c": lambda t: 0.2,
    "866 OP": lambda t: 0.04,
    "854 SP1": lambda t: 0.2
}, pmt_gate_high=True) #First stage of Optical Pumping

exp_sequence.create_section(name="wait", duration=2, dds_functions={})


# Create the single photon section

exp_sequence.create_section(name="Single Photon", duration=12, dds_functions={
    "850 SP1": gaussian(amplitude=0.15, mu=6, sigma=1),
}, pmt_gate_high=False)

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

exp_runner.experiment_trap_depth = 1.2

exp_runner.measure_expected_fluorescence()


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

file_name = file_name()


# ...existing code...

trap_depths = np.arange(0.8, 1.31, 0.1)

for depth in tqdm(trap_depths, desc="Scanning Trap Depths", unit="step"):
    exp_runner.experiment_trap_depth = depth
    time.sleep(1)  # Allow system to settle if needed

    exp_runner.clear_channels()
    exp_runner.start_experiment(N=5)

    total_counts = exp_runner.get_counts_in_window("signal-sp", lower_cutoff=20.3, upper_cutoff=32)

    counts_channel_4 = total_counts.get('single_photon_chan4', 0)
    counts_channel_3 = total_counts.get('single_photon_chan3', 0)

    print(f"Counts Channel 4: {counts_channel_4}, Counts Channel 3: {counts_channel_3}")

    if exp_runner.N_Valid_Pulses == 0:
        efficiency = 0
        efficiency_2 = 0
        error = 0
        error_2 = 0
    else:
        efficiency = (counts_channel_4 / exp_runner.N_Valid_Pulses) * 100
        efficiency_2 = (counts_channel_3 / exp_runner.N_Valid_Pulses) * 100

        p_4 = counts_channel_4 / exp_runner.N_Valid_Pulses
        p_3 = counts_channel_3 / exp_runner.N_Valid_Pulses
        error = (100 * (p_4 * (1 - p_4) / exp_runner.N_Valid_Pulses) ** 0.5)
        error_2 = (100 * (p_3 * (1 - p_3) / exp_runner.N_Valid_Pulses) ** 0.5)

    efficiencies.append(efficiency)
    efficiencies_2.append(efficiency_2)
    efficiency_errors.append(error)
    efficiency_errors_2.append(error_2)
    efficiency_results.append(efficiency)
    channel_3_counts.append(counts_channel_3)
    channel_4_counts.append(counts_channel_4)
    valid_pulses.append(exp_runner.N_Valid_Pulses)
    detunings.append(depth)  # Now detuning is trap depth

    print(f"Channel 3 Efficiency: {efficiency_2:.2f}% ± {error_2:.2f}%, Channel 4 Efficiency: {efficiency:.2f}% ± {error:.2f}%")

    save_data(
        file_name,
        voltage=depth,  # You may want to rename this argument in save_data
        channel_3_counts=counts_channel_3,
        channel_4_counts=counts_channel_4,
        valid_pulses=exp_runner.N_Valid_Pulses,
        channel_3=efficiency_2,
        channel_4=efficiency
    )

# ...existing code...