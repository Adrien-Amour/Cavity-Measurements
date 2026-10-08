import numpy as np
import pyqtgraph as pg
import plotly.graph_objects as go
from pyqtgraph.Qt import QtGui, QtWidgets
from adriq.experiment import *
from adriq.tdc_functions import filter_trailing_zeros, compute_time_diffs, filter_runs
from adriq.Optomechanics import *
import csv
import os
from tqdm import tqdm
import time
import webbrowser

dds_dict = load_dds_dict("singletone", r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\dds_config.cfg")
dds_dict = {key: value for key, value in dds_dict.items() if key == "854 Cav"}

# Create a Pulse_Sequencer instance
pulse_sequencer = Pulse_Sequencer(port="COM5", ps_end_pin=2, pmt_gate_pin=1, ps_sync_pin=0)

# Create an Experiment_Builder instance
experiment_builder = Experiment_Builder_Singletone(dds_dict, pulse_sequencer, N_Cycles=1E4)

# Set trapping parameters
experiment_builder.set_trapping_parameters(
    trapping_detuning_dict={},
    trapping_amplitude_dict={}
)

exp_runner = Experiment_Runner(
    dds_dict,
    pulse_sequencer,
    timeout=100,
    pmt_threshold=None,
    expected_fluorescence=None,
    pulse_expected_fluorescence=0,
    sp_threshold=None,
    load_timeout=50
)

hwp_mount = Client(Thorlabs_Motorised_WaveplateMount)
qwp_mount = Client(Standa_Motorised_WaveplateMount)
# Create sections
experiment_builder.create_section(
    name="Inject",
    duration=10,
    detunings={"854 Cav": 0},
    amplitudes={"854 Cav":1},
    pmt_gate_high=True
)

experiment_builder.create_section(
    name="Decay",
    duration=5,
    detunings={"854 Cav": 0},
    amplitudes={"854 Cav": 1},
    pmt_gate_high=True
)

# coarse = np.arange(20, 70.1, 1.0)

# fine1 = np.arange(23.5 - 0.5, 23.5 + 0.5 + 1e-9, 0.1)  # 23.0 → 24.0
# fine2 = np.arange(68.5 - 0.5, 68.5 + 0.5 + 1e-9, 0.1)  # 68.0 → 69.0
angles = np.arange(0, 120.1, 5)+4.2

# angles = np.sort(np.unique(np.concatenate([coarse, fine1, fine2])))
experiment_builder.flash()

raw_counts_channel_3 = []
raw_counts_channel_4 = []
valid_runs = []
angles_recorded = []

qwp_mount.move_to(45.9)

raw_counts_csv = "polariser_qwp_middle_45.9_hwp_scan_June_26.csv"

with open(raw_counts_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Angle (deg)", "Channel 3 Raw Counts", "Channel 4 Raw Counts", "Valid Runs"])

    for a in tqdm(angles, desc="Scanning Angles", unit="step"):
        print(f"Setting waveplate angle to {a} degrees")
        hwp_mount.move_to(a)
        exp_runner.clear_channels()
        exp_runner.start_experiment(N=100)

        time_diffs = exp_runner.get_counts_in_window("signal-sp", lower_cutoff=0, upper_cutoff=15)
        counts_channel_3 = time_diffs.get('single_photon_chan3', 0)
        counts_channel_4 = time_diffs.get('single_photon_chan4', 0)
        # Store raw counts and valid runs
        raw_counts_channel_3.append(counts_channel_3)
        raw_counts_channel_4.append(counts_channel_4)
        valid_runs.append(exp_runner.N_Valid_Pulses)
        angles_recorded.append(a)

        # Write the row immediately
        writer.writerow([a, counts_channel_3, counts_channel_4, exp_runner.N_Valid_Pulses])
        file.flush()  # Ensure data is written to disk

        print(f"Counts Channel 4: {counts_channel_4}, Counts Channel 3: {counts_channel_3}")

print(f"Raw counts and valid runs data saved to {raw_counts_csv}")


qwp_mount.move_to(0.9)

raw_counts_csv = "polariser_qwp_axis_1_0.9_hwp_scan_June_26.csv"

with open(raw_counts_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Angle (deg)", "Channel 3 Raw Counts", "Channel 4 Raw Counts", "Valid Runs"])

    for a in tqdm(angles, desc="Scanning Angles", unit="step"):
        print(f"Setting waveplate angle to {a} degrees")
        hwp_mount.move_to(a)
        exp_runner.clear_channels()
        exp_runner.start_experiment(N=10)

        time_diffs = exp_runner.get_counts_in_window("signal-sp", lower_cutoff=0, upper_cutoff=15)
        counts_channel_3 = time_diffs.get('single_photon_chan3', 0)
        counts_channel_4 = time_diffs.get('single_photon_chan4', 0)
        # Store raw counts and valid runs
        raw_counts_channel_3.append(counts_channel_3)
        raw_counts_channel_4.append(counts_channel_4)
        valid_runs.append(exp_runner.N_Valid_Pulses)
        angles_recorded.append(a)

        # Write the row immediately
        writer.writerow([a, counts_channel_3, counts_channel_4, exp_runner.N_Valid_Pulses])
        file.flush()  # Ensure data is written to disk

        print(f"Counts Channel 4: {counts_channel_4}, Counts Channel 3: {counts_channel_3}")

print(f"Raw counts and valid runs data saved to {raw_counts_csv}")


qwp_mount.move_to(90.9)
hwp_mount.move_to(angles[0])  # Reset HWP to the first angle
time.sleep(4)
raw_counts_csv = "polariser_qwp_axis_2_90.9_hwp_scan_June_26.csv"

with open(raw_counts_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Angle (deg)", "Channel 3 Raw Counts", "Channel 4 Raw Counts", "Valid Runs"])

    for a in tqdm(angles, desc="Scanning Angles", unit="step"):
        print(f"Setting waveplate angle to {a} degrees")
        hwp_mount.move_to(a)
        exp_runner.clear_channels()
        exp_runner.start_experiment(N=10)

        time_diffs = exp_runner.get_counts_in_window("signal-sp", lower_cutoff=0, upper_cutoff=15)
        counts_channel_3 = time_diffs.get('single_photon_chan3', 0)
        counts_channel_4 = time_diffs.get('single_photon_chan4', 0)
        # Store raw counts and valid runs
        raw_counts_channel_3.append(counts_channel_3)
        raw_counts_channel_4.append(counts_channel_4)
        valid_runs.append(exp_runner.N_Valid_Pulses)
        angles_recorded.append(a)

        # Write the row immediately
        writer.writerow([a, counts_channel_3, counts_channel_4, exp_runner.N_Valid_Pulses])
        file.flush()  # Ensure data is written to disk

        print(f"Counts Channel 4: {counts_channel_4}, Counts Channel 3: {counts_channel_3}")

print(f"Raw counts and valid runs data saved to {raw_counts_csv}")

