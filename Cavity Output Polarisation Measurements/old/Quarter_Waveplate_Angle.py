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

qwp_mount = Client(Standa_Motorised_WaveplateMount)


# Define the sinusoidal function with a peak at 80 and trough at 165
# We'll use a sine wave that's shifted and scaled to achieve this behavior

# Construct the sinusoidal function: sin-like shape peaking at 80, trough at 165
def custom_wave(x):
    # We want a peak at 80 and a trough at 165, so adjust the frequency and phase accordingly
    # Approximate the distance between peak and trough as half-period: 165 - 80 = 85
    # Full period is 170, so frequency = 2π / 170
    # Shift the sine wave so the peak aligns at x = 80
    return np.sin(2 * np.pi * (x - 40) / 180)

# Recalculate the spacing based on the derivative of this new wave
def spacing_from_derivative(x):
    # Derivative of sin is cos, lower values of |cos| (near peak/trough) → smaller spacing
    return 1 / (np.abs(np.cos(2 * np.pi * (x - 40) / 180)) + 0.1)

# Generate base uniform grid
x_uniform = np.linspace(0, 360, 10000)
spacing_values = spacing_from_derivative(x_uniform)

# Create non-uniform sampling based on derivative
cumulative_spacing = np.cumsum(spacing_values)
cumulative_spacing = (cumulative_spacing - cumulative_spacing.min()) / (cumulative_spacing.max() - cumulative_spacing.min())

# Interpolate to get non-uniform x values
n_points = 50
uniform_indices = np.linspace(0, 1, n_points)
custom_x = np.interp(uniform_indices, cumulative_spacing, x_uniform)
angles = custom_x
# Calculate y values for the custom wave
y_values = custom_wave(custom_x)

angles = np.arange(0, 361, 5, dtype=float)

# Plot the custom wave using scatter
# plt.figure(figsize=(10, 5))
# plt.scatter(custom_x, y_values, s=10, color='green', label='Custom sin-like wave')
# plt.title("Scatter Plot of Custom sin(x) with Peak at 80 and Trough at 165")
# plt.xlabel("x (degrees)")
# plt.ylabel("Custom sin(x)")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

#custom_x[:10]  # Display first few x values as output example



# Start timing the script
start_time = time.time()


two_photon_resonance = -7.8
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
    trapping_amplitude_dict={"397c": 0.8}
)
# Create the cooling section (cool for 6 useconds)
exp_sequence.create_cooling_section(length=6, amplitude_dict={"397c": 0.3, "854 SP1": 0.15, "850 RP": 0.2})

# Create a Gaussian output function for DDS 397a

# Create op section 
exp_sequence.create_section(name="Optical Pumping", duration=12, dds_functions={
    "397c": lambda t: 0.2,
    "866 OP": lambda t: 0.05,
    "854 SP1": lambda t: 0.2
}, pmt_gate_high=True) #First stage of Optical Pumping

exp_sequence.create_section(name="wait", duration=2, dds_functions={})


# Create the single photon section

exp_sequence.create_section(name="Single Photon", duration=16, dds_functions={
    "850 SP1": gaussian(amplitude=0.1, mu=8, sigma=2.5),
}, pmt_gate_high=False)

exp_sequence.build_ram_arrays()

# Plot the amplitude arrays
#exp_sequence.plot_amplitude_arrays()
exp_sequence.flash()

exp_runner = Experiment_Runner(
    dds_dict,
    pulse_sequencer,
    timeout=100,
    pmt_threshold=1100,
    expected_fluorescence= 6000,
    pulse_expected_fluorescence= 1100,
    sp_threshold=None,
    load_timeout = 50,
    cavity_lock=True
)

exp_runner.experiment_trap_depth = 1

exp_runner.measure_expected_fluorescence()


# Initialize required lists
efficiencies = []
efficiencies_2 = []
detunings = []
efficiency_results = []

# Perform the experiment and collect fluorescence data
efficiency_errors = []
efficiency_errors_2 = []



# Before the loop, initialize lists to store raw data
raw_counts_channel_3 = []
raw_counts_channel_4 = []
valid_runs = []
angles_recorded = []


raw_counts_csv = "QWP_full_final.csv"

with open(raw_counts_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Angle (deg)", "Channel 3 Raw Counts", "Channel 4 Raw Counts", "Valid Runs"])

    for a in tqdm(angles, desc="Scanning Angles", unit="step"):
        print(f"Setting waveplate angle to {a} degrees")
        qwp_mount.move_to(a)

        exp_runner.clear_channels()
        exp_runner.start_experiment(N=25)

        total_counts = exp_runner.get_counts_in_window("signal-sp", lower_cutoff=18.3, upper_cutoff=35)

        # Extract counts for channels
        counts_channel_4 = total_counts.get('single_photon_chan4', 0)
        counts_channel_3 = total_counts.get('single_photon_chan3', 0)

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