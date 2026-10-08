import numpy as np
import pyqtgraph as pg
import plotly.graph_objects as go
from pyqtgraph.Qt import QtGui, QtWidgets
from adriq.experiment import *
from adriq.tdc_functions import filter_trailing_zeros, compute_time_diffs, filter_runs
import csv
import os
from tqdm import tqdm
import time
import webbrowser
from adriq.WM_SCL import *  # for shift_866

# Start timing the script
start_time = time.time()

# Create the dictionary of DDS instances
calib_directory = r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment Files and VIs\AOM calibration VI\Calibration_Files"

def gaussian(amplitude, mu, sigma):  # Sigma is Half Width 1/e height (FWHM ~ 1.665*sigma)
    return lambda t: amplitude * np.exp(-((t - mu) ** 2) / (sigma ** 2))


dds_dict = load_dds_dict(
    "ram", r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\dds_config.cfg"
)

pulse_sequencer = Pulse_Sequencer()

# Mirror Find_Resonance timing and cycle count
exp_sequence = Experiment_Builder(dds_dict, pulse_sequencer, ram_step=0.04, N_Cycles=5e4)

# Same base detunings as Find_Resonance (we scan 866 shift instead)
exp_sequence.set_detunings(
    detuning_dict={
        "850 SP1": -5.3,
        "850 SP2": 0,
        "854 SP1": 5,
        "854 SP2": 0,
        "397a": -48,
        "854 Cav": 0,
        "397c": -18,
        "866": -18,
        "866 OP": 30,
        "850 RP": 0,
    }
)
exp_sequence.set_trapping_parameters(
    trapping_detuning_dict={"397c": -50}, trapping_amplitude_dict={"397c": 0.4}
)

# Cooling section (6 us)
exp_sequence.load_cooling_section(length=6)

# Optical pumping
exp_sequence.create_section(
    name="Optical Pumping",
    duration=12,
    dds_functions={
        "397c": lambda t: 0.1,
        "866 OP": lambda t: 0.07,
        "854 SP1": lambda t: 0.6,
    },
    pmt_gate_high=True,
)

# Wait gap
exp_sequence.create_section(name="wait", duration=2, dds_functions={})

# Single photon section (with coincidence detector like in Find_Resonance)
exp_sequence.create_section(
    name="Single Photon",
    duration=12,
    dds_functions={
        "850 SP1": gaussian(amplitude=0.15, mu=6, sigma=1),
    },
    pmt_gate_high=False,
    coincidence_detector=True,
    coincidence_low=1,
    coincidence_high=100,
)

exp_sequence.build_ram_arrays()
# exp_sequence.plot_amplitude_arrays()
exp_sequence.flash()

exp_runner = Experiment_Runner(
    dds_dict,
    pulse_sequencer,
    timeout=100,
    pmt_threshold=1800,
    expected_fluorescence=8000,
    pulse_expected_fluorescence=1800,
    sp_threshold=None,
    load_timeout=50,
    cavity_lock=True,
)

length = sum(section["duration"] for section in exp_sequence.playback_sections)

# Plot
fig = go.Figure()
html_file = "efficiency_vs_866_shift.html"
fig.write_html(html_file)
webbrowser.open(html_file)

# Use the same amplitude convention as your Find_Resonance script
amplitude_values = np.array([0.18])

# Define the 866 shift scan range
# Examples:
# shifty_values = np.arange(-0.01, 0.0101, 0.002)
shifty_values = np.arange(-0.005, 0.0051, 0.0005)

# Root filename for saving data
f = file_name()

# Prepare plot traces per amplitude
for amplitude in amplitude_values:
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name=f"Channel 4 Amplitude {amplitude}"))
    fig.add_trace(
        go.Scatter(
            x=[], y=[], mode="lines", name=f"Channel 3 Amplitude {amplitude}", line=dict(dash="dash")
        )
    )

for amplitude in tqdm(amplitude_values, desc="Amplitude Progress"):
    efficiencies = []
    efficiencies_2 = []
    exp_runner.experiment_trap_depth = 1.2

    current_shift = 0  # track cumulative 866 shift

    for shifty in tqdm(shifty_values, desc=f"Processing amplitude={amplitude}", leave=False):
        # Apply incremental shift to 866
        shift_increment = shifty - current_shift
        shift_866(shift_increment)
        current_shift = shifty



        exp_runner.measure_expected_fluorescence()

        exp_runner.clear_channels()
        exp_runner.start_experiment(N=3)

        total_counts = exp_runner.get_counts_in_window("signal-sp", lower_cutoff=20.3, upper_cutoff=32)

        # Compute efficiencies
        counts_channel_4 = total_counts.get("single_photon_chan4", 0)
        counts_channel_3 = total_counts.get("single_photon_chan3", 0)

        with open("counts_log.csv", mode="a", newline="") as logf:
            writer = csv.writer(logf)
            writer.writerow([
                counts_channel_3,
                counts_channel_4,
            ])

        if exp_runner.N_Valid_Pulses == 0:
            efficiency = 0
            efficiency_2 = 0
        else:
            efficiency = (counts_channel_4 / exp_runner.N_Valid_Pulses) * 100
            efficiency_2 = (counts_channel_3 / exp_runner.N_Valid_Pulses) * 100
        efficiencies.append(efficiency)
        efficiencies_2.append(efficiency_2)

        # Update plots
        idx = amplitude_values.tolist().index(amplitude)
        fig.data[idx * 2].update(x=shifty_values[: len(efficiencies)], y=efficiencies)
        fig.data[idx * 2 + 1].update(x=shifty_values[: len(efficiencies_2)], y=efficiencies_2)

        fig.update_layout(
            title=f"Efficiency vs. 866 Shift",
            xaxis_title="866 Shift",
            yaxis_title="Efficiency (%)",
            legend_title="Channels",
            template="plotly_white",
        )

        fig.write_html(html_file)

        # Save per-step data
        save_data(
            f,
            amplitude=amplitude,
            shift_866=shifty,
            counts_channel_3=counts_channel_3,
            counts_channel_4=counts_channel_4,
            valid_pulses=exp_runner.N_Valid_Pulses,
            efficiency_2=efficiency_2,
            efficiency=efficiency,
        )

# Save shift and efficiency of each channel to a CSV file
csv_file = "shift_866_efficiency.csv"
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["866 Shift", "Channel 3 Efficiency", "Channel 4 Efficiency"])
    for shifty, eff_3, eff_4 in zip(shifty_values, efficiencies_2, efficiencies):
        writer.writerow([shifty, eff_3, eff_4])

print(efficiencies)
print(efficiencies_2)

# End timing the script
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
print(f"Elapsed time: {elapsed_time / 60:.2f} minutes")
