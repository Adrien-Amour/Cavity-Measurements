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


two_photon_resonance = -7.9
# Create the dictionary of DDS instances
calib_directory = r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment Files and VIs\AOM calibration VI\Calibration_Files"

def gaussian(amplitude, mu, sigma): #Sigma is Half Width 1/e height (multiply by 2*sqrt(ln(2)) to get FWHM ~1.665)
    return lambda t: amplitude * np.exp(-((t - mu)**2) / (sigma**2))

dds_dict = load_dds_dict("ram", r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\dds_config.cfg")

pulse_sequencer = Pulse_Sequencer()

exp_sequence = Experiment_Builder(dds_dict, pulse_sequencer, ram_step=0.1, N_Cycles=5E4)

exp_sequence.set_detunings(detuning_dict={"850 SP1": two_photon_resonance, "850 SP2": 0, "854 SP1": 0, "854 SP2": 0, "397a": -48, "854 Cav": 0, "397c": -18, "866S": -18, "866 OP": 30, "850 RP": 0})


cooling_length = 30
exp_sequence.load_cooling(length=cooling_length)
exp_sequence.load_trapping()
# Create a Gaussian output function for DDS 397a

exp_sequence.load_section("pump_to_stretch")
exp_sequence.create_section(name="wait", duration=2, dds_functions={})


# Create the single photon section

exp_sequence.create_section(name="Single Photon1", duration=10, dds_functions={
    "850 SP1": gaussian(amplitude=0.028, mu=4, sigma=1),
}, pmt_gate_high=False,coincidence_detector=True, coincidence_low=1, coincidence_high=1)
exp_sequence.create_section(name="wait2", duration=40, dds_functions={})
exp_sequence.create_section(name="Single Photon2", duration=10, dds_functions={
    "850 SP1": gaussian(amplitude=0.25, mu=4, sigma=1),
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


exp_runner.clear_channels()
exp_runner.set_n_photon_detection(n_photons=1, lower_cutoff=0)
exp_runner.start_experiment(N=10)

total_counts = exp_runner.plot_time_diffs_histogram("signal-sp", lower_cutoff=45, upper_cutoff = 120, n_bins=100)

photon_times = np.array(exp_runner.photon_times)
photon_channels = np.array(exp_runner.photon_channels)


channels = np.unique(photon_channels)

fig, axes = plt.subplots(
    len(channels), 1,
    sharex=True,
    figsize=(9, 3 * len(channels)),
    squeeze=False
)

for axis, channel in zip(axes.ravel(), channels):
    axis.hist(
        photon_times[photon_channels == channel],
        bins=200,
        alpha=0.8
    )
    axis.set_title(f"Channel {channel}")
    axis.set_ylabel("Counts")
    axis.grid(alpha=0.3)

axes[-1, 0].set_xlabel("Arrival time difference (us)")
plt.tight_layout()
plt.show()