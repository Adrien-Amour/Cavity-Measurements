import time
import numpy as np
import configparser
from pathlib import Path
import csv
from datetime import datetime
from pathlib import Path


from adriq.Optomechanics import *
from adriq.experiment import *
from adriq.Servers import Client




#### Single Photon Generating Sequence ####
two_photon_resonance = -5.9

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

exp_sequence.create_section(name="Single Photon Generation", duration=9, dds_functions={
    "850 SP1": gaussian(amplitude=0.3, mu=4, sigma=1),
}, pmt_gate_high=False,
coincidence_detector=True, coincidence_low=1, coincidence_high=1)
exp_sequence.build_ram_arrays()

section_len = {s["name"]: s["duration"] for s in exp_sequence.playback_sections}

# Pull required durations
pump_to_stretch_len = section_len["pump_to_stretch"]
wait_len = section_len["wait"]
single_photon_len = section_len["Single Photon Generation"]

photon_window = (
    pump_to_stretch_len + wait_len,
    pump_to_stretch_len + wait_len + 9,
)

exp_sequence.flash()




exp_runner = Experiment_Runner(
    dds_dict,
    pulse_sequencer,
    timeout=100,
    pmt_threshold=2500,
    expected_fluorescence= 6000,
    pulse_expected_fluorescence= 1500,
    sp_threshold=None,
    load_timeout = 100,
    cavity_lock=False,
    trigger_mode="ram",
)

polarisation_analyser = Client(Polarisation_Analyser)



# 1. Run the experimental loops to get raw data structures
raw_data = run_online_tomography(exp_runner, polarisation_analyser, photon_window=photon_window, samples_per_basis=5)

# 2. Save it securely to disk
save_raw_tomography_csv("stretch_state_photon.csv", raw_data)

# 3. Analyze right away with zero online post-selection filtering
window = (4.5,9)
stokes, errors, etas, _ = perform_tomography_analysis(raw_data, post_select_window=window)
plot_basis_histograms(raw_data, basis="X")
plot_basis_histograms(raw_data, basis="Y")
plot_basis_histograms(raw_data, basis="Z")

print("Stokes vector:", stokes)
print("Errors:", errors)
print("Efficiencies:", etas)

Plot_Stokes_Vector(stokes)
Plot_Density_Matrix(stokes)
