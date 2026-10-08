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

def Tomography(Resonance):
    def gaussian(amplitude, mu, sigma): #Sigma is Half Width 1/e height (multiply by 2*sqrt(ln(2)) to get FWHM ~1.665)
        return lambda t: amplitude * np.exp(-((t - mu)**2) / (sigma**2))
    single_photon_len = 10

    dds_dict = load_dds_dict("ram", r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\dds_config.cfg")

    pulse_sequencer = Pulse_Sequencer()




    exp_sequence = Experiment_Builder(dds_dict, pulse_sequencer, ram_step=0.04, N_Cycles=5E4)

    exp_sequence.set_detunings(detuning_dict={"850 SP1": Resonance, "850 SP2": 0, "854 SP1": 0, "854 SP2": 0, "397a": -48, "854 Cav": 0, "397c": -18, "866S": -18, "866 OP": 30, "850 RP": 0})

    cooling_length = 20
    exp_sequence.load_cooling(length=cooling_length)
    exp_sequence.load_trapping()
    # Create a Gaussian output function for DDS 397a

    exp_sequence.load_section("pump_to_stretch")
    exp_sequence.load_section("pump_to_ground")
    exp_sequence.load_section("STIRAP")
    exp_sequence.load_section("quench_ground")
    prep_section_names = ["pump_to_stretch", "pump_to_ground", "STIRAP", "quench_ground"]

    prep_len = sum(
    s["duration"]
    for s in exp_sequence.playback_sections
    if s["name"] in prep_section_names
    )

    # Create the single photon section

    exp_sequence.create_section(name="Single Photon Generation", duration=single_photon_len, dds_functions={
        "850 SP1": gaussian(amplitude=0.25, mu=4, sigma=1),
    }, pmt_gate_high=False,
    coincidence_detector=True, coincidence_low=1, coincidence_high=1)
    exp_sequence.build_ram_arrays()

    lower_cutoff = prep_len + 0.3
    upper_cutoff = lower_cutoff + single_photon_len
    photon_window = (
        lower_cutoff,
        upper_cutoff
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
    raw_data = run_online_tomography(exp_runner, polarisation_analyser, photon_window=photon_window, samples_per_basis=50)
    return raw_data



plus_resonance = -26.0
raw_data_plus = Tomography(plus_resonance)
# 2. Save it securely to disk
save_raw_tomography_csv("plus_resonance_photon.csv", raw_data_plus)




# 3. Analyze right away with zero online post-selection filtering
window = (1.5,6.5)

minus_resonance = 24.15
raw_data_minus = Tomography(minus_resonance)
save_raw_tomography_csv("minus_resonance_photon.csv", raw_data_minus)




stokes, errors, etas, _ = perform_tomography_analysis(raw_data_plus, post_select_window=window)
print("Errors:", errors)
print("Efficiencies:", etas)
plot_basis_histograms(raw_data_plus, basis="X")
plot_basis_histograms(raw_data_plus, basis="Y")
plot_basis_histograms(raw_data_plus, basis="Z")
Plot_Stokes_Vector(stokes)
Plot_Density_Matrix(stokes)


stokes, errors, etas, _ = perform_tomography_analysis(raw_data_minus, post_select_window=window)
print("Errors:", errors)
print("Stokes vector:", stokes)
print("Efficiencies:", etas)
plot_basis_histograms(raw_data_minus, basis="X")
plot_basis_histograms(raw_data_minus, basis="Y")
plot_basis_histograms(raw_data_minus, basis="Z")
Plot_Stokes_Vector(stokes)
Plot_Density_Matrix(stokes) 