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

# Start timing the script
start_time = time.time()

# Create the dictionary of DDS instances
calib_directory = r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment Files and VIs\AOM calibration VI\Calibration_Files"

def gaussian(amplitude, mu, sigma): #Sigma is Half Width 1/e height (multiply by 2*sqrt(ln(2)) to get FWHM ~1.665)
    return lambda t: amplitude * np.exp(-((t - mu)**2) / (sigma**2))

dds_dict = load_dds_dict("ram", r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\dds_config.cfg")


pulse_sequencer = Pulse_Sequencer()

exp_sequence = Experiment_Builder(dds_dict, pulse_sequencer, ram_step=0.04, N_Cycles=5E4)

exp_sequence.set_detunings(detuning_dict={"850 SP1": 0, "850 SP2": 0, "854 SP1": 5, "854 SP2": 0, "397a": -48, "854 Cav": 0, "397b": -18, "397c":-18, "866S": -18, "866 OP": 30, "850 RP": 0})
exp_sequence.set_trapping_parameters(
    trapping_detuning_dict={"397b": -50},
    trapping_amplitude_dict={"397b": 0.2}
)
# Create the cooling section (cool for 6 useconds)
#exp_sequence.create_cooling_section(length=6, amplitude_dict={"397b": 0.15, "854 SP1": 0.4, "850 SP1": 0.4})

exp_sequence.load_cooling(length=12)
# Create a Gaussian output function for DDS 397a

# Create op section 
exp_sequence.create_section(name="Optical Pumping", duration=6, dds_functions={
    "397b": lambda t: 0.1,
    "854 SP1": lambda t: 0.1,
}, pmt_gate_high=True) #First stage of Optical Pumping

exp_sequence.create_section(name="wait", duration=2, dds_functions={})


# Create the single photon section
exp_sequence.create_section(name="Single Photon", duration=12, dds_functions={
    "850 SP1": gaussian(amplitude=0.15, mu=6, sigma=1),
}, pmt_gate_high=False, coincidence_detector=True, coincidence_low=1, coincidence_high=100)

exp_sequence.build_ram_arrays()

# Plot the amplitude arrays
#exp_sequence.plot_amplitude_arrays()
exp_sequence.flash()

exp_runner = Experiment_Runner(
    dds_dict,
    pulse_sequencer,
    timeout=100,
    pmt_threshold=1800,
    expected_fluorescence= 8000,
    pulse_expected_fluorescence= 1800,
    sp_threshold=None,
    load_timeout = 50,
    cavity_lock=False
)


length = sum(section['duration'] for section in exp_sequence.playback_sections)

fig = go.Figure()

html_file = "efficiency_vs_detuning.html"


fig.write_html(html_file)
webbrowser.open(html_file)
amplitude_values = np.array([0.18])
#amplitude_values = np.arange(0.04,0.21,0.04)
detunings = np.arange(-12,12, 0.2)
# detunings = np.arange(-7, -1.9, 0.5)

# detunings = np.concatenate((
#     np.array([-12]),
#     np.arange(2.2,2.6, 0.1) # Every 0.2 from -18 to -13 (exclusive)

# ))



file_name = file_name()

for amplitude in amplitude_values:
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name=f'Channel 4 Amplitude {amplitude}'))
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name=f'Channel 3 Amplitude {amplitude}', line=dict(dash='dash')))

for amplitude in tqdm(amplitude_values, desc="Amplitude Progress"):
    efficiencies = []
    efficiencies_2 = []
    exp_runner.experiment_trap_depth = 0.8
    for detuning in tqdm(detunings, desc=f"Processing amplitude={amplitude}", leave=False):
        exp_runner.extra = 0
        
        exp_sequence.edit_detunings({"850 SP1": detuning})
        exp_sequence.edit_section(name="Single Photon", dds_functions={
        "850 SP1": gaussian(amplitude=amplitude, mu=6, sigma=1)
        })

        exp_sequence.build_ram_arrays()
        exp_sequence.flash()

        exp_runner.measure_expected_fluorescence()

        exp_runner.clear_channels()
        exp_runner.start_experiment(N=3)

        total_counts = exp_runner.get_counts_in_window("signal-sp", lower_cutoff=20.3, upper_cutoff=32)
        #exp_runner.plot_time_diffs_histogram("signal-sp", lower_cutoff=20, upper_cutoff=32)

        print("Total Counts:", total_counts)
        # Calculate efficiency for channel 4
        counts_channel_4 = total_counts.get('single_photon_chan4', 0)
        counts_channel_3 = total_counts.get('single_photon_chan3', 0)

        print(counts_channel_3)
        print(counts_channel_4)
        with open("counts_log.csv", mode="a", newline="") as f:
            writer = csv.writer(f)
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
        print(efficiency, "%")

        # Update the plot for Channel 4
        fig.data[amplitude_values.tolist().index(amplitude) * 2].update(x=detunings[:len(efficiencies)], y=efficiencies)

        # Update the plot for Channel 3
        fig.data[amplitude_values.tolist().index(amplitude) * 2 + 1].update(x=detunings[:len(efficiencies_2)], y=efficiencies_2)

        # Update layout
        fig.update_layout(
            title=f'Efficiency vs. Width)',
            xaxis_title='Detuning (MHz)',
            yaxis_title='Efficiency (%)',
            legend_title='Channels',
            template='plotly_white'
        )

        # Save the plot as an HTML file
        fig.write_html(html_file)
        
        save_data(
            file_name,
            amplitude=amplitude,
            detuning=detuning,
            counts_channel_3=counts_channel_3,
            counts_channel_4=counts_channel_4,
            valid_pulses=exp_runner.N_Valid_Pulses,
            efficiency_2=efficiency_2,
            efficiency=efficiency
        )




# Save detuning and efficiency of each channel to a CSV file
csv_file = "detuning_efficiency.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Detuning", "Channel 3 Efficiency", "Channel 4 Efficiency"])
    for detuning, eff_3, eff_4 in zip(detunings, efficiencies_2, efficiencies):
        writer.writerow([detuning, eff_3, eff_4])

print(efficiencies)
print(efficiencies_2)

# End timing the script
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
print(f"Elapsed time: {elapsed_time / 60:.2f} minutes")
