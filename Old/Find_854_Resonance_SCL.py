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
from adriq.WM_SCL import *

# Start timing the script
start_time = time.time()

# Create the dictionary of DDS instances
calib_directory = r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment Files and VIs\AOM calibration VI\Calibration_Files"

def gaussian(amplitude, mu, sigma): #Sigma is Half Width 1/e height (multiply by 2*sqrt(ln(2)) to get FWHM ~1.665)
    return lambda t: amplitude * np.exp(-((t - mu)**2) / (sigma**2))

dds_dict = load_dds_dict("ram", r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\dds_config.cfg")

pulse_sequencer = Pulse_Sequencer()

exp_sequence = Experiment_Builder(dds_dict, pulse_sequencer, ram_step=0.04, N_Cycles=5E4)

exp_sequence.set_detunings(detuning_dict={"850 SP1": 0, "850 SP2": 0, "854 SP1": 0, "854 SP2": 0, "397a": -48, "854 Cav": 0, "397c": -18, "866": -18, "866 OP": 30, "850 RP": 0})
exp_sequence.set_trapping_parameters(
    trapping_detuning_dict={"397c": -50},
    trapping_amplitude_dict={"397c": 0.3}
)
# Create the cooling section (cool for 6 useconds)
exp_sequence.load_cooling_section(length=6)

# Create the single photon section
exp_sequence.create_section(name="Single Photon", duration=12, dds_functions={
    "854 Cav": lambda t: 0.07,
}, pmt_gate_high=False, coincidence_detector=True, coincidence_low=1, coincidence_high=100)

exp_sequence.build_ram_arrays()

# Plot the amplitude arrays
#exp_sequence.plot_amplitude_arrays()
exp_sequence.flash()

exp_runner = Experiment_Runner(
    dds_dict,
    pulse_sequencer,
    timeout=100,
    pmt_threshold=1500,
    expected_fluorescence= 0,
    pulse_expected_fluorescence= 0,
    sp_threshold=None,
    load_timeout = 50,
    cavity_lock=False # we need to keep it in ream mode
)


length = sum(section['duration'] for section in exp_sequence.playback_sections)

fig = go.Figure()

html_file = "efficiency_vs_detuning.html"


fig.write_html(html_file)
webbrowser.open(html_file)
amplitude_values = np.array([0.02])




file_name = file_name()
# ...existing imports and setup...

# shifty_values = np.arange(-0.01, 0.011, 0.002)  # Scan shift from -0.01 to +0.01
# shifty_values = np.arange(-0.002, 0.002, 0.0002)  # Scan shift from -0.002 to +0.002
shifty_values = np.arange(-0.0003, 0.00031, 0.0001)  # Scan shift from -0.0003 to +0.0002



for amplitude in amplitude_values:
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name=f'Channel 4 Amplitude {amplitude}'))

for amplitude in tqdm(amplitude_values, desc="Amplitude Progress"):
    efficiencies = [] # total efficiencies 
    exp_runner.experiment_trap_depth = 1.2
    current_shift = 0
    for shifty in tqdm(shifty_values, desc=f"Processing amplitude={amplitude}", leave=False):
        exp_runner.extra = 0


        # Calculate incremental shift (accumulating)
        shift_increment = shifty - current_shift
        shift_854(shift_increment)  # You must define this function to apply the shift
        current_shift = shifty
        exp_runner.measure_expected_fluorescence()
        exp_runner.clear_channels()
        exp_runner.start_experiment(N=3)

        total_counts = exp_runner.get_counts_in_window("signal-sp", lower_cutoff=6.3, upper_cutoff=20)
        counts_channel_4 = total_counts.get('single_photon_chan4', 0)
        counts_channel_3 = total_counts.get('single_photon_chan3', 0)

        with open("counts_log.csv", mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                counts_channel_3,
                counts_channel_4,
            ])

        if exp_runner.N_Valid_Pulses == 0:
            efficiency_1 = 0
            efficiency_2 = 0
        else:
            efficiency_1 = (counts_channel_4 / exp_runner.N_Valid_Pulses) * 100
            efficiency_2 = (counts_channel_3 / exp_runner.N_Valid_Pulses) * 100
        efficiencies.append(efficiency_1+efficiency_2)

        # Update the plot for Channel 4
        fig.data[amplitude_values.tolist().index(amplitude) * 2].update(x=shifty_values[:len(efficiencies)], y=efficiencies)

        # Update the plot for Channel 3

        fig.update_layout(
            title=f'Efficiency vs. Shift',
            xaxis_title='Shift',
            yaxis_title='Efficiency (%)',
            legend_title='Channels',
            template='plotly_white'
        )

        fig.write_html(html_file)

        save_data(
            file_name,
            amplitude=amplitude,
            shifty=shifty,
            counts_channel_3=counts_channel_3,
            counts_channel_4=counts_channel_4,
            valid_pulses=exp_runner.N_Valid_Pulses,
            efficiency=efficiencies
        )

    # Reset shift to zero at the end of scan
    shift_854(-current_shift)
    # Find the shift value where Channel 4 efficiency was maximized
max_index = np.argmax(efficiencies)
optimal_shift = shifty_values[max_index]

print(efficiencies, shifty_values, max_index)
print(f"Max efficiency for Channel 4 at shift: {optimal_shift}")

# --- Lorentzian fit and print central shift ---
import scipy.optimize

def lorentzian(x, a, x0, gamma, c):
    return a * gamma**2 / ((x - x0)**2 + gamma**2) + c

try:
    # Initial guess: amplitude, center, width, offset
    p0 = [max(efficiencies), shifty_values[np.argmax(efficiencies)], 0.0001, min(efficiencies)]
    popt, pcov = scipy.optimize.curve_fit(lorentzian, shifty_values, efficiencies, p0=p0)
    fitted_center = popt[1]
    print(f"Lorentzian fit center (central shift): {fitted_center}")
except Exception as e:
    print(f"Lorentzian fit failed: {e}")


# Option to shift by optimal or Lorentzian center
shift_choice = None
if 'fitted_center' in locals():
    print("Choose shift to apply:")
    print("1: Optimal shift (max efficiency)")
    print("2: Lorentzian fit center")
    try:
        shift_choice = input("Enter 1 or 2: ").strip()
    except Exception:
        shift_choice = '1'
    if shift_choice == '2':
        shift_to_apply = fitted_center
        print(f"Applying Lorentzian fit center: {fitted_center}")
    else:
        shift_to_apply = optimal_shift
        print(f"Applying optimal shift: {optimal_shift}")
else:
    shift_to_apply = optimal_shift
    print(f"Applying optimal shift: {optimal_shift}")

shift_854(shift_to_apply)
print("Final shift applied:", shift_to_apply)