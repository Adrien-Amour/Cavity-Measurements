from adriq.ad9910 import general_setting_master, general_setting_slave, general_setting_standalone, single_tone_profile_setting, interpolate_rf_power
from adriq.pulse_sequencer import control_pulse_sequencer, write_pulse_sequencer
from adriq.experiment import *
from adriq.RedLabs_Dac import Redlabs_DAC

import csv  # Import CSV module for saving data
from tqdm import tqdm

redlabds_dac = Client(Redlabs_DAC)

cav_resonance = 34.2
        
#probe_detuning_range = np.arange(-30, 30.2, 2)  # Detuning values from -30 to 30
#probe_detuning_range = np.append(np.arange(-30, 30.2, 2), np.array([-45, -40, -35, 35, 40, 45]))
#probe_detuning_range = np.sort(probe_detuning_range)

fluorescence_results = []

dds_dict = load_dds_dict("singletone", r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\dds_config.cfg")

# Create a Pulse_Sequencer instance
pulse_sequencer = Pulse_Sequencer(port="COM5", ps_end_pin=2, pmt_gate_pin=1, ps_sync_pin=0)

# Create an Experiment_Builder instance
experiment_builder = Experiment_Builder_Singletone(dds_dict, pulse_sequencer, N_Cycles=5E4)

# Set trapping parameters
experiment_builder.set_trapping_parameters(
    trapping_detuning_dict={"397c": -50,
                            "854 SP1": -25,
                            "850 RP": -50},
    trapping_amplitude_dict={"397c": 0.8,
                             "854 SP1": 0.1,
                             "850 RP": 0.7}
)

exp_runner = Experiment_Runner(
    dds_dict,
    pulse_sequencer,
    timeout=100,
    pmt_threshold=2500,
    expected_fluorescence=6500,
    pulse_expected_fluorescence=9000,
    sp_threshold=None,
    load_timeout=50
)


# Create sections
experiment_builder.create_section(
    name="Cool",
    duration=10,
    detunings={"397c": -18,
                "854 SP1": -25,
                "850 RP": -50,},
    amplitudes={"397c": 0.2,
                "854 SP1": 1,
                "850 RP": 0.7,},
    pmt_gate_high=True
)


experiment_builder.create_section(
    name="Probe",
    duration=10,
    detunings={"397c": -18,
                "850 RP": -50,
                "854 Cav": cav_resonance},
    amplitudes={"397c": 0.2,
                "850 RP": 0.7,
                "854 Cav": 0},
    pmt_gate_high=True
)

Piezo_Voltages = np.arange(-10,15,5)

experiment_builder.flash()
exp_runner.measure_expected_fluorescence()

# Perform the experiment and collect fluorescence data
for v in tqdm(Piezo_Voltages, desc="Scanning Piezo Voltages", unit="step"):
    redlabds_dac.set_piezo_voltage(v)
    time.sleep(1)  # sleep for the piezo to settle
    exp_runner.start_experiment(N=10)
    exp_runner.plot_time_diffs_histogram("signal-f", lower_cutoff=0, upper_cutoff=40)
    if exp_runner.N_Valid_Pulses == 0:
        fluorescence_results.append(0)
        fluorescence_rate = 0
    else:
        fluorescence_rate = exp_runner.get_counts_in_window("signal-f", lower_cutoff=26, upper_cutoff=34)['pmt_counts_chan'] / exp_runner.N_Valid_Pulses
        fluorescence_results.append(fluorescence_rate / (8E-6))# convert to cps

    print(f"Fluorescence: {fluorescence_rate}")

# Save fluorescence vs. voltage data to a CSV file
csv_filename = "fluorescence_vs_voltage.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Piezo Voltage (V)", "Fluorescence Rate"])  # Write header
    for voltage, fluorescence in zip(Piezo_Voltages, fluorescence_results):
        writer.writerow([voltage, fluorescence])  # Write data rows

print(f"Fluorescence vs. Voltage data saved to {csv_filename}")

# Plot fluorescence vs. voltage
plt.figure(figsize=(8, 6))
plt.plot(Piezo_Voltages, fluorescence_results, marker="o", linestyle="-", color="b", label="Fluorescence Rate")
plt.title("Fluorescence vs. Piezo Voltage")
plt.xlabel("Piezo Voltage (V)")
plt.ylabel("Fluorescence Rate(cps)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("fluorescence_vs_voltage.png")  # Save the plot as a PNG file
plt.show()