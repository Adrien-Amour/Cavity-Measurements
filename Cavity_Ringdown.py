from adriq.ad9910 import general_setting_master, general_setting_slave, general_setting_standalone, single_tone_profile_setting, interpolate_rf_power
from adriq.pulse_sequencer import control_pulse_sequencer, write_pulse_sequencer
from adriq.experiment import *
import csv



        
fluorescence_results = []

dds_dict = load_dds_dict("ram", r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\dds_config.cfg")
dds_dict = {key: value for key, value in dds_dict.items() if key == "854 Cav"}

# Create a Pulse_Sequencer instance
pulse_sequencer = Pulse_Sequencer(port="COM5", ps_end_pin=2, pmt_gate_pin=1, ps_sync_pin=0)

# Create an Experiment_Builder instance
experiment_builder = Experiment_Builder(dds_dict, pulse_sequencer, N_Cycles=1E5)
experiment_builder.set_detunings(detuning_dict={"854 Cav": 0})

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

experiment_builder.set_trapping_parameters(
    trapping_detuning_dict={},
    trapping_amplitude_dict={},
)
experiment_builder.create_cooling_section(
    length=10,
    amplitude_dict={"854 Cav": 0.02})
experiment_builder.create_section(
    name="Decay",
    duration=5,
    dds_functions={"854 Cav": lambda t, a=0.0: a if float(t) <= 5 else 0.0},
    pmt_gate_high=True,
    coincidence_detector=True,
    coincidence_low=1,
    coincidence_high=100,
)
experiment_builder.build_ram_arrays()
experiment_builder.flash()

file_name = generate_file_name()
exp_runner.start_experiment(N=25)

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the delayed exponential decay function
def delayed_exponential_decay(x, A, width, delay):
    return np.where(x < delay, A, A * np.exp(-(x - delay) * 2 * np.pi * width))

# Get the time differences
inject_duration_us = 10.0
decay_duration_us = 5.0
inject_start_us = 0.0
inject_end_us = inject_start_us + inject_duration_us
decay_start_us = inject_end_us
decay_end_us = decay_start_us + decay_duration_us

# Pull all time differences across Inject + Decay so we can compute steady-state (Inject) and decay rates.
time_diffs_all = exp_runner.get_time_diffs(
    "signal-sp",
    lower_cutoff=None,
    upper_cutoff=None,
)

time_diffs_sp3_all = time_diffs_all.get("single_photon_chan3", [])
time_diffs_sp4_all = time_diffs_all.get("single_photon_chan4", [])

# Keep decay-only arrays for the existing ringdown fits/plots below.
time_diffs_sp3 = [t for t in time_diffs_sp3_all if decay_start_us <= t <= decay_end_us]
time_diffs_sp4 = [t for t in time_diffs_sp4_all if decay_start_us <= t <= decay_end_us]
print()
print(f"Channel 3: {len(time_diffs_sp3_all)} total counts (Inject+Decay), {len(time_diffs_sp3)} decay-window counts")
print(f"Channel 4: {len(time_diffs_sp4_all)} total counts (Inject+Decay), {len(time_diffs_sp4)} decay-window counts")


def _counts_in_window(time_list, start_us, end_us):
    if time_list is None:
        return 0
    arr = np.asarray(time_list, dtype=float)
    if arr.size == 0:
        return 0
    return int(np.count_nonzero((arr >= start_us) & (arr < end_us)))


def _rate_cps(counts, window_width_us, n_sequences=None):
    if window_width_us <= 0:
        return float("nan")
    denom_s = (window_width_us * 1e-6) if n_sequences is None else (n_sequences * window_width_us * 1e-6)
    return counts / denom_s


# Estimate how many identical Inject/Decay windows were executed.
# Assumption: `exp_runner.start_experiment(N=...)` runs N repeats, each containing `experiment_builder.N_Cycles` cycles.
N_repeats = 200
N_cycles = int(getattr(experiment_builder, "N_Cycles", 1) or 1)
n_sequences = max(1, int(N_repeats * N_cycles))

# Define a "steady-state" window as the last 2 µs of the Inject period (adjust if you want).
steady_state_start_us = max(inject_start_us, inject_end_us - 2.0)
steady_state_end_us = inject_end_us

for chan_name, chan_times in [
    ("Channel 3", time_diffs_sp3_all),
    ("Channel 4", time_diffs_sp4_all),
]:
    inj_counts = _counts_in_window(chan_times, inject_start_us, inject_end_us)
    ss_counts = _counts_in_window(chan_times, steady_state_start_us, steady_state_end_us)
    dec_counts = _counts_in_window(chan_times, decay_start_us, decay_end_us)

    inj_cps = _rate_cps(inj_counts, inject_duration_us, n_sequences=n_sequences)
    ss_cps = _rate_cps(ss_counts, steady_state_end_us - steady_state_start_us, n_sequences=n_sequences)
    dec_cps = _rate_cps(dec_counts, decay_duration_us, n_sequences=n_sequences)

    print(
        f"{chan_name} rates (counts/s per sequence): "
        f"Inject={inj_cps:.3g} cps, SteadyState[{steady_state_start_us:.1f},{steady_state_end_us:.1f}]µs={ss_cps:.3g} cps, "
        f"Decay={dec_cps:.3g} cps"
    )








# Fit the delayed exponential decay
hist, bin_edges = np.histogram(time_diffs_sp3, bins=200, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Remove non-finite values
finite_mask = np.isfinite(hist) & np.isfinite(bin_centers)
hist = hist[finite_mask]
bin_centers = bin_centers[finite_mask]

# Initial guess for the parameters A, width, and delay
initial_guess = [6000, 0.5, 11]

# Fit the curve
popt, pcov = curve_fit(delayed_exponential_decay, bin_centers, hist, p0=initial_guess)

# Plot the fitted curve
x_fit = np.linspace(min(bin_centers), max(bin_centers), 1000)
y_fit = delayed_exponential_decay(x_fit, *popt)
# plt.plot(x_fit, y_fit, 'r-', label='fit: A=%5.3f, width=%5.3f, delay=%5.3f' % tuple(popt))
for z in zip(bin_centers, hist):
    save_data(
        file_name, bins=z[0], values=z[1]
    )

# Plot the histogram and the fitted curve on the same graph
plt.hist(time_diffs_sp3, bins=1000, density=True, alpha=0.6, color='g', label='Histogram')

x_fit = np.linspace(min(bin_centers), max(bin_centers), 1000)
y_fit = delayed_exponential_decay(x_fit, *popt)
plt.plot(x_fit, y_fit, 'r-', label='Fit: A=%5.3f, width=%5.3f, delay=%5.3f' % tuple(popt))

plt.xlabel('Time Difference (microseconds)')
plt.ylabel('Density')
plt.title('Histogram and Delayed Exponential Decay Fit for single_photon_chan_3')
plt.legend()
plt.show()



















# Fit the delayed exponential decay
hist, bin_edges = np.histogram(time_diffs_sp4, bins=200, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Remove non-finite values
finite_mask = np.isfinite(hist) & np.isfinite(bin_centers)
hist = hist[finite_mask]
bin_centers = bin_centers[finite_mask]

# Initial guess for the parameters A, width, and delay
initial_guess = [6000, 0.5, 11]

# Fit the curve
popt, pcov = curve_fit(delayed_exponential_decay, bin_centers, hist, p0=initial_guess)

# Plot the fitted curve
x_fit = np.linspace(min(bin_centers), max(bin_centers), 1000)
y_fit = delayed_exponential_decay(x_fit, *popt)
# plt.plot(x_fit, y_fit, 'r-', label='fit: A=%5.3f, width=%5.3f, delay=%5.3f' % tuple(popt))
for z in zip(bin_centers, hist):
    save_data(
        file_name, bins=z[0], values=z[1]
    )

# Plot the histogram and the fitted curve on the same graph
plt.hist(time_diffs_sp4, bins=1000, density=True, alpha=0.6, color='g', label='Histogram')

x_fit = np.linspace(min(bin_centers), max(bin_centers), 1000)
y_fit = delayed_exponential_decay(x_fit, *popt)
plt.plot(x_fit, y_fit, 'r-', label='Fit: A=%5.3f, width=%5.3f, delay=%5.3f' % tuple(popt))

plt.xlabel('Time Difference (microseconds)')
plt.ylabel('Density')
plt.title('Histogram and Delayed Exponential Decay Fit for single_photon_chan_4')
plt.legend()
plt.show()


bins = np.linspace(
    min(min(time_diffs_sp3, default=0), min(time_diffs_sp4, default=0)),
    max(max(time_diffs_sp3, default=1), max(time_diffs_sp4, default=1)),
    200
)

# Compute histograms
hist3, _ = np.histogram(time_diffs_sp3, bins=bins)
hist4, _ = np.histogram(time_diffs_sp4, bins=bins)

# Normalize both histograms to the same maximum height
if hist3.max() > 0:
    hist3 = hist3 / hist3.max()
if hist4.max() > 0:
    hist4 = hist4 / hist4.max()

bin_centers = (bins[:-1] + bins[1:]) / 2

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(bin_centers, hist3, label='Channel 3', color='b', alpha=0.7)
plt.plot(bin_centers, hist4, label='Channel 4', color='r', alpha=0.7)
plt.xlabel('Time Difference (microseconds)')
plt.ylabel('Normalized Counts (max=1)')
plt.title('Normalized Histograms for Channels 3 and 4')
plt.legend()
plt.show()


# ...existing code...

# Fit the delayed exponential decay for channel 4
hist, bin_edges = np.histogram(time_diffs_sp4, bins=200, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Remove non-finite values
finite_mask = np.isfinite(hist) & np.isfinite(bin_centers)
hist = hist[finite_mask]
bin_centers = bin_centers[finite_mask]

# Initial guess for the parameters A, width, and delay
initial_guess = [6000, 0.5, 11]

# Fit the curve for channel 4
popt4, pcov4 = curve_fit(delayed_exponential_decay, bin_centers, hist, p0=initial_guess)
print("Channel 4 fit parameters: A=%.3f, width=%.3f, delay=%.3f" % tuple(popt4))

# Fit the delayed exponential decay for channel 3 (if enough data)
hist3_fit, bin_edges3 = np.histogram(time_diffs_sp3, bins=200, density=True)
bin_centers3 = (bin_edges3[:-1] + bin_edges3[1:]) / 2
finite_mask3 = np.isfinite(hist3_fit) & np.isfinite(bin_centers3)
hist3_fit = hist3_fit[finite_mask3]
bin_centers3 = bin_centers3[finite_mask3]
try:
    popt3, pcov3 = curve_fit(delayed_exponential_decay, bin_centers3, hist3_fit, p0=initial_guess)
    print("Channel 3 fit parameters: A=%.3f, width=%.3f, delay=%.3f" % tuple(popt3))
except Exception as e:
    print("Channel 3 fit failed:", e)

# ...existing code for normalized histograms...
bins = np.linspace(
    min(min(time_diffs_sp3, default=0), min(time_diffs_sp4, default=0)),
    max(max(time_diffs_sp3, default=1), max(time_diffs_sp4, default=1)),
    200
)

# Compute histograms
hist3, _ = np.histogram(time_diffs_sp3, bins=bins)
hist4, _ = np.histogram(time_diffs_sp4, bins=bins)

# Normalize both histograms to the same maximum height
if hist3.max() > 0:
    hist3 = hist3 / hist3.max()
if hist4.max() > 0:
    hist4 = hist4 / hist4.max()

bin_centers = (bins[:-1] + bins[1:]) / 2

plt.figure(figsize=(8, 5))
plt.plot(bin_centers, hist3, label='Channel 3', color='b', alpha=0.7)
plt.plot(bin_centers, hist4, label='Channel 4', color='r', alpha=0.7)
plt.xlabel('Time Difference (microseconds)')
plt.ylabel('Normalized Counts (max=1)')
plt.title('Normalized Histograms for Channels 3 and 4')
plt.legend()
plt.show()