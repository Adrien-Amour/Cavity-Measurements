from adriq.ad9910 import general_setting_master, general_setting_slave, general_setting_standalone, single_tone_profile_setting, interpolate_rf_power
from adriq.pulse_sequencer import control_pulse_sequencer, write_pulse_sequencer
from adriq.experiment import *

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import csv

try:
    from scipy.optimize import curve_fit
    from scipy.special import wofz
except Exception:  # scipy is optional at import time
    curve_fit = None
    wofz = None


def _voigt_unit_area(x: np.ndarray, sigma: float, gamma: float) -> np.ndarray:
    """Unit-area Voigt profile.

    Parameters
    ----------
    x:
        Frequency detuning axis (same units as sigma/gamma).
    sigma:
        Gaussian standard deviation (same units as x).
    gamma:
        Lorentzian HWHM (half-width at half-maximum; same units as x).
        Note: Lorentzian FWHM = 2 * gamma.
    """
    if wofz is None:
        raise RuntimeError("SciPy is required for Voigt fitting (missing scipy).")

    sigma = float(sigma)
    gamma = float(gamma)
    if sigma <= 0:
        sigma = 1e-12
    z = ((x + 1j * gamma) / (sigma * np.sqrt(2.0))).astype(np.complex128)
    return np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))


def gaussian_broadened_voigt(
    x: np.ndarray,
    amplitude: float,
    x0: float,
    sigma_g: float,
    offset: float,
    gamma_l: float,
) -> np.ndarray:
    """Voigt (Gaussian-broadened Lorentzian) with a constant offset."""
    return offset + amplitude * _voigt_unit_area(x - x0, sigma=sigma_g, gamma=gamma_l)


def fit_voigt_fixed_lorentz_fwhm(
    x_mhz: np.ndarray,
    y: np.ndarray,
    lorentz_fwhm_mhz: float = 0.5,
):
    """Fit a Voigt profile with Lorentzian FWHM fixed.

    Returns (popt, pcov) from scipy.optimize.curve_fit.
    popt = [amplitude, x0, sigma_g, offset]
    """
    if curve_fit is None:
        raise RuntimeError(
            "SciPy is required for fitting. Install with `pip install scipy` (in your active env)."
        )

    x_mhz = np.asarray(x_mhz, dtype=float)
    y = np.asarray(y, dtype=float)
    gamma_l_mhz = float(lorentz_fwhm_mhz) / 2.0  # HWHM

    offset0 = float(np.nanmin(y))
    amp0 = float(np.nanmax(y) - offset0)
    x00 = float(x_mhz[int(np.nanargmax(y))])
    sigma0 = 0.2  # MHz; a reasonable starting guess

    def model(x, amplitude, x0, sigma_g, offset):
        return gaussian_broadened_voigt(
            x,
            amplitude=amplitude,
            x0=x0,
            sigma_g=sigma_g,
            offset=offset,
            gamma_l=gamma_l_mhz,
        )

    bounds = (
        [0.0, np.min(x_mhz), 1e-6, -np.inf],
        [np.inf, np.max(x_mhz), np.inf, np.inf],
    )
    popt, pcov = curve_fit(
        model,
        x_mhz,
        y,
        p0=[amp0, x00, sigma0, offset0],
        bounds=bounds,
        maxfev=20000,
    )
    return popt, pcov, gamma_l_mhz

detunings = np.arange(-0.8, 0.825, 0.05) #detuning range in MHz
efficiencies_chan3 = []
efficiencies_chan4 = []
        
fluorescence_results = []

dds_dict = load_dds_dict("singletone", r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\dds_config.cfg")
dds_dict = {key: value for key, value in dds_dict.items() if key == "854 Cav"}

# Create a Pulse_Sequencer instance
pulse_sequencer = Pulse_Sequencer(port="COM5", ps_end_pin=2, pmt_gate_pin=1, ps_sync_pin=0)

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

for detuning in detunings:
    print(f"Running experiment for detuning {detuning} MHz...")
    # Create an Experiment_Builder instance
    experiment_builder = Experiment_Builder_Singletone(dds_dict, pulse_sequencer, N_Cycles=int(1e3))

    # Set trapping parameters
    experiment_builder.set_trapping_parameters(
        trapping_detuning_dict={},
        trapping_amplitude_dict={}
    )

    # Create sections
    experiment_builder.create_section(
        name="Inject",
        duration=10,
        detunings={"854 Cav": detuning},
        amplitudes={"854 Cav":0.3},
        pmt_gate_high=True
    )

    experiment_builder.create_section(
        name="Decay",
        duration=5,
        detunings={"854 Cav": detuning},
        amplitudes={"854 Cav": 0.3},
        pmt_gate_high=True
    )

    experiment_builder.flash()
    exp_runner.clear_channels()
    exp_runner.start_experiment(N=20)

    # Pull all time differences across Inject + Decay so we can compute steady-state (Inject) and decay rates.
    time_diffs = exp_runner.get_time_diffs(
        "signal-sp",
        lower_cutoff=0,
        upper_cutoff=15,
    )

    time_diffs_sp3 = time_diffs.get("single_photon_chan3", [])
    time_diffs_sp4 = time_diffs.get("single_photon_chan4", [])
    efficiency_chan3 = len(time_diffs_sp3) / exp_runner.N_Valid_Pulses
    efficiency_chan4 = len(time_diffs_sp4) / exp_runner.N_Valid_Pulses
    efficiencies_chan3.append(efficiency_chan3)
    efficiencies_chan4.append(efficiency_chan4)


# --- Save + plot spectra ---
out_dir = Path(__file__).resolve().parent

csv_path = out_dir / "cavity_resonance_spectra.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["detuning_MHz", "efficiency_chan3", "efficiency_chan4"])
    for detuning_mhz, eff3, eff4 in zip(detunings, efficiencies_chan3, efficiencies_chan4):
        writer.writerow([float(detuning_mhz), float(eff3), float(eff4)])

fig_path = out_dir / "cavity_resonance_spectra.png"
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(7, 6))

ax1.plot(detunings, efficiencies_chan3, marker="o", linewidth=1)
ax1.set_ylabel("Chan3 eff. (counts/shot)")
ax1.grid(True, alpha=0.3)

ax2.plot(detunings, efficiencies_chan4, marker="o", linewidth=1)
ax2.set_xlabel("Detuning (MHz)")
ax2.set_ylabel("Chan4 eff. (counts/shot)")
ax2.grid(True, alpha=0.3)


# --- Voigt fit (Lorentzian FWHM fixed at 500 kHz = 0.5 MHz) ---
lorentz_fwhm_mhz = 0.5

try:
    x_fit = np.linspace(float(np.min(detunings)), float(np.max(detunings)), 2000)

    popt3, pcov3, gamma_l_mhz = fit_voigt_fixed_lorentz_fwhm(
        detunings, efficiencies_chan3, lorentz_fwhm_mhz=lorentz_fwhm_mhz
    )
    y3_fit = gaussian_broadened_voigt(
        x_fit,
        amplitude=popt3[0],
        x0=popt3[1],
        sigma_g=popt3[2],
        offset=popt3[3],
        gamma_l=gamma_l_mhz,
    )
    ax1.plot(x_fit, y3_fit, linewidth=2, label=f"Voigt fit (L-FWHM={lorentz_fwhm_mhz:.3f} MHz)")
    ax1.legend(loc="best")

    popt4, pcov4, gamma_l_mhz = fit_voigt_fixed_lorentz_fwhm(
        detunings, efficiencies_chan4, lorentz_fwhm_mhz=lorentz_fwhm_mhz
    )
    y4_fit = gaussian_broadened_voigt(
        x_fit,
        amplitude=popt4[0],
        x0=popt4[1],
        sigma_g=popt4[2],
        offset=popt4[3],
        gamma_l=gamma_l_mhz,
    )
    ax2.plot(x_fit, y4_fit, linewidth=2, label=f"Voigt fit (L-FWHM={lorentz_fwhm_mhz:.3f} MHz)")
    ax2.legend(loc="best")

    gauss_fwhm_factor = 2.0 * np.sqrt(2.0 * np.log(2.0))
    sigma3_mhz = float(popt3[2])
    sigma4_mhz = float(popt4[2])
    print("\n--- Voigt fit results (Lorentzian FWHM fixed) ---")
    print(f"Lorentzian: FWHM={lorentz_fwhm_mhz:.6f} MHz (HWHM={lorentz_fwhm_mhz/2:.6f} MHz)")
    print(
        "Chan3: x0={:.6f} MHz, sigma_G={:.6f} MHz (G-FWHM={:.6f} MHz)".format(
            float(popt3[1]), sigma3_mhz, gauss_fwhm_factor * sigma3_mhz
        )
    )
    print(
        "Chan4: x0={:.6f} MHz, sigma_G={:.6f} MHz (G-FWHM={:.6f} MHz)".format(
            float(popt4[1]), sigma4_mhz, gauss_fwhm_factor * sigma4_mhz
        )
    )
except Exception as e:
    print(f"Voigt fit skipped/failed: {e}")

fig.suptitle("Cavity resonance spectra")
fig.tight_layout()
fig.savefig(fig_path, dpi=200)
plt.show()

def fwhm_from_curve(x, y, baseline):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    ymax = float(np.max(y))
    half = float(baseline) + 0.5 * (ymax - float(baseline))

    s = y - half
    # indices i where s[i] and s[i+1] have opposite sign (a crossing)
    crossings = np.where(np.diff(np.signbit(s)))[0]
    if crossings.size < 2:
        raise RuntimeError("Could not find two half-maximum crossings to compute FWHM.")

    imax = int(np.argmax(y))
    left_candidates = crossings[crossings < imax]
    right_candidates = crossings[crossings >= imax]
    if left_candidates.size == 0 or right_candidates.size == 0:
        raise RuntimeError("Half-maximum crossings not found on both sides of the peak.")

    iL = int(left_candidates[-1])
    iR = int(right_candidates[0])

    def lin_x_at_half(i):
        x0, x1 = x[i], x[i + 1]
        y0, y1 = y[i], y[i + 1]
        return x0 + (half - y0) * (x1 - x0) / (y1 - y0)

    x_left = lin_x_at_half(iL)
    x_right = lin_x_at_half(iR)
    return x_right - x_left


# --- Overall Voigt FWHM (approx + numerical from fitted curve) ---
gauss_fwhm_factor = 2.0 * np.sqrt(2.0 * np.log(2.0))
L = float(lorentz_fwhm_mhz)

G3 = gauss_fwhm_factor * float(popt3[2])
G4 = gauss_fwhm_factor * float(popt4[2])

# Olivero–Longbothum approximation
V3_approx = 0.5346 * L + np.sqrt(0.2166 * L**2 + G3**2)
V4_approx = 0.5346 * L + np.sqrt(0.2166 * L**2 + G4**2)

# Numerical FWHM from the plotted fit curve (uses fitted offset as baseline)
V3_num = fwhm_from_curve(x_fit, y3_fit, baseline=float(popt3[3]))
V4_num = fwhm_from_curve(x_fit, y4_fit, baseline=float(popt4[3]))

print("\n--- Overall Voigt FWHM ---")
print(f"Chan3: FWHM ≈ {V3_approx:.6f} MHz (approx), {V3_num:.6f} MHz (numerical)")
print(f"Chan4: FWHM ≈ {V4_approx:.6f} MHz (approx), {V4_num:.6f} MHz (numerical)")