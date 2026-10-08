"""PyQt GUI for cavity ringdown measurements.

UI (as requested):
- Input boxes for: 854 Cav power (Inject amplitude), detuning (used in both sections), N_cycles, and number of runs.
- Button: "Run ringdown"
- Two side-by-side plots: Channel 3 and Channel 4, each showing histogram + delayed exponential decay fit.

Notes:
- This script depends on your `adriq` stack (Pulse_Sequencer/Experiment_Builder_Singletone/Experiment_Runner).
- The experiment is run in a background thread so the UI stays responsive.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Matplotlib (embedded)
try:
    # Newer Matplotlib (QtAgg)
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except Exception:  # pragma: no cover
    # Older Matplotlib (Qt5Agg)
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from matplotlib.figure import Figure

# Qt (prefer PyQt6, fall back to PyQt5)
try:
    from PyQt6 import QtCore, QtGui, QtWidgets

    QT6 = True
except Exception:  # pragma: no cover
    from PyQt5 import QtCore, QtGui, QtWidgets

    QT6 = False


def delayed_exponential_decay(x: np.ndarray, A: float, width: float, delay: float) -> np.ndarray:
    # Same functional form you used in the script.
    return np.where(x < delay, A, A * np.exp(-(x - delay) * 2 * np.pi * width))


def _safe_import_scipy_curve_fit():
    try:
        from scipy.optimize import curve_fit  # type: ignore

        return curve_fit
    except Exception:
        return None


@dataclass(frozen=True)
class RingdownInputs:
    power: float
    detuning: float
    n_cycles: int
    runs: int


@dataclass(frozen=True)
class ChannelPlotData:
    time_diffs: np.ndarray
    hist_bins: int
    density: bool
    bin_centers: np.ndarray
    hist: np.ndarray
    x_fit: Optional[np.ndarray]
    y_fit: Optional[np.ndarray]
    fit_params: Optional[Tuple[float, float, float]]  # (A, width, delay)


class PlotWidget(QtWidgets.QWidget):
    def __init__(self, title: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._title = title

        self._figure = Figure(constrained_layout=True)
        self._canvas = FigureCanvas(self._figure)
        self._ax = self._figure.add_subplot(111)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

        self._set_empty()

    def _set_empty(self, message: str = "No data") -> None:
        self._ax.clear()
        self._ax.set_title(self._title)
        self._ax.text(0.5, 0.5, message, ha="center", va="center", transform=self._ax.transAxes)
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        self._canvas.draw_idle()

    def set_error(self, message: str) -> None:
        self._set_empty(message)

    def set_data(self, data: ChannelPlotData, channel_label: str) -> None:
        self._ax.clear()

        if data.time_diffs.size == 0 or data.bin_centers.size == 0:
            self._set_empty("No counts")
            return

        # Histogram (same spirit as your snippet: density=True, green bars)
        self._ax.hist(
            data.time_diffs,
            bins=data.hist_bins,
            density=data.density,
            alpha=0.6,
            color="g",
            label="Histogram",
        )

        # Fit curve if available
        if data.x_fit is not None and data.y_fit is not None and data.fit_params is not None:
            A, width, delay = data.fit_params
            self._ax.plot(
                data.x_fit,
                data.y_fit,
                "r-",
                label=f"Fit: A={A:5.3f}, width={width:5.3f}, delay={delay:5.3f}",
            )

        self._ax.set_xlabel("Time Difference (microseconds)")
        self._ax.set_ylabel("Density" if data.density else "Counts")
        self._ax.set_title(
            f"Histogram and Delayed Exponential Decay Fit for {channel_label}"
        )
        self._ax.legend(loc="best")
        self._canvas.draw_idle()


class RingdownWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(dict)  # {"ch3": ChannelPlotData, "ch4": ChannelPlotData, "summary": str}
    failed = QtCore.pyqtSignal(str)

    def __init__(self, inputs: RingdownInputs, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._inputs = inputs

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            result = self._run_ringdown(self._inputs)
            self.finished.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())

    def _run_ringdown(self, inputs: RingdownInputs) -> Dict[str, Any]:
        # Import here so the GUI can still open if adriq isn't importable.
        from adriq.experiment import (
            Experiment_Builder_Singletone,
            Experiment_Runner,
            Pulse_Sequencer,
            load_dds_dict,
        )

        # Match your original config approach.
        dds_dict = load_dds_dict(
            "singletone",
            r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\dds_config.cfg",
        )
        dds_dict = {key: value for key, value in dds_dict.items() if key == "854 Cav"}

        pulse_sequencer = Pulse_Sequencer(port="COM5", ps_end_pin=2, pmt_gate_pin=1, ps_sync_pin=0)

        experiment_builder = Experiment_Builder_Singletone(dds_dict, pulse_sequencer, N_Cycles=inputs.n_cycles)
        experiment_builder.set_trapping_parameters(trapping_detuning_dict={}, trapping_amplitude_dict={})

        exp_runner = Experiment_Runner(
            dds_dict,
            pulse_sequencer,
            timeout=100,
            pmt_threshold=None,
            expected_fluorescence=None,
            pulse_expected_fluorescence=0,
            sp_threshold=None,
            load_timeout=50,
        )

        # Sections: detuning same in both sections, power applied in Inject, off in Decay.
        experiment_builder.create_section(
            name="Inject",
            duration=10,
            detunings={"854 Cav": inputs.detuning},
            amplitudes={"854 Cav": inputs.power},
            pmt_gate_high=True,
        )

        experiment_builder.create_section(
            name="Decay",
            duration=5,
            detunings={"854 Cav": inputs.detuning},
            amplitudes={"854 Cav": 0},
            pmt_gate_high=True,
        )

        experiment_builder.flash()
        exp_runner.start_experiment(N=inputs.runs)

        inject_duration_us = 10.0
        decay_duration_us = 5.0
        inject_start_us = 0.0
        inject_end_us = inject_start_us + inject_duration_us
        decay_start_us = inject_end_us
        decay_end_us = decay_start_us + decay_duration_us

        # Pull Inject+Decay so we can compute steady-state and decay-window rates.
        time_diffs_all = exp_runner.get_time_diffs(
            "signal-sp",
            lower_cutoff=inject_start_us,
            upper_cutoff=decay_end_us,
        )
        time_diffs_sp3_all = np.asarray(time_diffs_all.get("single_photon_chan3", []), dtype=float)
        time_diffs_sp4_all = np.asarray(time_diffs_all.get("single_photon_chan4", []), dtype=float)

        # Use decay-only window for the ringdown histogram+fit (matches your original script).
        time_diffs_sp3_decay = time_diffs_sp3_all[(time_diffs_sp3_all >= decay_start_us) & (time_diffs_sp3_all < decay_end_us)]
        time_diffs_sp4_decay = time_diffs_sp4_all[(time_diffs_sp4_all >= decay_start_us) & (time_diffs_sp4_all < decay_end_us)]

        ch3 = self._make_plot_data(time_diffs_sp3_decay)
        ch4 = self._make_plot_data(time_diffs_sp4_decay)

        n_sequences = max(1, int(inputs.runs) * int(inputs.n_cycles))

        # Define "steady-state" as the last 2 µs of Inject.
        steady_state_start_us = max(inject_start_us, inject_end_us - 2.0)
        steady_state_end_us = inject_end_us

        def _counts_in_window(arr: np.ndarray, start_us: float, end_us: float) -> int:
            if arr.size == 0:
                return 0
            return int(np.count_nonzero((arr >= start_us) & (arr < end_us)))

        def _rate_cps(counts: int, window_width_us: float) -> float:
            if window_width_us <= 0:
                return float("nan")
            return counts / (n_sequences * window_width_us * 1e-6)

        def _format_channel_summary(label: str, arr: np.ndarray) -> str:
            inj_counts = _counts_in_window(arr, inject_start_us, inject_end_us)
            ss_counts = _counts_in_window(arr, steady_state_start_us, steady_state_end_us)
            dec_counts = _counts_in_window(arr, decay_start_us, decay_end_us)

            inj_cps = _rate_cps(inj_counts, inject_duration_us)
            ss_cps = _rate_cps(ss_counts, steady_state_end_us - steady_state_start_us)
            dec_cps = _rate_cps(dec_counts, decay_duration_us)

            return (
                f"{label}: SteadyState={ss_cps:.3g} cps ({ss_counts} cnts in {steady_state_end_us - steady_state_start_us:.1f} µs), "
                f"Inject={inj_cps:.3g} cps ({inj_counts} cnts), Decay={dec_cps:.3g} cps ({dec_counts} cnts)"
            )

        summary = (
            f"Normalization: {inputs.runs} runs × {inputs.n_cycles} cycles = {n_sequences} sequences\n"
            + _format_channel_summary("Ch3", time_diffs_sp3_all)
            + "\n"
            + _format_channel_summary("Ch4", time_diffs_sp4_all)
        )

        return {"ch3": ch3, "ch4": ch4, "summary": summary}

    def _make_plot_data(self, time_diffs: np.ndarray) -> ChannelPlotData:
        hist_bins = 1000
        density = True

        if time_diffs.size == 0:
            return ChannelPlotData(
                time_diffs=time_diffs,
                hist_bins=hist_bins,
                density=density,
                bin_centers=np.array([]),
                hist=np.array([]),
                x_fit=None,
                y_fit=None,
                fit_params=None,
            )

        # Use a coarser histogram for fitting stability (like your script).
        fit_bins = 200
        hist, bin_edges = np.histogram(time_diffs, bins=fit_bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        finite_mask = np.isfinite(hist) & np.isfinite(bin_centers)
        hist = hist[finite_mask]
        bin_centers = bin_centers[finite_mask]

        curve_fit = _safe_import_scipy_curve_fit()
        if curve_fit is None or hist.size < 5:
            return ChannelPlotData(
                time_diffs=time_diffs,
                hist_bins=hist_bins,
                density=density,
                bin_centers=bin_centers,
                hist=hist,
                x_fit=None,
                y_fit=None,
                fit_params=None,
            )

        A0 = float(np.nanmax(hist)) if hist.size else 1.0
        width0 = 0.5
        delay0 = 11.0
        initial_guess = [A0, width0, delay0]

        # Fit and prepare curve for plotting
        popt, _pcov = curve_fit(delayed_exponential_decay, bin_centers, hist, p0=initial_guess, maxfev=20000)
        x_fit = np.linspace(float(np.min(bin_centers)), float(np.max(bin_centers)), 1000)
        y_fit = delayed_exponential_decay(x_fit, *popt)

        return ChannelPlotData(
            time_diffs=time_diffs,
            hist_bins=hist_bins,
            density=density,
            bin_centers=bin_centers,
            hist=hist,
            x_fit=x_fit,
            y_fit=y_fit,
            fit_params=(float(popt[0]), float(popt[1]), float(popt[2])),
        )


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Cavity Ringdown")

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        self._power = QtWidgets.QLineEdit("0.04")
        self._detuning = QtWidgets.QLineEdit("0.0")
        self._n_cycles = QtWidgets.QLineEdit("1000")
        self._runs = QtWidgets.QLineEdit("50")

        # Basic numeric validation
        self._power.setValidator(QtGui.QDoubleValidator(0.0, 1e9, 6))
        self._detuning.setValidator(QtGui.QDoubleValidator(-1e9, 1e9, 6))
        self._n_cycles.setValidator(QtGui.QIntValidator(1, 10**9))
        self._runs.setValidator(QtGui.QIntValidator(1, 10**9))

        form = QtWidgets.QFormLayout()
        form.addRow("854 Cav power", self._power)
        form.addRow("Detuning", self._detuning)
        form.addRow("N_cycles", self._n_cycles)
        form.addRow("Number of runs", self._runs)

        self._run_btn = QtWidgets.QPushButton("Run ringdown")
        self._run_btn.clicked.connect(self._on_run_clicked)

        self._status = QtWidgets.QLabel("")
        self._status.setWordWrap(True)

        self._plot_ch3 = PlotWidget("Channel 3")
        self._plot_ch4 = PlotWidget("Channel 4")

        plots_layout = QtWidgets.QHBoxLayout()
        plots_layout.addWidget(self._plot_ch3, 1)
        plots_layout.addWidget(self._plot_ch4, 1)

        layout = QtWidgets.QVBoxLayout(central)
        layout.addLayout(form)
        layout.addWidget(self._run_btn)
        layout.addWidget(self._status)
        layout.addLayout(plots_layout, 1)

        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[RingdownWorker] = None

    def _set_running(self, running: bool) -> None:
        self._run_btn.setEnabled(not running)
        self._power.setEnabled(not running)
        self._detuning.setEnabled(not running)
        self._n_cycles.setEnabled(not running)
        self._runs.setEnabled(not running)
        if running:
            self._status.setText("Running…")

    def _read_inputs(self) -> RingdownInputs:
        power = float(self._power.text())
        detuning = float(self._detuning.text())
        n_cycles = int(float(self._n_cycles.text()))
        runs = int(float(self._runs.text()))
        return RingdownInputs(power=power, detuning=detuning, n_cycles=n_cycles, runs=runs)

    def _on_run_clicked(self) -> None:
        try:
            inputs = self._read_inputs()
        except Exception:
            self._status.setText("Invalid input(s).")
            return

        self._set_running(True)

        self._thread = QtCore.QThread(self)
        self._worker = RingdownWorker(inputs)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.failed.connect(self._on_worker_failed)

        # Cleanup
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._on_thread_finished)

        self._thread.start()

    @QtCore.pyqtSlot(dict)
    def _on_worker_finished(self, result: dict) -> None:
        ch3: ChannelPlotData = result.get("ch3")
        ch4: ChannelPlotData = result.get("ch4")
        summary: str = result.get("summary", "")

        if ch3 is not None:
            self._plot_ch3.set_data(ch3, "single_photon_chan_3")
        if ch4 is not None:
            self._plot_ch4.set_data(ch4, "single_photon_chan_4")

        self._status.setText(summary)

    @QtCore.pyqtSlot(str)
    def _on_worker_failed(self, tb: str) -> None:
        # Keep UX minimal: show error in the status label and clear plots.
        self._plot_ch3.set_error("Error")
        self._plot_ch4.set_error("Error")
        self._status.setText(tb)

    @QtCore.pyqtSlot()
    def _on_thread_finished(self) -> None:
        self._worker = None
        self._thread = None
        self._set_running(False)


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(1200, 700)
    w.show()
    return app.exec() if QT6 else app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
