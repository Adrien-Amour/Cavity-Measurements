from adriq.Counters import *

from pyqtgraph.Qt import QtWidgets, QtCore
import pyqtgraph as pg
from collections import deque
import time
import sys

# ==========================================================
# Configuration
# ==========================================================

history_length = 500
update_period_ms = 100

ch3_name = "single_photon_chan3"
ch4_name = "single_photon_chan4"

# ==========================================================
# Connect to QuTau
# ==========================================================

qutau_reader = Client(QuTau_Reader)
qutau_reader.start_counting()

last_qutau_time = None
t0 = time.time()

# ==========================================================
# Data buffers
# ==========================================================

times = deque(maxlen=history_length)

ch3_data = deque(maxlen=history_length)
ch4_data = deque(maxlen=history_length)

f3_data = deque(maxlen=history_length)
f4_data = deque(maxlen=history_length)

# ==========================================================
# Qt Application
# ==========================================================

app = QtWidgets.QApplication(sys.argv)

window = QtWidgets.QWidget()
window.setWindowTitle("QuTau Live Monitor")
window.resize(1200, 800)

layout = QtWidgets.QVBoxLayout(window)

# ----------------------------------------------------------
# Checkboxes
# ----------------------------------------------------------

checkbox_layout = QtWidgets.QHBoxLayout()

ch3_checkbox = QtWidgets.QCheckBox("Channel 3")
ch3_checkbox.setChecked(True)

ch4_checkbox = QtWidgets.QCheckBox("Channel 4")
ch4_checkbox.setChecked(True)

f3_checkbox = QtWidgets.QCheckBox("3/(3+4)")
f3_checkbox.setChecked(True)

f4_checkbox = QtWidgets.QCheckBox("4/(3+4)")
f4_checkbox.setChecked(True)

checkbox_layout.addWidget(ch3_checkbox)
checkbox_layout.addWidget(ch4_checkbox)
checkbox_layout.addWidget(f3_checkbox)
checkbox_layout.addWidget(f4_checkbox)

checkbox_layout.addStretch()

layout.addLayout(checkbox_layout)

# ----------------------------------------------------------
# Plot widget
# ----------------------------------------------------------

plot_widget = pg.PlotWidget()
plot_widget.setLabel("left", "Value")
plot_widget.setLabel("bottom", "Time", units="s")
plot_widget.showGrid(x=True, y=True)

legend = plot_widget.addLegend()

layout.addWidget(plot_widget)

# ----------------------------------------------------------
# Curves
# ----------------------------------------------------------

curve_ch3 = plot_widget.plot(
    name="Channel 3",
    pen=pg.mkPen((255, 0, 0), width=2)
)

curve_ch4 = plot_widget.plot(
    name="Channel 4",
    pen=pg.mkPen((0, 0, 255), width=2)
)

curve_f3 = plot_widget.plot(
    name="3/(3+4)",
    pen=pg.mkPen((0, 150, 0), width=2)
)

curve_f4 = plot_widget.plot(
    name="4/(3+4)",
    pen=pg.mkPen((255, 165, 0), width=2)
)

# ----------------------------------------------------------
# Checkbox connections
# ----------------------------------------------------------

ch3_checkbox.toggled.connect(curve_ch3.setVisible)
ch4_checkbox.toggled.connect(curve_ch4.setVisible)
f3_checkbox.toggled.connect(curve_f3.setVisible)
f4_checkbox.toggled.connect(curve_f4.setVisible)

# ==========================================================
# Update function
# ==========================================================

def update():
    global last_qutau_time

    try:
        q_times, counts = qutau_reader.get_counts()

        if not q_times:
            return

        current_qutau_time = q_times[-1]

        if current_qutau_time == last_qutau_time:
            return

        last_qutau_time = current_qutau_time

        ch3_hist = counts.get(ch3_name, [])
        ch4_hist = counts.get(ch4_name, [])

        ch3 = ch3_hist[-1] if ch3_hist else 0
        ch4 = ch4_hist[-1] if ch4_hist else 0

        total = ch3 + ch4

        f3 = ch3 / total if total > 0 else 0
        f4 = ch4 / total if total > 0 else 0

        t = time.time() - t0

        times.append(t)

        ch3_data.append(ch3)
        ch4_data.append(ch4)

        f3_data.append(f3)
        f4_data.append(f4)

        curve_ch3.setData(times, ch3_data)
        curve_ch4.setData(times, ch4_data)

        curve_f3.setData(times, f3_data)
        curve_f4.setData(times, f4_data)

    except Exception as e:
        print(e)

# ==========================================================
# Timer
# ==========================================================

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(update_period_ms)

# ==========================================================
# Cleanup on close
# ==========================================================

def cleanup():
    print("Stopping QuTau counting...")
    qutau_reader.stop_counting()

app.aboutToQuit.connect(cleanup)

# ==========================================================
# Run application
# ==========================================================

window.show()
sys.exit(app.exec_())