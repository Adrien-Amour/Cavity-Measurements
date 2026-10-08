from adriq.Optomechanics import *
from adriq.Counters import *
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

hwp_mount = Client(Thorlabs_Motorised_WaveplateMount)
qwp_mount = Client(Standa_Motorised_WaveplateMount)



angles = np.arange(0, 180, 5) + 0.9
hwp_mount.move_to(69.2)
qwp_mount.move_to(angles[0])
time.sleep(2.0)  # Wait for the waveplate to settle
qutau_reader = Client(QuTau_Reader)

raw_counts_csv = "QWP_Scan_axis2_June_2026.csv"

samples_per_angle = 20
poll_dt = 0.05
settle_time = 2.0

ch3_name = "single_photon_chan3"
ch4_name = "single_photon_chan4"

angle_list = []
ch3_avg_list = []
ch4_avg_list = []

qutau_reader.start_counting()

try:
    with open(raw_counts_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Angle (deg)",
            "Channel 3 Avg Count Rate",
            "Channel 4 Avg Count Rate",
        ])

        for angle in tqdm(angles, desc="Scanning Angles", unit="step"):
            qwp_mount.move_to(float(angle))
            time.sleep(settle_time)

            fresh_ch3 = []
            fresh_ch4 = []
            last_time = None

            while len(fresh_ch3) < samples_per_angle or len(fresh_ch4) < samples_per_angle:
                times, counts = qutau_reader.get_counts()

                if not times:
                    time.sleep(poll_dt)
                    continue

                current_time = times[-1]
                if current_time == last_time:
                    time.sleep(poll_dt)
                    continue

                last_time = current_time

                ch3_hist = counts.get(ch3_name, [])
                ch4_hist = counts.get(ch4_name, [])

                if ch3_hist:
                    fresh_ch3.append(ch3_hist[-1])
                if ch4_hist:
                    fresh_ch4.append(ch4_hist[-1])

                time.sleep(poll_dt)

            ch3_avg = float(np.mean(fresh_ch3[:samples_per_angle]))
            ch4_avg = float(np.mean(fresh_ch4[:samples_per_angle]))

            writer.writerow([angle, ch3_avg, ch4_avg])
            file.flush()

            angle_list.append(angle)
            ch3_avg_list.append(ch3_avg)
            ch4_avg_list.append(ch4_avg)

finally:
    qutau_reader.stop_counting()

plt.figure(figsize=(8, 5))
plt.plot(angle_list, ch3_avg_list, "o-", label="Channel 3")
plt.plot(angle_list, ch4_avg_list, "o-", label="Channel 4")
plt.xlabel("Angle (deg)")
plt.ylabel("Average Count Rate")
plt.title("QuTau Count Rate vs Waveplate Angle")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()