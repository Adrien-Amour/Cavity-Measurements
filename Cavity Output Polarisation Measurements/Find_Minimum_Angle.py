from adriq.Optomechanics import *
from adriq.Counters import *
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def stokes_from_extinction(qwp_angle, hwp_angle, extinction_channel):
    q = np.deg2rad(qwp_angle - 0.9)      # + clockwise
    h = np.deg2rad(hwp_angle - 24.2)     # + counterclockwise

    def R(a):
        return np.array([[np.cos(a), -np.sin(a)],
                            [np.sin(a),  np.cos(a)]])

    JQ = R(-q) @ np.diag([1, 1j]) @ R(q)
    JH = R(h)  @ np.diag([1, -1]) @ R(-h)

    output = np.array([1, 0] if extinction_channel == 4 else [0, 1])
    psi = JQ.conj().T @ JH.conj().T @ output

    H, V = psi
    return np.array([abs(H)**2 - abs(V)**2,
                        2*np.real(np.conj(H)*V),
                        2*np.imag(np.conj(H)*V)])

hwp_mount = Client(Thorlabs_Motorised_WaveplateMount)
qwp_mount = Client(Standa_Motorised_WaveplateMount)
polarisation_analyser = Client(Polarisation_Analyser)
qutau_reader = Client(QuTau_Reader)
 
qwp_guess, hwp_guess = 352.7, 36.6
qwp_angles = np.arange(-0.3, 0.31, 0.1) + qwp_guess
hwp_angles = np.arange(-0.4, 0.41, 0.1) + hwp_guess
samples, poll_dt, settle_time = 20, 0.05, 2.0
ch3_name, ch4_name = "single_photon_chan3", "single_photon_chan4"
 
def get_counts():
    ch3, ch4, last = [], [], None
    while len(ch3) < samples or len(ch4) < samples:
        times, counts = qutau_reader.get_counts()
        if times and times[-1] != last:
            last = times[-1]
            if counts.get(ch3_name): ch3.append(counts[ch3_name][-1])
            if counts.get(ch4_name): ch4.append(counts[ch4_name][-1])
        time.sleep(poll_dt)
    return np.mean(ch3[:samples]), np.mean(ch4[:samples])
 
ER, CH4 = {}, {}
qutau_reader.start_counting()
 

for q in tqdm(qwp_angles, desc="QWP"):
    qwp_mount.move_to(float(q))
    time.sleep(settle_time)
    ER[q], CH4[q] = [], []

    for h in hwp_angles:
        hwp_mount.move_to(float(h))
        time.sleep(settle_time)
        ch3, ch4 = get_counts()
        ER[q].append(ch4 / ch3 if ch3 else np.inf)
        CH4[q].append(ch4)

 
plt.figure()
for q in qwp_angles:
    plt.plot(hwp_angles, ER[q], label=f"{q:.1f}°")
plt.xlabel("HWP angle (°)")
plt.ylabel("Channel 4 / Channel 3s")
plt.title("Extinction ratio")
plt.legend(title="QWP", fontsize=6, ncol=2)
plt.tight_layout()
 
plt.figure()
for q in qwp_angles:
    plt.plot(hwp_angles, CH4[q], label=f"{q:.1f}°")
plt.xlabel("HWP angle (°)")
plt.ylabel("Average Channel 4 counts")
plt.title("Channel 4")
plt.legend(title="QWP", fontsize=6, ncol=2)
plt.tight_layout()
plt.show()


q, h, er = min((q, h, er) for q, vals in ER.items() for h, er in zip(hwp_angles, vals))
print(f"Min ER = {er:.3f} at QWP = {q:.1f}°, HWP = {h:.1f}°")

stokes = stokes_from_extinction(q, h, 4)
print(f"Stokes vector: {stokes}")

polarisation_analyser.set_X_basis()
ch3_x, ch4_x = get_counts()
polarisation_analyser.set_Y_basis()
ch3_y, ch4_y = get_counts()
polarisation_analyser.set_Z_basis()
ch3_z, ch4_z = get_counts()
print(f"X basis counts: {ch3_x:.0f}, {ch4_x:.0f}")
print(f"Y basis counts: {ch3_y:.0f}, {ch4_y:.0f}")
print(f"Z basis counts: {ch3_z:.0f}, {ch4_z:.0f}")


X, Y, Z = stokes
 
p3_x, p4_x = (1 + X)/2, (1 - X)/2
p3_y, p4_y = (1 + Y)/2, (1 - Y)/2
p3_z, p4_z = (1 - Z)/2, (1 + Z)/2   # ch3=L, ch4=R
 
r_x = (ch3_x/ch4_x) * (p4_x/p3_x)
r_y = (ch3_y/ch4_y) * (p4_y/p3_y)
r_z = (ch3_z/ch4_z) * (p4_z/p3_z)


print(f"Coupling ratio η3/η4: X={r_x:.3f}, Y={r_y:.3f}, Z={r_z:.3f}")
print(f"Mean η3/η4 = {np.mean([r_x, r_y, r_z]):.3f}")


polarisation_analyser.set_X_basis(invert=True)
ch3_x_i, ch4_x_i = get_counts()
polarisation_analyser.set_Y_basis(invert=True)
ch3_y_i, ch4_y_i = get_counts()
polarisation_analyser.set_Z_basis(invert=True)
ch3_z_i, ch4_z_i = get_counts()
print(f"X basis counts: {ch3_x_i:.0f}, {ch4_x_i:.0f}")
print(f"Y basis counts: {ch3_y_i:.0f}, {ch4_y_i:.0f}")
print(f"Z basis counts: {ch3_z_i:.0f}, {ch4_z_i:.0f}")


X, Y, Z = stokes
 
p3_x, p4_x = (1 - X)/2, (1 + X)/2
p3_y, p4_y = (1 - Y)/2, (1 + Y)/2
p3_z, p4_z = (1 + Z)/2, (1 - Z)/2   # ch3=L, ch4=R
 
r_x = (ch3_x_i/ch4_x_i) * (p4_x/p3_x)
r_y = (ch3_y_i/ch4_y_i) * (p4_y/p3_y)
r_z = (ch3_z_i/ch4_z_i) * (p4_z/p3_z)
 
print(f"Coupling ratio η3/η4: X={1/r_x:.3f}, Y={r_y:.3f}, Z={r_z:.3f}")
print(f"Mean η3/η4 = {np.mean([r_x, r_y, r_z]):.3f}")