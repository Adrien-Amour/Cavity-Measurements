import time
import numpy as np
import pyvisa
from concurrent.futures import ThreadPoolExecutor
from pyvisa.errors import VisaIOError
from adriq.Optomechanics import *
from adriq.Servers import Client
from adriq.ad9910 import *

N_READS, INTERNAL_AVERAGES, SETTLE_TIME = 5, 3000, 2.0
polarisation_analyser = Client(Polarisation_Analyser)
cav_laser = create_laser_objects(
    r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\dds_config.cfg",
    include_lasers=["854 Cav"]
)[0]


def connect_power_meters():
    rm = pyvisa.ResourceManager()
    resources = list(rm.list_resources())
    print("\nVISA resources:")
    for i, r in enumerate(resources):
        try:
            with rm.open_resource(r) as x:
                x.timeout = 1000
                idn = x.query("*IDN?").strip()
        except Exception:
            idn = "no IDN response"
        print(f"[{i}] {r}  {idn}")

    while True:
        try:
            idx = [int(x) for x in input("Select PM1 and PM2 indices (e.g. 0 1): ").replace(",", " ").split()]
            if len(idx) == 2 and len(set(idx)) == 2 and all(0 <= i < len(resources) for i in idx):
                break
        except ValueError:
            pass
        print("Enter two different valid indices.")

    meters = []
    for n, i in enumerate(idx, 1):
        pm = rm.open_resource(resources[i])
        pm.timeout = 15000
        pm.write(f"SENS:AVER:COUN {INTERNAL_AVERAGES}")
        print(f"PM{n}: {resources[i]}  averaging set to {INTERNAL_AVERAGES}")
        meters.append(pm)
    return rm, meters


def read_power(pm):
    for attempt in range(5):
        try:
            return float(pm.query("READ?"))
        except VisaIOError:
            if attempt == 4:
                raise
            time.sleep(0.1)


def mean_sem(x):
    return np.mean(x), np.std(x, ddof=1) / np.sqrt(len(x))


rm, meters = connect_power_meters()
executor = ThreadPoolExecutor(max_workers=2)


def acquire(n):
    return np.array([list(executor.map(read_power, meters)) for _ in range(n)])

def measure_fraction():
    cav_laser.update_detuning(0, 1, 0)
    time.sleep(SETTLE_TIME/4)
    signal = acquire(N_READS)
    cav_laser.update_detuning(0, 0, 0)
    time.sleep(SETTLE_TIME/4)
    bg = acquire(1)[0]
    p = signal - bg
    total = p.sum(axis=1)
    if np.any(total <= 0): raise RuntimeError("Non-positive background-subtracted total power.")
    f = p[:, 0] / total
    return f, 1 - f, p, bg


def set_basis(basis, invert=False):
    fn = getattr(polarisation_analyser, f"set_{basis}_basis")
    time.sleep(SETTLE_TIME)
    fn(invert=True) if invert else fn()


def infer_population(basis):
    set_basis(basis)
    f1_all, _, p, bg = measure_fraction()
    set_basis(basis, True)
    g1_all, _, q, bg_inv = measure_fraction()

    f1, f1_err = mean_sem(f1_all)
    g1, g1_err = mean_sem(g1_all)
    f2, g2 = 1 - f1, 1 - g1
    R = np.sqrt((f1 * g2) / (f2 * g1))
    eta = np.sqrt((f1 * g1) / (f2 * g2))
    R_err = 0.5 * R * np.sqrt((f1_err / (f1 * f2))**2 + (g1_err / (g1 * g2))**2)
    P1, P2 = R / (1 + R), 1 / (1 + R)
    S, S_err = (R - 1) / (R + 1), 2 * R_err / (R + 1)**2
    print(f"Basis {basis}: f1={f1:.6f}+/-{f1_err:.6f}, g1={g1:.6f}+/-{g1_err:.6f}, R={R:.4f}+/-{R_err:.4f}, P1={P1:.6f}, P2={P2:.6f}, eta={eta:.6f}")
    a, b = {"X": ("H", "V"), "Y": ("D", "A"), "Z": ("L", "R")}[basis]
    print(f"  normal:   PM1={p[:,0].mean():.6e} W, PM2={p[:,1].mean():.6e} W, bg=({bg[0]:.3e},{bg[1]:.3e}) W, f1={f1:.6f}+/-{f1_err:.6f}")
    print(f"  inverted: PM1={q[:,0].mean():.6e} W, PM2={q[:,1].mean():.6e} W, bg=({bg_inv[0]:.3e},{bg_inv[1]:.3e}) W, g1={g1:.6f}+/-{g1_err:.6f}")
    print(f"  P_{a}/P_{b}={R:.4f}+/-{R_err:.4f}, P_{a}={P1:.6f}, P_{b}={P2:.6f}, eta_PM1/eta_PM2={eta:.6f}\n")
    return S, S_err, eta


try:
    results = {b: (print(f"Measuring {b} basis..."), infer_population(b))[1] for b in "XYZ"}
finally:
    cav_laser.update_detuning(0, 0, 0)
    executor.shutdown(wait=True)
    for pm in meters:
        pm.close()
    rm.close()

S = np.array([results[b][0] for b in "XYZ"])
S_err = np.array([results[b][1] for b in "XYZ"])
etas = np.array([results[b][2] for b in "XYZ"])
print(f"Stokes = {S} +/- {S_err}")
print(f"|S| = {np.linalg.norm(S):.6f}")
print(f"eta_HV, eta_DA, eta_LR = {etas}")

Plot_Stokes_Vector(S)