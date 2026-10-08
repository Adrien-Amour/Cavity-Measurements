import tkinter as tk
from tkinter import ttk
from decimal import Decimal
from adriq.RedLabs_Dac import *
from adriq.Custom_Tkinter import CustomSpinbox
from adriq.Servers import Client

# Initialize DAC client

# Test application
class PiezoControlApp(tk.Tk, Redlabs_Dac):
    def __init__(self):
        super().__init__()
        DAC = Client(Redlabs_DAC)

        self.title("Piezo Voltage Control") 
        self.geometry("300x150")

        # Frame for label and spinbox
        frame = tk.Frame(self)
        frame.pack(pady=10)

        # Label
        label = tk.Label(frame, text="Piezo Voltage:")
        label.pack(side=tk.LEFT, padx=5)

        # CustomSpinbox
        self.spinbox = CustomSpinbox(frame, from_=-10.0, to=10.0, initial_value=0.0, increment=0.1, width=10)
        self.spinbox.set_callback(self.set_piezo_voltage)
        self.spinbox.pack(side=tk.LEFT, padx=5)

        # Start periodic voltage reading
        self.update_spinbox_with_current_voltage()

    def set_piezo_voltage(self, voltage):
        try:
            DAC.set_piezo_voltage(voltage, step_size=0.005, rate=5)
            print(f"Piezo voltage set to: {voltage} V")
        except Exception as e:
            print(f"Error setting piezo voltage: {e}")

    def update_spinbox_with_current_voltage(self):
        """Periodically update the spinbox with the current piezo voltage."""
        try:
            current_voltage = DAC.get_piezo_v()
            self.spinbox.var.set(f"{current_voltage:.3f}")  # Update the spinbox's StringVar directly
        except Exception as e:
            print(f"Error reading piezo voltage: {e}")

        # Schedule the next update after 1 second (1000 ms)
        self.after(1000, self.update_spinbox_with_current_voltage)


