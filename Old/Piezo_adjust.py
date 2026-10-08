from adriq.RedLabs_Dac import *
from adriq.Servers import Client

DAC = Client(Redlabs_DAC)

# Request user input for the voltage
try:
    voltage = float(input("Enter the desired piezo voltage: "))
    DAC.set_piezo_voltage(voltage, step_size=0.005, rate=5)
except ValueError:
    print("Invalid input. Please enter a numeric value.")
