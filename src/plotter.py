import numpy as np
import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, data):
        self.Garr = data["Garr"]
        self.omega = data["cavomegaarr"]
        self.spec = data["opticalspec"]

    def plot_spectra(self):
        plt.figure(figsize=(10,6))
        for gg, G in enumerate(self.Garr):
            plt.plot(self.omega, self.spec[:, gg], label=f"G={G:.2f}")
        plt.yscale("log")
        plt.xlabel(r"$\omega$")
        plt.ylabel(r"$S(\omega)$")
        plt.legend(fontsize=8)
        plt.show()

    def plot_centered(self):
        ref = self.spec[:, 0]
        centered = self.spec - ref[:, None]

        plt.figure(figsize=(10,4))
        for gg, G in enumerate(self.Garr):
            plt.plot(self.omega, centered[:, gg], label=f"G={G:.2f}")
        plt.xlabel(r"$\omega$")
        plt.ylabel(r"$\Delta S$")
        plt.legend(fontsize=8)
        plt.show()
