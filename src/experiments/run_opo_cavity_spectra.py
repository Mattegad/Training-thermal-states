import numpy as np
import matplotlib.pyplot as plt

from src.experiments.opo_cavity_spectra import OPOCavitySpectralSimulation
from src.config import physical_params, numerical_params


# --------------------------------------------------
# Valeurs de G (comme dans l’article)
# --------------------------------------------------
G_values = (np.linspace(0.0, 0.9, 6) * 4.5) * 1e3 / (2.42e14)

# --------------------------------------------------
# Simulation
# --------------------------------------------------
simu = OPOCavitySpectralSimulation(
    G_values=G_values,
    physical_params=physical_params,
    numerical_params=numerical_params
)

w, spectra, moments = simu.run()
simu.save(prefix="fig6")

# --------------------------------------------------
# PLOTS — Fig. 6a
# --------------------------------------------------
plt.figure(figsize=(7, 4))
for i, G in enumerate(G_values):
    plt.plot(w, spectra[i], label=f"G/Gth={G/4.5:.2f}")

plt.xlabel(r"$\omega$")
plt.ylabel(r"$S(\omega)$")
plt.title("Spectres d’émission – OPO réel")
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------------------------------
# Moments spectraux
# --------------------------------------------------
plt.figure(figsize=(7, 4))
for m in range(1, 5):
    plt.plot(
        G_values / 4.5,
        moments[:, m],
        "o-",
        label=fr"$M_{m}$"
    )

plt.xlabel(r"$G / G_{\mathrm{th}}$")
plt.ylabel("Moments spectraux normalisés")
plt.legend()
plt.tight_layout()
plt.show()

print("\n✔ Simulation Fig. 6 terminée")
