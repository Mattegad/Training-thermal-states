import numpy as np
import matplotlib.pyplot as plt

from models.OPO_sde import OPO_SDE
from models.cavity_sde import KerrCavityPositiveP
from observables.spectra_sde import SpectraSDE
from src.config import physical_params, numerical_params


G_values = np.linspace(0.0, 0.9, 6) * 4.5  # |G| jusqu’au seuil (en ms^-1)

# ------------------------------------------------------------
# Temps normalisés
# ------------------------------------------------------------

gamma_s = physical_params.gamma_s
gamma_c = physical_params.gamma_c
U = physical_params.U
n_traj = numerical_params.n_traj
dt_s = numerical_params.dt_s
dt_c = numerical_params.dt_c

t_opo_relax = 20 / gamma_s
t_opo_sample = 1000 / gamma_s
opo_sample_period = 1 / gamma_s

t_cav_relax = 100 / gamma_c
t_cav_store = 200 / gamma_c
cav_store_period = 0.5 / gamma_c

# ============================================================
# 2) Analyse spectrale
# ============================================================

spectra = SpectraSDE(
    dt=cav_store_period,
    max_tau=200 / gamma_c,
    n_bins=2048
)

# ============================================================
# 3) Boucle principale (Positive-P)
# ============================================================

all_moments = []
all_spectra = []

for G in G_values:
    print(f"\n=== Gain G = {G:.3f} ===")

    spectra_traj = []

    for n in range(n_traj):
        if n % 50 == 0:
            print(f"  Trajectoire {n}/{n_traj}")

        # ---------------- OPO ----------------
        opo = OPO_SDE(gamma_s, G, dt_s)
        opo_samples = opo.run_and_sample(
            t_relax=t_opo_relax,
            t_sample=t_opo_sample,
            sample_period=opo_sample_period
        )

        # ---------------- Cavité ----------------
        cavity = KerrCavityPositiveP(gamma_c, U, dt_c)
        cav_output = []

        for a_in, a_in_t in opo_samples:
            out = cavity.run_stationary(
                a_in=a_in,
                a_in_t=a_in_t,
                t_relax=t_cav_relax,
                t_store=t_cav_store,
                store_period=cav_store_period
            )
            cav_output.append(out)

        cav_output = np.concatenate(cav_output)

        # ---------------- Spectre ----------------
        w, S = spectra.spectrum(cav_output)
        spectra_traj.append(S)

    # -------- Moyenne Positive-P --------
    S_mean = np.mean(spectra_traj, axis=0)
    all_spectra.append(S_mean)

    # -------- Moments spectraux --------
    moments = spectra.spectral_moments(w, S_mean, max_order=4)
    all_moments.append(moments)

# ============================================================
# 4) Conversion en tableaux
# ============================================================

all_spectra = np.array(all_spectra)
all_moments = np.array(all_moments)

# ============================================================
# 5) PLOTS — Fig. 6a (spectres différentiels)
# ============================================================

plt.figure(figsize=(7, 4))
for i, G in enumerate(G_values):
    plt.plot(w, all_spectra[i], label=f"G/Gth={G/4.5:.2f}")

plt.xlabel(r"$\omega$")
plt.ylabel(r"$S(\omega)$")
plt.title("Spectres d’émission – OPO réel (Fig. 6a)")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 6) PLOTS — Moments spectraux (Fig. 6a inset)
# ============================================================

plt.figure(figsize=(7, 4))
for m in range(1, 5):
    plt.plot(
        G_values / 4.5,
        all_moments[:, m],
        "o-",
        label=fr"$M_{m}$"
    )

plt.xlabel(r"$G / G_{\mathrm{th}}$")
plt.ylabel("Moments spectraux normalisés")
plt.title("Moments spectraux du spectre d’émission")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 7) Sauvegarde des données
# ============================================================

np.save("frequencies.npy", w)
np.save("spectra_mean.npy", all_spectra)
np.save("spectral_moments.npy", all_moments)
np.save("G_values.npy", G_values)

print("\n✔ Simulation complète terminée")
