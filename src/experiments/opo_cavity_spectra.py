import numpy as np
import os
from src.models.OPO_sde import OPO_SDE
from src.models.cavity_sde import KerrCavityPositiveP
from src.observables.spectra_sde import SpectraSDE
from tqdm import trange
from tqdm import tqdm


class OPOCavitySpectralSimulation:
    """
    Simulation complète OPO réel + cavité Kerr
    Reproduction de la Fig. 6 (spectres + moments)
    """

    def __init__(
        self,
        G_values,
        physical_params,
        numerical_params
    ):
        # Paramètres physiques
        self.gamma_s = physical_params.gamma_s
        self.gamma_c = physical_params.gamma_c
        self.U = physical_params.U
        self.alpha_dep = physical_params.alpha_dep

        # Paramètres numériques
        self.n_traj = numerical_params.n_traj
        self.dt_s = numerical_params.dt_s
        self.dt_c = numerical_params.dt_c

        self.G_values = G_values

        # Temps OPO
        self.t_opo_relax = 20 / self.gamma_s
        self.t_opo_sample = 1000 / self.gamma_s
        self.opo_sample_period = 1 / self.gamma_s

        # Temps cavité
        self.t_cav_relax = 100 / self.gamma_c
        self.t_cav_store = 200 / self.gamma_c
        self.cav_store_period = 0.5 / self.gamma_c

        # Analyse spectrale
        self.spectra = SpectraSDE(
            dt=self.cav_store_period,
            max_tau=200 / self.gamma_c,
            n_bins=2048
        )

        # Résultats
        self.all_spectra = []
        self.all_moments = []
        self.freqs = None

    # --------------------------------------------------
    # Run complet
    # --------------------------------------------------
    def run(self):
        for G in tqdm(self.G_values, desc="Pump beam sweep"):
            print(f"\n=== Pump beam G = {G:.4e} ===")

            spectra_traj = []

            for n in trange(self.n_traj, desc="Positive-P trajectories"):
                if n % 50 == 0:
                    print(f"  Trajectoire {n}/{self.n_traj}")

                # -------- OPO --------
                opo = OPO_SDE(self.gamma_s, G, self.dt_s)
                opo_samples = opo.run_and_sample(
                    t_relax=self.t_opo_relax,
                    t_sample=self.t_opo_sample,
                    sample_period=self.opo_sample_period
                )

                # -------- Cavité --------
                cavity = KerrCavityPositiveP(self.gamma_c, self.U, self.dt_c)
                cav_output = []

                for a_out, a_out_t in tqdm(opo_samples, desc="Cavity runs", leave=False):
                    out = cavity.run_stationary(
                        a_in=np.sqrt(0.7*self.gamma_s)*a_out + self.alpha_dep,  # the 0.7 comes from the coupling efficiency
                        a_in_t=np.sqrt(0.7*self.gamma_s)*a_out_t + self.alpha_dep,
                        t_relax=self.t_cav_relax,
                        t_store=self.t_cav_store,
                        store_period=self.cav_store_period
                    )
                    cav_output.append(out)

                cav_output = np.concatenate(cav_output)

                # -------- Spectre --------
                w, S = self.spectra.spectrum(cav_output)
                spectra_traj.append(S)

            # Moyenne Positive-P
            S_mean = np.mean(spectra_traj, axis=0)
            self.all_spectra.append(S_mean)

            # Moments spectraux
            moments = self.spectra.spectral_moments(w, S_mean, max_order=4)
            self.all_moments.append(moments)

            self.freqs = w

        self.all_spectra = np.array(self.all_spectra)
        self.all_moments = np.array(self.all_moments)

        return self.freqs, self.all_spectra, self.all_moments

    # --------------------------------------------------
    # Sauvegarde
    # --------------------------------------------------

    def save(self, prefix="opo_cavity", outdir="output_sde"):
        os.makedirs(outdir, exist_ok=True)

        np.save(f"{outdir}/{prefix}_frequencies.npy", self.freqs)
        np.save(f"{outdir}/{prefix}_spectra.npy", self.all_spectra)
        np.save(f"{outdir}/{prefix}_moments.npy", self.all_moments)
        np.save(f"{outdir}/{prefix}_G_values.npy", self.G_values)

