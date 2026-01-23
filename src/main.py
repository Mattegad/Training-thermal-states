import numpy as np
import matplotlib.pyplot as plt
from microcavity_module import OPOField, Microcavity, SpectralAnalysis
from config import PhysicalParams, NumericalParams

class SimulationPipeline:
    """Pipeline complet : OPO -> Microcavity -> Analyse"""
    def __init__(self):
        # Paramètres généraux
        self.Ntraj = NumericalParams.Ntraj
        self.targeted_displacement = PhysicalParams.targeted_displacement
        self.filename_base = "GdepforNtraj1e3_Esourceav0pt5gammaovepsp"

        # Paramètres OPO
        self.gammaOPO = PhysicalParams.gammaOPO
        self.Ntest = NumericalParams.Ntest
        self.eps1 = PhysicalParams.eps1
        self.weight = PhysicalParams.weight
        self.angle = PhysicalParams.angle

        # Paramètres microcavité
        self.U = PhysicalParams.Ufr
        self.gammacav = PhysicalParams.gammacav
        self.eps_p = PhysicalParams.eps_p
        self.hbar = PhysicalParams.hbar
        self.unitrescaling = 1000.0

        # Initialisation OPO et microcavité
        self.opo = OPOField(
            gammaOPO=self.gammaOPO,
            Ntest=self.Ntest,
            eps1=self.eps1,
            weight=self.weight,
            angle=self.angle
        )
        self.cav = Microcavity(
            U=self.U,
            gammacav=self.gammacav,
            eps_p=self.eps_p,
            unitrescaling=self.unitrescaling,
            hbar=self.hbar
        )

    def run_simulations(self):
        # --- OPO ---
        Garr = self.opo.generate_G_array()
        self.opo.simulate_trajectories(Ntraj=self.Ntraj)

        Nrelt = self.opo.Nrelt
        Nstates = self.opo.Nstates

        # --- Microcavité ---
        allopticalspecarr = np.zeros((Nrelt, Nstates))
        convratioarr = np.zeros(Nstates)

        Esourceave = self.targeted_displacement * self.gammacav / (2*np.sqrt(self.gammacav*self.eps_p))

        for gg, G in enumerate(Garr):
            print(f"\n--- Running state {gg+1}/{Nstates}, G={G:.3f} ---")
            
            phis_out, cphitils_out = self.opo.run_for_G(G, y0=None, numstepsperoutput=100)

            # Couplage OPO → microcavité
            psis_in = np.exp(1j*self.angle) * np.sqrt(self.gammaOPO*self.eps1) * (self.weight*phis_out) + Esourceave
            cpsitils_in = np.exp(-1j*self.angle) * np.sqrt(self.gammaOPO*self.eps1) * (self.weight*cphitils_out) + np.conj(Esourceave)

            # Simulation microcavité
            psi_out, cpsitil_out, cavtimearray, meancavg1, cavomegaarr, opticalspecarr, Convratio = \
                self.cav.simulate(psis_in, cpsitils_in, Esourceave, f"{self.filename_base}_G={G:.3f}")

            allopticalspecarr[:, gg] = opticalspecarr
            convratioarr[gg] = Convratio

            # Sauvegarde CSV
            np.savetxt(f"{self.filename_base}_polspec.csv", opticalspecarr.reshape(1,-1),
                       delimiter=",", newline=",", fmt='%.6e', footer="\n", comments="")

        self.allopticalspecarr = allopticalspecarr
        self.cavomegaarr = cavomegaarr
        self.Garr = Garr
        self.convratioarr = convratioarr
        print("\n--- Microcavity simulations complete ---")

    def run_analysis(self, Ntrain=10, ridgepar=0.01):
        # Transpose pour correspondre à Nstates x Nrelt
        spectra = self.allopticalspecarr.T
        analysis = SpectralAnalysis(spectra, self.cavomegaarr, self.Garr, Ntrain=Ntrain, ridgepar=ridgepar, analytic_reference=False)
        analysis.compute_moments()
        analysis.ridge_regression()
        y_pred = analysis.predict()

        # --- Plot spectres absolus pour le train ---
        plt.figure(figsize=(10,6))
        for i in range(analysis.Ntrain):
            plt.plot(self.cavomegaarr, spectra[i,:], label=f"G={self.Garr[i]:.2f}")
        plt.xlabel(r"$\omega$ (µeV)")
        plt.ylabel("Intensity")
        plt.yscale("log")
        plt.legend()
        plt.title("Spectres absolus (train)")
        plt.show()

        # --- Plot moments et prédictions ---
        plt.figure(figsize=(12,6))
        for i, M in enumerate(analysis.moments):
            plt.subplot(2,3,i+1)
            plt.scatter(self.Garr, M)
            plt.xlabel("G")
            plt.ylabel(f"M_{i}")
        plt.subplot(2,3,6)
        absrms = np.sqrt(np.nanmean((y_pred - self.Garr)**2))
        relrms = np.sqrt(np.nanmean((y_pred - self.Garr)**2)/np.nanmean(self.Garr**2))
        plt.text(0,0.7,f"Prediction error (abs): {absrms:.3f}, (rel): {100*relrms:.2f}%")
        plt.suptitle("Moments & Ridge regression")
        plt.tight_layout()
        plt.show()

        # --- Comparaison prédiction vs réalité ---
        plt.figure()
        plt.scatter(self.Garr, self.Garr, label="G true")
        plt.scatter(self.Garr, y_pred, label="G predicted")
        plt.xlabel("True G")
        plt.ylabel("Predicted G")
        plt.legend()
        plt.title("Prediction vs True")
        plt.show()

        # --- Plots finaux centrés et relatifs ---
        spectre_ref = self.allopticalspecarr[:, 0]
        spectres_centres = self.allopticalspecarr - spectre_ref[:, None]
        spectres_relatives = spectres_centres / spectre_ref[:, None]

        plt.figure(figsize=(12,4))

        # 1️⃣ Spectres absolus
        plt.subplot(1,3,1)
        for gg in range(len(self.Garr)):
            plt.plot(self.cavomegaarr, self.allopticalspecarr[:, gg], label=f"G={self.Garr[gg]:.2f}")
        plt.xlabel(r"$\omega$ (µeV)")
        plt.ylabel(r"$S(\omega)$")
        plt.title("Spectres absolus")
        plt.legend(fontsize=8)

        # 2️⃣ Différence absolue
        plt.subplot(1,3,2)
        for gg in range(len(self.Garr)):
            plt.plot(self.cavomegaarr, spectres_centres[:, gg], label=f"G={self.Garr[gg]:.2f}")
        plt.xlabel(r"$\omega$ (µeV)")
        plt.ylabel(r"$\Delta S(\omega)$")
        plt.title("Différence au cas G=0")
        plt.legend(fontsize=8)

        # 3️⃣ Variation relative
        plt.subplot(1,3,3)
        for gg in range(len(self.Garr)):
            plt.plot(self.cavomegaarr, spectres_relatives[:, gg], label=f"G={self.Garr[gg]:.2f}")
        plt.xlabel(r"$\omega$ (µeV)")
        plt.ylabel(r"$(S_G - S_0)/S_0$")
        plt.title("Variation relative")
        plt.legend(fontsize=8)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    pipeline = SimulationPipeline()
    pipeline.run_simulations()
    pipeline.run_analysis(Ntrain=10, ridgepar=0.01)
    print("\n✔ Full simulation and analysis complete")