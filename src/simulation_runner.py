import os
import json
import numpy as np
from joblib import Parallel, delayed
from src.microcavity_module import SpectralAnalysis


class SimulationRunner:
    def __init__(self, opo, cav, params, outdir):
        self.opo = opo
        self.cav = cav
        self.params = params
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)

    def run_one_G(self, gg, G, opo, cav, params):
        phis_out, cphitils_out = opo.run_for_G(G)

        Esourceave = (
            params["targeteddisplacement"]
            * cav.gammacav
            / (2*np.sqrt(cav.gammacav*cav.eps_p))
        )

        psis_in = (
            np.exp(1j*opo.angle)
            * np.sqrt(opo.gammaOPO*opo.eps1)
            * opo.weight * phis_out
            + Esourceave
        )

        cpsitils_in = (
            np.exp(-1j*opo.angle)
            * np.sqrt(opo.gammaOPO*opo.eps1)
            * opo.weight * cphitils_out
            + np.conj(Esourceave)
        )

        _, _, _, _, cavomegaarr, opticalspecarr, meanopticalspecarr, Convratio = \
            cav.simulate(psis_in, cpsitils_in, Esourceave)

        return gg, opticalspecarr, meanopticalspecarr, Convratio, cavomegaarr

    def run_simulation(self, n_jobs=8):
        print(">>> Running full simulation")

        Garr = self.opo.generate_G_array()
        self.opo.simulate_trajectories(Ntraj=self.params["Ntraj"])

        allopticalspecarr = np.zeros((self.opo.Nrelt, len(Garr)))
        convratioarr = np.zeros(len(Garr))

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(self.run_one_G)(gg, G, self.opo, self.cav, self.params)
            for gg, G in enumerate(Garr)
        )

        Nrelt = results[0][4].size  # le 0 correspond à gg=0 puis 4 à cavomegaarr
        allopticalspecarr = np.zeros((len(Garr), Nrelt))
        convratioarr = np.zeros(len(Garr))

        for gg, spec, meanspec, conv, omega in results:
            allopticalspecarr[gg, :] = meanspec
            convratioarr[gg] = conv
            cavomegaarr = omega  # same for all G

        # === SAVE ===
        np.save(os.path.join(self.outdir, "opticalspec.npy"), allopticalspecarr)
        np.savetxt(os.path.join(self.outdir, "Garr.csv"), Garr, delimiter=",")
        np.savetxt(os.path.join(self.outdir, "cavomegaarr.csv"), cavomegaarr, delimiter=",")
        np.savetxt(os.path.join(self.outdir, "convratio.csv"), convratioarr, delimiter=",")

        with open(os.path.join(self.outdir, "metadata.json"), "w") as f:
            json.dump(self.params, f, indent=2)

        print(">>> Simulation finished and saved")

        return {
            "Garr": Garr,
            "cavomegaarr": cavomegaarr,
            "opticalspec": allopticalspecarr,
            "convratio": convratioarr
        }
    
    def perform_analysis(self, data):
        analysis = SpectralAnalysis(
            data["opticalspec"], data["cavomegaarr"], data["Garr"],
            Ntrain=self.params.get("n_train", 10), ridgepar=1e-3, mode="absdiff", ref_index=0
        )
        analysis.compute_moments()
        analysis.ridge_regression()
        analysis.predict()
        analysis_results = analysis.get_results()
        nrmse = analysis.compute_nrmse()
        print(f"\n📉 Regression performance for G: " f"{100*nrmse:.3f}% NRMSE")

        # Save analysis results
        np.savetxt(os.path.join(self.outdir, "G_pred.csv"), analysis_results["G_pred"], delimiter=",")
        np.savetxt(os.path.join(self.outdir, "moments.csv"), analysis_results["moments"], delimiter=",")
        np.savetxt(os.path.join(self.outdir, "G_true.csv"), analysis_results["G_true"], delimiter=",") 
        print(">>> Analysis finished and saved")

        return {
            "G_pred": analysis_results["G_pred"],
            "moments": analysis_results["moments"],
            "G_true": analysis_results["G_true"],
        }
