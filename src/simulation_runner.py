import os
import json
import numpy as np


class SimulationRunner:
    def __init__(self, opo, cav, params, outdir):
        self.opo = opo
        self.cav = cav
        self.params = params
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)

    def run_simulation(self):
        print(">>> Running full simulation")

        Garr = self.opo.generate_G_array()
        self.opo.simulate_trajectories(Ntraj=self.params["Ntraj"])

        allopticalspecarr = np.zeros((self.opo.Nrelt, len(Garr)))
        convratioarr = np.zeros(len(Garr))

        for gg, G in enumerate(Garr):
            print(f"G = {G:.3f} ({gg+1}/{len(Garr)})")

            phis_out, cphitils_out = self.opo.run_for_G(G)

            Esourceave = (
                self.params["targeteddisplacement"]
                * self.cav.gammacav
                / (2*np.sqrt(self.cav.gammacav*self.cav.eps_p))
            )

            psis_in = (
                np.exp(1j*self.opo.angle)
                * np.sqrt(self.opo.gammaOPO*self.opo.eps1)
                * self.opo.weight * phis_out
                + Esourceave
            )

            cpsitils_in = (
                np.exp(-1j*self.opo.angle)
                * np.sqrt(self.opo.gammaOPO*self.opo.eps1)
                * self.opo.weight * cphitils_out
                + np.conj(Esourceave)
            )

            _, _, _, _, cavomegaarr, opticalspecarr, Convratio = \
                self.cav.simulate(psis_in, cpsitils_in, Esourceave)

            allopticalspecarr[:, gg] = opticalspecarr
            convratioarr[gg] = Convratio

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
            "convratio": convratioarr,
        }
