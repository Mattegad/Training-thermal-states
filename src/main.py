from microcavity_module import OPOField, Microcavity
from config import PhysicalParams, NumericalParams
from simulation_runner import SimulationRunner
from data_manager import DataManager
from plotter import Plotter

if __name__ == "__main__":

    outdir = "data/run_GdepforNtraj1e3"

    opo = OPOField(
        gammaOPO=PhysicalParams.gammaOPO,
        Ntest=NumericalParams.Ntest,
        eps1=PhysicalParams.eps1,
        weight=PhysicalParams.weight,
        angle=PhysicalParams.angle,
    )

    cav = Microcavity(
        U=PhysicalParams.Ufr,
        gammacav=PhysicalParams.gammacav,
        eps_p=PhysicalParams.eps_p,
        unitrescaling=1000.0,
        hbar=PhysicalParams.hbar,
    )

    params = {
        "Ntraj": NumericalParams.Ntraj,
        "targeteddisplacement": PhysicalParams.targeted_displacement,
    }

    runner = SimulationRunner(opo, cav, params, outdir)
    manager = DataManager(outdir, runner)

    data = manager.get_data()

    plotter = Plotter(data)
    plotter.plot_spectra()
    plotter.plot_centered()
