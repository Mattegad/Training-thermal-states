# train.py
import os
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import Ridge
from qutip import destroy

from src.config import physical_params, numerical_params, squeezing_params, spec_config
from src.hamiltonian import HamiltonianFactory
from src.dissipation import DissipationFactory
from src.steady_state import SteadyStateSolver
from src.spectrum import SpectrumCalculator
from src.moments import MomentsCalculator
from src.plot_utils import Plotter
from src.utils import normalize_moments


class Trainer:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # file names depend ONLY on the type of spectrum
        tag = spec_config.which
        self.train_file = os.path.join(output_dir, f"training_data_{tag}.npz")
        self.test_file  = os.path.join(output_dir, f"testing_data_{tag}.npz")

        self.moments_calc = MomentsCalculator()
        self.plotter      = Plotter()

        self.data = None

    # ------------------------------------------------------------------
    # Spectrum for given r
    # ------------------------------------------------------------------
    def generate_spectrum(self, r):
        """Compute stationary spectrum for given r for the selected 'which'."""

        N = numerical_params.Ncut
        a = destroy(N)

        # Hamiltonian
        H = HamiltonianFactory.make("kerr", a, U_val=physical_params.U)

        # Dissipation
        c_ops = DissipationFactory.make(
            "squeezed", a,
            alpha=squeezing_params.alphaD_default,
            r=r,
            theta=squeezing_params.theta_default,
            nbar=squeezing_params.nbar_default,
            gamma=physical_params.gamma_c
        )

        # steady state
        rho_ss = SteadyStateSolver(H, c_ops).compute()

        # frequencies
        wlist = np.linspace(numerical_params.wmin,
                           numerical_params.wmax,
                           numerical_params.n_w)

        # spectrum calculator
        calc = SpectrumCalculator(
            H, a, rho_ss, c_ops,
            wlist=wlist,
            which=spec_config.which      # <———— KEY POINT
        )

        return calc.compute()  # returns (w, spectrum)

    # ------------------------------------------------------------------
    # Compute dataset for this 'which'
    # ------------------------------------------------------------------
    def compute_dataset(self):
        r_train = np.linspace(0.0, 2.0, numerical_params.n_train)
        r_test  = np.random.uniform(0.0, 2.0, numerical_params.n_test)

        result_train = {"r": r_train, "omega": [], "S": [], "M": []}
        result_test  = {"r": r_test,  "omega": [], "S": [], "M": []}

        print(f"\n📘 Computing TRAINING data for spectrum '{spec_config.which}'")
        for r in tqdm(r_train):
            w, S = self.generate_spectrum(r)
            result_train["omega"].append(w)
            result_train["S"].append(S)
            result_train["M"].append(self.moments_calc.compute(w, S))

        print(f"\n📗 Computing TEST data for spectrum '{spec_config.which}'")
        for r in tqdm(r_test):
            w, S = self.generate_spectrum(r)
            result_test["omega"].append(w)
            result_test["S"].append(S)
            result_test["M"].append(self.moments_calc.compute(w, S))

        np.savez_compressed(self.train_file, **result_train)
        np.savez_compressed(self.test_file,  **result_test)

        print(f"💾 Data saved in '{self.output_dir}'")
        self.data = {"train": result_train, "test": result_test}
        return self.data

    # ------------------------------------------------------------------
    def load_dataset(self):
        if not (os.path.exists(self.train_file) and os.path.exists(self.test_file)):
            print(f"⚠️ No dataset found for '{spec_config.which}', generating it...")
            return self.compute_dataset()

        print(f"📂 Loading dataset for '{spec_config.which}'")
        train = dict(np.load(self.train_file, allow_pickle=True))
        test  = dict(np.load(self.test_file,  allow_pickle=True))
        self.data = {"train": train, "test": test}
        return self.data

    # ------------------------------------------------------------------
    # Regression
    # ------------------------------------------------------------------
    def train_regression(self):
        d = self.data
        Xtr = np.vstack(d["train"]["M"])
        ytr = d["train"]["r"]
        Xte = np.vstack(d["test"]["M"])
        yte = d["test"]["r"]

        model = Ridge(alpha=1e-14).fit(Xtr, ytr)
        pred  = model.predict(Xte)

        nrmse = np.sqrt(np.mean((pred - yte)**2) / np.mean(yte**2))

        print(f"\n📉 Regression performance for '{spec_config.which}': "
              f"{100*nrmse:.3f}% NRMSE")

        return model, pred

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def generate_plots(self, model, pred, show=False):
        d = self.data

        def out(name):
            return None if show else os.path.join(self.output_dir, name)

        # Spectra
        self.plotter.plot_spectra(
            d["train"]["omega"],
            d["train"]["S"],
            d["train"]["r"],
            outname=out(f"spectra_training_{spec_config.which}.png")
        )

        # Moments
        all_r = np.concatenate([d["train"]["r"], d["test"]["r"]])
        all_M = np.vstack([
            np.vstack(d["train"]["M"]),
            np.vstack(d["test"]["M"])
        ])
        self.plotter.plot_moments_vs_r(
            all_r, normalize_moments(all_M),
            outname=out(f"moments_vs_r_{spec_config.which}.png")
        )

        # Pred vs true
        self.plotter.plot_predicted_vs_true(
            d["test"]["r"], pred,
            outname=out(f"pred_vs_true_{spec_config.which}.png")
        )

        if show:
            print("👀 Plots displayed")
        else:
            print(f"📁 Plots saved into '{self.output_dir}/'")

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    def run(self, replot_only=False, show=False):
        if replot_only:
            self.load_dataset()
            model, pred = self.train_regression()
            self.generate_plots(model, pred, show)
            return

        self.load_dataset()
        model, pred = self.train_regression()
        self.generate_plots(model, pred, show)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replot-only", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    Trainer().run(replot_only=args.replot_only, show=args.show)
