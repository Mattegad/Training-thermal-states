# train.py
import os
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import Ridge
from qutip import destroy

from src.config import physical_params, numerical_params, squeezing_params
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

        # files
        self.train_std = os.path.join(output_dir, "training_data_std.npz")
        self.test_std  = os.path.join(output_dir, "testing_data_std.npz")
        self.train_si  = os.path.join(output_dir, "training_data_si.npz")
        self.test_si   = os.path.join(output_dir, "testing_data_si.npz")

        # tools
        self.moments_calc = MomentsCalculator()
        self.plotter = Plotter()

        # storage
        self.data = None

    # ------------------------------------------------------------------
    # --- Core physics -------------------------------------------------
    # ------------------------------------------------------------------
    def generate_spectrum(self, r_val, compute_si=True):
        """Compute stationary spectrum for a given r."""
        Ncut = numerical_params.Ncut
        a = destroy(Ncut)

        # Hamiltonian
        H = HamiltonianFactory.make("kerr", a, U_val=physical_params.U)

        # Dissipation
        c_ops = DissipationFactory.make("squeezed", a,
                                        alpha=squeezing_params.alphaD_default,
                                        r=r_val,
                                        theta=squeezing_params.theta_default,
                                        nbar=squeezing_params.nbar_default,
                                        gamma=physical_params.gamma_c)

        # Steady state
        rho_ss = SteadyStateSolver(H, c_ops).compute()

        # Frequencies
        wlist = np.linspace(numerical_params.wmin, numerical_params.wmax, numerical_params.n_w)

        # Spectrum
        calc = SpectrumCalculator(H, a, rho_ss, c_ops, wlist=wlist, compute_si=compute_si)
        return calc.compute()

    # ------------------------------------------------------------------
    # --- Data generation ---------------------------------------------
    # ------------------------------------------------------------------
    def compute_dataset(self):
        """Compute and save all training and test data."""

        r_train = np.linspace(0.0, 2.0, numerical_params.n_train)
        r_test  = np.random.uniform(0.0, 2.0, numerical_params.n_test)

        # Storage dict
        result = {
            "train_std":   {"r": r_train, "omega": [], "S": [], "M": []},
            "train_si":    {"r": r_train, "omega": [], "S": [], "M": []},
            "test_std":    {"r": r_test,  "omega": [], "S": [], "M": []},
            "test_si":     {"r": r_test,  "omega": [], "S": [], "M": []},
        }

        # ------------------------
        # TRAINING
        # ------------------------
        print("\n📘 Computing TRAINING data")
        for r in tqdm(r_train):
            # Standard spectrum
            w, S = self.generate_spectrum(r, compute_si=False)
            result["train_std"]["omega"].append(w)
            result["train_std"]["S"].append(S)
            result["train_std"]["M"].append(self.moments_calc.compute(w, S))

            # S_i spectrum
            w, _, S_i = self.generate_spectrum(r, compute_si=True)
            result["train_si"]["omega"].append(w)
            result["train_si"]["S"].append(S_i)
            result["train_si"]["M"].append(self.moments_calc.compute(w, S_i))

        # ------------------------
        # TEST
        # ------------------------
        print("\n📗 Computing TEST data")
        for r in tqdm(r_test):
            # Standard
            w, S = self.generate_spectrum(r, compute_si=False)
            result["test_std"]["omega"].append(w)
            result["test_std"]["S"].append(S)
            result["test_std"]["M"].append(self.moments_calc.compute(w, S))

            # S_i
            w, _, S_i = self.generate_spectrum(r, compute_si=True)
            result["test_si"]["omega"].append(w)
            result["test_si"]["S"].append(S_i)
            result["test_si"]["M"].append(self.moments_calc.compute(w, S_i))

        # Save
        np.savez_compressed(self.train_std, **result["train_std"])
        np.savez_compressed(self.train_si,  **result["train_si"])
        np.savez_compressed(self.test_std,  **result["test_std"])
        np.savez_compressed(self.test_si,   **result["test_si"])

        print(f"\n💾 Data saved to '{self.output_dir}/'")

        self.data = result
        return result

    # ------------------------------------------------------------------
    # --- Load data ----------------------------------------------------
    # ------------------------------------------------------------------
    def load_dataset(self):
        files = [self.train_std, self.test_std, self.train_si, self.test_si]
        if not all(os.path.exists(f) for f in files):
            print("⚠️ No saved data found → computing dataset.")
            return self.compute_dataset()

        print("📂 Loading dataset from disk...")
        data = {
            "train_std": dict(np.load(self.train_std, allow_pickle=True)),
            "train_si":  dict(np.load(self.train_si,  allow_pickle=True)),
            "test_std":  dict(np.load(self.test_std,  allow_pickle=True)),
            "test_si":   dict(np.load(self.test_si,   allow_pickle=True)),
        }
        self.data = data
        return data

    # ------------------------------------------------------------------
    # --- Regression ---------------------------------------------------
    # ------------------------------------------------------------------
    def train_regression(self):
        """Train Ridge regression models for STD and SI- spectra."""
        d = self.data

        def prepare(split, kind):
            X = np.vstack(d[split + "_" + kind]["M"])
            y = d[split + "_" + kind]["r"]
            return X, y

        Xtr_std, ytr_std = prepare("train", "std")
        Xte_std, yte_std = prepare("test",  "std")

        Xtr_si, ytr_si = prepare("train", "si")
        Xte_si, yte_si = prepare("test",  "si")

        model_std = Ridge(alpha=1e-14).fit(Xtr_std, ytr_std)
        model_si  = Ridge(alpha=1e-14).fit(Xtr_si,  ytr_si)

        pred_std = model_std.predict(Xte_std)
        pred_si  = model_si.predict(Xte_si)

        nrmse_std = np.sqrt(np.mean((pred_std - yte_std)**2) / np.mean(yte_std**2))
        nrmse_si  = np.sqrt(np.mean((pred_si  - yte_si)**2) / np.mean(yte_si**2))

        print(f"\n📉 Regression performance:")
        print(f"  → STD : {100*nrmse_std:.3f} % NRMSE")
        print(f"  → SI- : {100*nrmse_si:.3f} % NRMSE")

        return model_std, model_si, pred_std, pred_si

    # ------------------------------------------------------------------
    # --- Plotting -----------------------------------------------------
    # ------------------------------------------------------------------
    def generate_plots(self, model_std, model_si, pred_std, pred_si, show=False):
        d = self.data

        def out(name):
            return None if show else os.path.join(self.output_dir, name)

        # SPECTRA
        self.plotter.plot_spectra(
            d["train_std"]["omega"],
            d["train_std"]["S"],
            d["train_std"]["r"],
            outname=out("spectra_training_std.png")
        )

        self.plotter.plot_spectra(
            d["train_si"]["omega"],
            d["train_si"]["S"],
            d["train_si"]["r"],
            outname=out("spectra_training_si.png")
        )

        # MOMENTS VS R
        all_r_std = np.concatenate([d["train_std"]["r"], d["test_std"]["r"]])
        all_r_si  = np.concatenate([d["train_si"]["r"],  d["test_si"]["r"]])

        all_M_std = np.vstack([np.vstack(d["train_std"]["M"]),
                               np.vstack(d["test_std"]["M"])])
        all_M_si  = np.vstack([np.vstack(d["train_si"]["M"]),
                               np.vstack(d["test_si"]["M"])])

        self.plotter.plot_moments_vs_r(
            all_r_std, normalize_moments(all_M_std),
            outname=out("moments_vs_r_std.png")
        )
        self.plotter.plot_moments_vs_r(
            all_r_si, normalize_moments(all_M_si),
            outname=out("moments_vs_r_si.png")
        )

        # PREDICTION VS TRUE
        self.plotter.plot_predicted_vs_true(
            d["test_std"]["r"], pred_std,
            outname=out("pred_vs_true_std.png")
        )
        self.plotter.plot_predicted_vs_true(
            d["test_si"]["r"], pred_si,
            outname=out("pred_vs_true_si.png")
        )

        if show:
            print("👀 Plots displayed")
        else:
            print(f"📁 Plots saved in '{self.output_dir}/'")

    # ------------------------------------------------------------------
    # --- Full pipeline ------------------------------------------------
    # ------------------------------------------------------------------
    def run(self, replot_only=False, show=False):
        if replot_only:
            self.load_dataset()
            model_std, model_si, pred_std, pred_si = self.train_regression()
            self.generate_plots(model_std, model_si, pred_std, pred_si, show=show)
            return

        self.load_dataset()         # loads or generates
        model_std, model_si, pred_std, pred_si = self.train_regression()
        self.generate_plots(model_std, model_si, pred_std, pred_si, show=show)


# ======================================================================
# MAIN SCRIPT ENTRY
# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replot-only", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    Trainer().run(replot_only=args.replot_only, show=args.show)
