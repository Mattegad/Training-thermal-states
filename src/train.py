# train.py
"""
Main script to:
 - generate spectra for a set of r values (training set)
 - compute moments M0..M4
 - fit a linear regression mapping moments -> r
 - test on random r values
 - save / reload results to avoid recomputation
 - support a '--replot-only' mode to just reload and replot results
"""
import os
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import Ridge
from src.simulate import create_operators, hamiltonian_kerr, collapse_ops_for_squeezed_env, compute_steady_state
from src.spectrum import compute_spectrum_via_correlation
from src.moments import compute_moments
from src.plot_utils import plot_spectra, plot_moments_vs_r, plot_predicted_vs_true
from src.utils import normalize_moments
from src.params import theta_default, alphaD_default, nbar_default, wmin, wmax, n_w

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(OUTPUT_DIR, "training_data.npz")
TEST_FILE = os.path.join(OUTPUT_DIR, "testing_data.npz")

# --------------------------------------------------------------
# Data generation functions
# --------------------------------------------------------------

def generate_spectrum_for_r(r_val, theta=theta_default, alpha=alphaD_default, nbar=nbar_default):
    """Compute one steady-state spectrum for a given squeezing r."""
    a, _ = create_operators()
    H = hamiltonian_kerr(a)
    c_ops = collapse_ops_for_squeezed_env(a, alpha=alpha, r=r_val, theta=theta, nbar=nbar)
    rho_ss = compute_steady_state(H, c_ops)
    wlist = np.linspace(wmin, wmax, n_w)
    w, S, _, _ = compute_spectrum_via_correlation(H, c_ops, a, rho_ss, wlist=wlist)
    return w, S

def compute_and_save_data():
    """Compute training and test data, save to disk."""
    # --- TRAINING ---
    r_train = np.linspace(0.0, 2.0, 10)
    omega_list, S_list, M_list = [], [], []
    print("Computing training spectra...")
    for r in tqdm(r_train):
        w, S = generate_spectrum_for_r(r)
        omega_list.append(w)
        S_list.append(S)
        M = compute_moments(w, S, max_m=4)
        M_list.append(M)
    np.savez_compressed(TRAIN_FILE,
                        r_train=r_train,
                        omega_list=omega_list,
                        S_list=S_list,
                        M_list=M_list)

    # --- TESTING ---
    r_test = np.random.uniform(0.0, 2.0, 20)
    M_list_test, omega_list_test, S_list_test = [], [], []
    print("Computing testing spectra...")
    for r in tqdm(r_test):
        w, S = generate_spectrum_for_r(r)
        omega_list_test.append(w)
        S_list_test.append(S)
        M = compute_moments(w, S, max_m=4)
        M_list_test.append(M)
    np.savez_compressed(TEST_FILE,
                        r_test=r_test,
                        omega_list_test=omega_list_test,
                        S_list_test=S_list_test,
                        M_list_test=M_list_test)

    print(f"✅ Data saved to '{OUTPUT_DIR}/'")
    return (r_train, omega_list, S_list, M_list,
            r_test, omega_list_test, S_list_test, M_list_test)

def load_data():
    """Load previously computed data if available."""
    if os.path.exists(TRAIN_FILE) and os.path.exists(TEST_FILE):
        print("📂 Loading data from disk...")
        train = np.load(TRAIN_FILE, allow_pickle=True)
        test = np.load(TEST_FILE, allow_pickle=True)
        return (train["r_train"], train["omega_list"], train["S_list"], train["M_list"],
                test["r_test"], test["omega_list_test"], test["S_list_test"], test["M_list_test"])
    else:
        print("⚠️ No saved data found, computing everything...")
        return compute_and_save_data()

# --------------------------------------------------------------
# Main function
# --------------------------------------------------------------

def main(replot_only=False, show=False):
    # --- Load or compute data ---
    if replot_only:
        if not (os.path.exists(TRAIN_FILE) and os.path.exists(TEST_FILE)):
            print("⚠️ No saved data found, need to compute first.")
            return
        (r_train, omega_list, S_list, M_list,
         r_test, omega_list_test, S_list_test, M_list_test) = load_data()
    else:
        (r_train, omega_list, S_list, M_list,
         r_test, omega_list_test, S_list_test, M_list_test) = load_data()

    # --- Train regression ---
    X_train = np.vstack(M_list)
    y_train = r_train
    model = Ridge(alpha=1e-14, fit_intercept=True)
    model.fit(X_train, y_train)

    # --- Predict on test set ---
    X_test = np.vstack(M_list_test)
    y_test = r_test
    y_pred = model.predict(X_test)

    nrmse = np.sqrt(np.mean((y_pred - y_test)**2) / np.mean(y_test**2))
    print(f"NRMSE = {100.*nrmse:.3f} %")

    # --- Replot / visualize ---
    if show:
        out = None  # show interactively
    else:
        out = lambda name: os.path.join(OUTPUT_DIR, name)

    plot_spectra(omega_list, S_list, r_train,
                 outname=None if show else out("spectra_training.png"))
    all_r = np.concatenate([r_train, r_test])
    all_M = np.vstack([X_train, X_test])
    all_M_norm = normalize_moments(all_M)
    plot_moments_vs_r(all_r, all_M_norm,
                      outname=None if show else out("moments_vs_r.png"))
    plot_predicted_vs_true(y_test, y_pred,
                           outname=None if show else out("pred_vs_true.png"))

    if show:
        print("👀 Figures displayed interactively.")
    else:
        print("✅ Plots saved to 'output/' folder.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, replot, or show results.")
    parser.add_argument("--replot-only", action="store_true",
                        help="Skip computation and only replot existing data.")
    parser.add_argument("--show", action="store_true",
                        help="Display plots instead of saving them.")
    args = parser.parse_args()

    main(replot_only=args.replot_only, show=args.show)

