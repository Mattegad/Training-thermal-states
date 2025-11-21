# train_extended.py
import os
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import Ridge
from src.simulate import create_operators, hamiltonian_kerr, collapse_ops_for_squeezed_env, compute_steady_state
from src.spectrum_new import compute_spectrum_via_correlation
from src.moments import compute_moments
from src.plot_utils import plot_spectra, plot_moments_vs_r, plot_predicted_vs_true
from src.utils import normalize_moments
from src.params import theta_default, alphaD_default, nbar_default, wmin, wmax, n_w, n_train, n_test

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Files for standard spectrum
TRAIN_FILE_STD = os.path.join(OUTPUT_DIR, "training_data_std.npz")
TEST_FILE_STD = os.path.join(OUTPUT_DIR, "testing_data_std.npz")

# Files for Si- spectrum
TRAIN_FILE_SI = os.path.join(OUTPUT_DIR, "training_data_si.npz")
TEST_FILE_SI = os.path.join(OUTPUT_DIR, "testing_data_si.npz")



# --------------------------------------------------------------
# Data generation functions
# --------------------------------------------------------------

def generate_spectrum_for_r(r_val, theta=theta_default, alpha=alphaD_default, nbar=nbar_default, compute_si=True):
    """Compute one steady-state spectrum for a given squeezing r."""
    a, _ = create_operators()
    H = hamiltonian_kerr(a)
    c_ops = collapse_ops_for_squeezed_env(a, alpha=alpha, r=r_val, theta=theta, nbar=nbar)
    rho_ss = compute_steady_state(H, c_ops)
    wlist = np.linspace(wmin, wmax, n_w)
    if compute_si:
        w, S, S_i = compute_spectrum_via_correlation(
            H, c_ops, a, rho_ss, wlist=wlist,
            compute_si=True
        )
        return w, S, S_i
    else:
        w, S = compute_spectrum_via_correlation(H, c_ops, a, rho_ss, wlist=wlist)
        return w, S

def compute_and_save_data():
    """Compute training and test data for both standard and Si- spectra."""
    # --- TRAINING ---
    r_train = np.linspace(0.0, 2.0, n_train)
    omega_list_std, S_list_std, M_list_std = [], [], []
    omega_list_si, S_list_si, M_list_si = [], [], []

    print("Computing training spectra...")
    for r in tqdm(r_train):
        # Standard spectrum
        w, S = generate_spectrum_for_r(r, compute_si=False)
        omega_list_std.append(w)
        S_list_std.append(S)
        M_list_std.append(compute_moments(w, S, max_m=4))

        # Si- spectrum
        w, _, S_i = generate_spectrum_for_r(r, compute_si=True)
        omega_list_si.append(w)
        S_list_si.append(S_i)
        M_list_si.append(compute_moments(w, S_i, max_m=4))

    np.savez_compressed(TRAIN_FILE_STD,
                        r_train=r_train,
                        omega_list=omega_list_std,
                        S_list=S_list_std,
                        M_list=M_list_std)

    np.savez_compressed(TRAIN_FILE_SI,
                        r_train=r_train,
                        omega_list=omega_list_si,
                        S_list=S_list_si,
                        M_list=M_list_si)

    # --- TESTING ---
    r_test = np.random.uniform(0.0, 2.0, n_test)
    omega_list_test_std, S_list_test_std, M_list_test_std = [], [], []
    omega_list_test_si, S_list_test_si, M_list_test_si = [], [], []

    print("Computing testing spectra...")
    for r in tqdm(r_test):
        w, S = generate_spectrum_for_r(r, compute_si=False)
        omega_list_test_std.append(w)
        S_list_test_std.append(S)
        M_list_test_std.append(compute_moments(w, S, max_m=4))

        w, _, S_i = generate_spectrum_for_r(r, compute_si=True)
        omega_list_test_si.append(w)
        S_list_test_si.append(S_i)
        M_list_test_si.append(compute_moments(w, S_i, max_m=4))

    np.savez_compressed(TEST_FILE_STD,
                        r_test=r_test,
                        omega_list_test=omega_list_test_std,
                        S_list_test=S_list_test_std,
                        M_list_test=M_list_test_std)

    np.savez_compressed(TEST_FILE_SI,
                        r_test=r_test,
                        omega_list_test=omega_list_test_si,
                        S_list_test=S_list_test_si,
                        M_list_test=M_list_test_si)

    print(f"✅ Data saved to '{OUTPUT_DIR}/'")
    return (r_train, omega_list_std, S_list_std, M_list_std,
            r_train, omega_list_si, S_list_si, M_list_si,
            r_test, omega_list_test_std, S_list_test_std, M_list_test_std,
            r_test, omega_list_test_si, S_list_test_si, M_list_test_si)

def load_data():
    """Load previously computed data for both spectra."""
    if all(os.path.exists(f) for f in [TRAIN_FILE_STD, TEST_FILE_STD, TRAIN_FILE_SI, TEST_FILE_SI]):
        print("📂 Loading data from disk...")
        train_std = np.load(TRAIN_FILE_STD, allow_pickle=True)
        test_std = np.load(TEST_FILE_STD, allow_pickle=True)
        train_si = np.load(TRAIN_FILE_SI, allow_pickle=True)
        test_si = np.load(TEST_FILE_SI, allow_pickle=True)
        return (train_std["r_train"], train_std["omega_list"], train_std["S_list"], train_std["M_list"],
                train_si["r_train"], train_si["omega_list"], train_si["S_list"], train_si["M_list"],
                test_std["r_test"], test_std["omega_list_test"], test_std["S_list_test"], test_std["M_list_test"],
                test_si["r_test"], test_si["omega_list_test"], test_si["S_list_test"], test_si["M_list_test"])
    else:
        print("⚠️ No saved data found, computing everything...")
        return compute_and_save_data()

# --------------------------------------------------------------
# Main function
# --------------------------------------------------------------

def main(replot_only=False, show=False):
    # --- Load or compute data ---
    if replot_only:
        if not all(os.path.exists(f) for f in [TRAIN_FILE_STD, TEST_FILE_STD, TRAIN_FILE_SI, TEST_FILE_SI]):
            print("⚠️ No saved data found, need to compute first.")
            return
        (r_train_std, omega_list_std, S_list_std, M_list_std,
         r_train_si, omega_list_si, S_list_si, M_list_si,
         r_test_std, omega_list_test_std, S_list_test_std, M_list_test_std,
         r_test_si, omega_list_test_si, S_list_test_si, M_list_test_si) = load_data()
    else:
        (r_train_std, omega_list_std, S_list_std, M_list_std,
         r_train_si, omega_list_si, S_list_si, M_list_si,
         r_test_std, omega_list_test_std, S_list_test_std, M_list_test_std,
         r_test_si, omega_list_test_si, S_list_test_si, M_list_test_si) = load_data()

    # --- Train regression for standard spectrum ---
    X_train_std = np.vstack(M_list_std)
    y_train_std = r_train_std
    model_std = Ridge(alpha=1e-14, fit_intercept=True)
    model_std.fit(X_train_std, y_train_std)

    X_test_std = np.vstack(M_list_test_std)
    y_test_std = r_test_std
    y_pred_std = model_std.predict(X_test_std)
    nrmse_std = np.sqrt(np.mean((y_pred_std - y_test_std)**2) / np.mean(y_test_std**2))
    print(f"NRMSE (standard) = {100.*nrmse_std:.3f} %")

    # --- Train regression for Si- spectrum ---
    X_train_si = np.vstack(M_list_si)
    y_train_si = r_train_si
    model_si = Ridge(alpha=1e-14, fit_intercept=True)
    model_si.fit(X_train_si, y_train_si)

    X_test_si = np.vstack(M_list_test_si)
    y_test_si = r_test_si
    y_pred_si = model_si.predict(X_test_si)
    nrmse_si = np.sqrt(np.mean((y_pred_si - y_test_si)**2) / np.mean(y_test_si**2))
    print(f"NRMSE (Si-) = {100.*nrmse_si:.3f} %")

    # --- Replot / visualize ---
    if show:
        out = None
    else:
        out = lambda name: os.path.join(OUTPUT_DIR, name)

    plot_spectra(omega_list_std, S_list_std, r_train_std,
                 outname=None if show else out("spectra_training_std.png"))
    plot_spectra(omega_list_si, S_list_si, r_train_si,
                 outname=None if show else out("spectra_training_si.png"))

    all_r_std = np.concatenate([r_train_std, r_test_std])
    all_M_std = np.vstack([X_train_std, X_test_std])
    all_M_norm_std = normalize_moments(all_M_std)
    plot_moments_vs_r(all_r_std, all_M_norm_std,
                      outname=None if show else out("moments_vs_r_std.png"))

    all_r_si = np.concatenate([r_train_si, r_test_si])
    all_M_si = np.vstack([X_train_si, X_test_si])
    all_M_norm_si = normalize_moments(all_M_si)
    plot_moments_vs_r(all_r_si, all_M_norm_si,
                      outname=None if show else out("moments_vs_r_si.png"))

    plot_predicted_vs_true(y_test_std, y_pred_std,
                           outname=None if show else out("pred_vs_true_std.png"))
    plot_predicted_vs_true(y_test_si, y_pred_si,
                           outname=None if show else out("pred_vs_true_si.png"))

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
