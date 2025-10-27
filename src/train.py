# train.py
"""
Main script to:
 - generate spectra for a set of r values (training set)
 - compute moments M0..M4
 - fit a linear regression (ordinary least squares) mapping moments -> r
 - test on random r values and produce plots similar to Fig.2b,c
"""
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import Ridge, LinearRegression
from simulate import create_operators, hamiltonian_kerr, collapse_ops_for_squeezed_env, compute_steady_state
from spectrum import compute_spectrum_via_correlation
from moments import compute_moments
from plot_utils import plot_spectra, plot_moments_vs_r, plot_predicted_vs_true
from params import theta_default, alphaD_default, nbar_default, wmin, wmax, n_w

def generate_spectrum_for_r(r_val, theta=theta_default, alpha=alphaD_default, nbar=nbar_default):
    a, _ = create_operators()
    H = hamiltonian_kerr(a)
    c_ops = collapse_ops_for_squeezed_env(a, alpha=alpha, r=r_val, theta=theta, nbar=nbar)
    rho_ss = compute_steady_state(H, c_ops)
    wlist = np.linspace(wmin, wmax, n_w)
    w, S, t, corr = compute_spectrum_via_correlation(H, c_ops, a, rho_ss, wlist=wlist)
    return w, S

def main():
    # training set: 10 equidistant r in [0,2]
    r_train = np.linspace(0.0, 2.0, 10)
    omega_list = []
    S_list = []
    M_list = []
    print("Computing training spectra...")
    for r in tqdm(r_train):
        w, S = generate_spectrum_for_r(r)
        omega_list.append(w)
        S_list.append(S)
        M = compute_moments(w, S, max_m=4)  # M[0..4]
        M_list.append(M)
    M_array = np.vstack(M_list)  # shape (10,5)
    # use moments columns as features (we can omit M0 for regression or keep all)
    X_train = M_array  # shape (N,5)
    y_train = r_train

    # Fit ridge with tiny regularization (paper mentions ridge ~ 1e-14)
    model = Ridge(alpha=1e-14, fit_intercept=True)
    model.fit(X_train, y_train)

    # Test on 100 random r values in [0,2]
    Ntest = 100
    r_test = np.random.uniform(0.0, 2.0, Ntest)
    M_list_test = []
    S_list_test = []
    print("Computing testing spectra...")
    for r in tqdm(r_test):
        w, S = generate_spectrum_for_r(r)
        M = compute_moments(w, S, max_m=4)
        M_list_test.append(M)
        S_list_test.append((w, S))
    X_test = np.vstack(M_list_test)
    y_test = r_test
    y_pred = model.predict(X_test)

    # compute NRMSE
    nrmse = np.sqrt(np.mean((y_pred - y_test)**2) / np.mean(y_test**2))
    print(f"NRMSE = {100.*nrmse:.3f} %")

    # make plots akin to Fig.2
    # a) spectra for training r (subset)
    plot_spectra([omega_list[i] for i in range(len(omega_list))], S_list, r_train)
    # b) moments vs r (training circ., testing stars)
    # stack training and testing for plotting
    all_r = np.concatenate([r_train, r_test])
    all_M = np.vstack([M_array, X_test])
    # normalize each moment column for display
    from utils import normalize_moments
    all_M_norm = normalize_moments(all_M)
    plot_moments_vs_r(all_r, all_M_norm)
    # c) predicted vs true
    plot_predicted_vs_true(y_test, y_pred)

if __name__ == "__main__":
    main()
