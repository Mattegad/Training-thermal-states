# plot_utils.py
import matplotlib.pyplot as plt
import numpy as np

def plot_spectra(omega_list, S_list, r_list, outname=None):
    """
    Plot several spectra S(omega) for different r on same plot (Figure2a style).
    """
    plt.figure(figsize=(7,4.5))
    for omega, S, r in zip(omega_list, S_list, r_list):
        plt.plot(omega, S, label=f"r={r:.2f}")
    plt.xlabel("Frequency (arb. units)")
    plt.ylabel("S(ω) (arb.)")
    plt.legend()
    plt.tight_layout()
    if outname:
        plt.savefig(outname)
    plt.show()

def plot_moments_vs_r(r_vals, moments_vals, outname=None):
    """
    moments_vals shape: (N, n_moments)
    """
    plt.figure(figsize=(7,4.5))
    n_m = moments_vals.shape[1]
    for m in range(n_m):
        col = moments_vals[:, m]
        # rescale for display: divide by max
        mx = np.max(np.abs(col)) if np.max(np.abs(col))!=0 else 1.0
        plt.plot(r_vals, col / mx, 'o-' if m==0 else 's--', label=f"M{m}")
    plt.xlabel("r")
    plt.ylabel("rescaled moment")
    plt.legend()
    plt.tight_layout()
    if outname:
        plt.savefig(outname)
    plt.show()

def plot_predicted_vs_true(y_true, y_pred, outname=None):
    plt.figure(figsize=(5,5))
    plt.scatter(y_true, y_pred, c='C0')
    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    plt.plot([mn,mx],[mn,mx],'k--',alpha=0.6)
    plt.xlabel("true r")
    plt.ylabel("predicted r")
    plt.tight_layout()
    if outname:
        plt.savefig(outname)
    plt.show()
