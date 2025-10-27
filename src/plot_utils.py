# plot_utils.py
"""
Plot utilities for spectroscopy project.
Handles plotting of spectra, moments vs r, and regression results.
Automatically ensures valid output filenames and formats.
"""
import os
import matplotlib.pyplot as plt
import numpy as np

def _safe_savefig(fig, outname=None):
    """Save figure safely with .png default format and fallback."""
    if outname is None:
        fig.show()  # display interactively if no filename
        return
    # Ensure valid extension
    valid_exts = [".png", ".jpg", ".jpeg", ".pdf", ".svg"]
    root, ext = os.path.splitext(str(outname))
    if ext.lower() not in valid_exts:
        outname = root + ".png"
    try:
        fig.savefig(outname, bbox_inches="tight", dpi=300)
        print(f"💾 Figure saved: {outname}")
    except Exception as e:
        print(f"⚠️ Could not save figure '{outname}': {e}")
        fig.show()

def plot_spectra(omega_list, S_list, r_train, outname=None):
    """Plot training spectra as Figure 2(a)-like."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for w, S, r in zip(omega_list, S_list, r_train):
        ax.plot(w, S / np.max(S), label=f"r={r:.2f}")
    ax.set_xlabel("ω / κ")
    ax.set_ylabel("Normalized S(ω)")
    ax.set_title("Training spectra vs squeezing parameter r")
    ax.legend(fontsize=8, loc="best")
    _safe_savefig(fig, outname)
    plt.close(fig)

def plot_moments_vs_r(r_values, moments, outname=None):
    """Plot moments M0..M4 vs r."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for i in range(moments.shape[1]):
        ax.plot(r_values, moments[:, i], "o", label=f"M{i}")
    ax.set_xlabel("r")
    ax.set_ylabel("Normalized moments")
    ax.legend(fontsize=8)
    ax.set_title("Moments vs squeezing parameter r")
    _safe_savefig(fig, outname)
    plt.close(fig)

def plot_predicted_vs_true(y_true, y_pred, outname=None):
    """Plot predicted vs true r values."""
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(y_true, y_pred, "o", alpha=0.7)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "k--")
    ax.set_xlabel("True r")
    ax.set_ylabel("Predicted r")
    ax.set_title("Regression performance")
    ax.grid(True, ls=":")
    _safe_savefig(fig, outname)
    plt.close(fig)
