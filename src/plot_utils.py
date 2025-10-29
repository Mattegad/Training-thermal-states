# plot_utils.py
"""
Plot utilities for spectroscopy project.
Handles plotting of spectra, moments vs r, and regression results.
Automatically ensures valid output filenames and formats.
"""
import os
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap

colors = ["#7D5BA6", "#9B77C3", "#B497D6", "#E6A87C", "#F2C08C"]
cmap = LinearSegmentedColormap.from_list("orange_to_violet", colors)

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
    """Plot training spectra with colorbar for r values (blue to red)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Colormap : bleu à rouge
    norm = Normalize(vmin=0.0, vmax=2.0)  # r varie de 0 à 2

    # Tracer chaque spectre avec la couleur correspondante
    for w, S, r in zip(omega_list, S_list, r_train):
        ax.plot(w*1e3, S, color=cmap(norm(r)))

    ax.set_xlabel("$\omega$ (meV)")
    ax.set_ylabel("S (a.u.)")
    ax.set_yscale("log")
    ax.set_title("Training spectra vs squeezing parameter r")

    # Ajouter colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # nécessaire pour colorbar
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("r")
    plt.margins(x=0)
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
