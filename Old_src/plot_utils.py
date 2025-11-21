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
from matplotlib.ticker import MultipleLocator, FuncFormatter

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
    ax.tick_params(which='both', direction='in')
    ax.tick_params(top=True, right=True)

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
        ax.plot(r_values, moments[:, i], "o", markerfacecolor='none', label=f"M{i}")
    ax.set_xlabel("r")
    ax.set_ylabel("Normalized moments")
    ax.legend(fontsize=8)
    ax.set_title("Moments vs squeezing parameter r")
    plt.margins(x=0)
    plt.margins(y=0)

    # Ticks majeurs tous les 1.0 (affichés avec labels)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    # Ticks mineurs tous les 0.5 (sans labels)
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))

    # Même chose pour Y, seulement si échelle linéaire
    if ax.get_yscale() != 'log':
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    # Optionnel : format des labels
    ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{val:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{val:g}"))

    # Activer les ticks mineurs visuellement
    ax.tick_params(which='minor', length=2, color='black')
    ax.tick_params(which='both', direction='in')
    ax.tick_params(which='both', top=True, right=True)
    _safe_savefig(fig, outname)
    plt.close(fig)

def plot_predicted_vs_true(y_true, y_pred, outname=None):
    """Plot predicted vs true r values."""
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(y_true, y_pred, "o", markerfacecolor='none', markeredgecolor='blue', alpha=0.7)
    lims = [0,2]
    #lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "k")
    ax.set_xlabel("r target")
    ax.set_ylabel("r predicted")
    ax.set_title("Regression performance")
    ax.tick_params(direction='in')
    ax.tick_params(top=True, right=True)
    plt.margins(x=0)
    plt.margins(y=0)
    _safe_savefig(fig, outname)
    plt.close(fig)
