# src/plot_utils.py
import os
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator, FuncFormatter
import numpy as np

class Plotter:
    """
    Classe utilitaire pour tracer :
    - Spectres S(ω)
    - Moments vs r
    - Résultats de régression
    """

    def __init__(self):
        colors = ["#7D5BA6", "#9B77C3", "#B497D6", "#E6A87C", "#F2C08C"]
        self.cmap = LinearSegmentedColormap.from_list("orange_to_violet", colors)
        self.norm = Normalize(vmin=0.0, vmax=2.0)  # plage typique du param r

    def _safe_savefig(self, fig, outname=None):
        """Sauvegarde sécurisée, fallback vers affichage interactif."""
        if outname is None:
            fig.show()
            return
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
        plt.close(fig)

    def plot_spectra(self, omega_list, S_list, r_values, outname=None):
        fig, ax = plt.subplots(figsize=(6, 4))
        for w, S, r in zip(omega_list, S_list, r_values):
            ax.plot(w*1e3, S, color=self.cmap(self.norm(r)))
        ax.set_xlabel("$\omega$ (meV)")
        ax.set_ylabel("S (a.u.)")
        ax.set_yscale("log")
        ax.set_title("Training spectra vs squeezing parameter r")
        ax.tick_params(which='both', direction='in', top=True, right=True)
        sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=self.norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="r")
        plt.margins(x=0)
        self._safe_savefig(fig, outname)

    def plot_moments_vs_r(self, r_values, moments, outname=None):
        fig, ax = plt.subplots(figsize=(6, 4))
        for i in range(moments.shape[1]):
            ax.plot(r_values, moments[:, i], "o", markerfacecolor='none', label=f"M{i}")
        ax.set_xlabel("r")
        ax.set_ylabel("Normalized moments")
        ax.legend(fontsize=8)
        ax.set_title("Moments vs squeezing parameter r")
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        if ax.get_yscale() != 'log':
            ax.yaxis.set_major_locator(MultipleLocator(0.5))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{val:g}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{val:g}"))
        self._safe_savefig(fig, outname)

    def plot_predicted_vs_true(self, y_true, y_pred, outname=None):
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.plot(y_true, y_pred, "o", markerfacecolor='none', markeredgecolor='blue', alpha=0.7)
        lims = [0, 2]
        ax.plot(lims, lims, "k")
        ax.set_xlabel("r target")
        ax.set_ylabel("r predicted")
        ax.set_title("Regression performance")
        ax.tick_params(direction='in', top=True, right=True)
        plt.margins(x=0)
        plt.margins(y=0)
        self._safe_savefig(fig, outname)
