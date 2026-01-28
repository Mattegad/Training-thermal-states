import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
import os
import numpy as np
from src.config import NumericalParams


class Plotter:
    def __init__(self, data, outdir=None):
        self.Garr = data["Garr"]
        self.omega = data["cavomegaarr"]
        self.spec = data["opticalspec"]
        self.convratio = data["convratio"]
        self.moments = data["moments"]
        self.outdir = outdir
        self.G_true = data["G_true"]
        self.G_pred = data["G_pred"]
        colors = ["#7619E7", "#9B77C3", "#B497D6", "#E6A87C", "#F4CCA2"]
        self.cmap = LinearSegmentedColormap.from_list("orange_to_violet", colors)
        self.norm = Normalize(vmin=0.0, vmax=4.5)  # plage typique du param G
        self.Ntest = NumericalParams().Ntest
        self.Ntrain = NumericalParams().n_train
    
    def out(self, name):
        return os.path.join(self.outdir, name)
    
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

    def plot_spectra(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        for gg, G in enumerate(self.Garr[:self.Ntest]):
            ax.plot(self.omega, self.spec[gg, :], label=f"G={G:.2f}")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\omega$")
        ax.set_ylabel(r"$S(\omega)$")
        ax.legend(fontsize=8)
        self._safe_savefig(fig, outname=self.out("spectra.png"))

    def plot_centered(self):
        ref = self.spec[0, :]
        centered = self.spec - ref[None, :]

        fig, ax = plt.subplots(figsize=(10, 4))
        for gg, G in enumerate(self.Garr):
            ax.plot(self.omega, centered[gg, :], color=self.cmap(self.norm(G)))
        ax.set_xlabel(r"$\omega$ ($\mu$eV)")
        ax.set_ylabel(r"$S-S_0$ (a.u.)")
        ax.tick_params(which='both', direction='in', top=True, right=True)
        sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=self.norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=r"G (ms$^{-1}$)")
        self._safe_savefig(fig, outname=self.out("centered_spectra.png"))

    def plot_moments(self):
        moments = self.moments
        G = self.G_true
        fig, ax = plt.subplots(figsize=(8, 6))
        for k in range(moments.shape[0]):
            ax.plot(G[:self.Ntrain], moments[k][:self.Ntrain]/np.max(moments[k][:self.Ntrain]), "o", color=self.cmap(self.norm(k)), markerfacecolor='none', label=f"M{k}")
            ax.plot(G[self.Ntrain:], moments[k][self.Ntrain:]/np.max(moments[k][self.Ntrain:]), "x", color=self.cmap(self.norm(k)), label=f"M{k}")
        ax.legend()
        ax.set_xlabel("G")
        ax.set_ylabel("Moments")
        self._safe_savefig(fig, outname=self.out("moments.png"))    

    def plot_prediction(self):
        Gt = self.G_true
        Gp = self.G_pred

        fig, ax = plt.subplots(figsize=(6, 6))   
        ax.plot(Gt, Gp, "o", markerfacecolor='none')
        ax.plot(Gt, Gt, 'g')
        ax.set_xlabel(r"G true ($ms^{-1}$)")
        ax.set_ylabel(r"G predicted ($ms^{-1}$)")
        ax.tick_params(which='both', direction='in', top=True, right=True)
        self._safe_savefig(fig, outname=self.out("prediction.png"))
