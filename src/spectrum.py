import numpy as np
from qutip import correlation_2op_1t, Qobj, expect, qeye
from src.config import physical_params, numerical_params


class SpectrumCalculator:
    """
    Version propre : les corrélations requises sont définies dans un dispatcher.
    Pas de if/else répétitifs.
    """

    def __init__(self, H: Qobj, a: Qobj, rho_ss: Qobj, c_ops,
                 wlist=None, tlist=None, which="S"):

        self.H = H
        self.a = a
        self.rho_ss = rho_ss
        self.c_ops = c_ops
        self.which = which

        # Fréquences
        self.wlist = wlist if wlist is not None else np.linspace(
            numerical_params.wmin, numerical_params.wmax, numerical_params.n_w
        )

        # Temps
        if tlist is None:
            tmax = 100.0 / physical_params.gamma_c
            nt = 2000
            self.tlist = np.linspace(0.0, tmax, nt)
            self.tlist_neg = np.linspace(-tmax, 0.0, nt)
        else:
            self.tlist = tlist

        if not isinstance(rho_ss, Qobj):
            raise TypeError("rho_ss must be a QuTiP Qobj density matrix")


    # ----------------------------------------------------------------------
    # Fluctuation operator
    # ----------------------------------------------------------------------
    def compute_fluctuation_operator(self):
        a_mean = expect(self.a, self.rho_ss)
        N = self.a.shape[0]
        return self.a - a_mean * qeye(N)

    # ======================================================================
    # DISPATCHER : corrélations nécessaires selon le spectre
    # ======================================================================
    def required_correlations(self):
        """
        Retourne un dictionnaire :
            nom_corrélation : (tlist à utiliser, opérateur A, opérateur B, reverse)
        """
        a = self.compute_fluctuation_operator()
        adag = a.dag()

        return {
            "S": {
                "adag_a": (self.tlist, adag, a, False)
            },

            "Si_minus": {
                "a_a_pos":      (self.tlist,      a,      a,      False),
                "adag_adag_pos": (self.tlist,     adag,   adag,   True),
                "adag_a":       (self.tlist,      adag,   a,      False),
            },

            "Si_plus": {
                "a_a_pos":        (self.tlist,      a,      a,      False),
                "a_a_neg":        (self.tlist_neg,  a,      a,      True),
                "adag_adag_pos":  (self.tlist,      adag,   adag,   True),
                "adag_adag_neg":  (self.tlist_neg,  adag,   adag,   False),
                "adag_a":         (self.tlist,      adag,   a,      False),
                "adag_a_rev":     (self.tlist,      adag,   a,      True),
            }
        }[self.which]

    # ======================================================================
    # CALCUL GÉNÉRIQUE DES CORRÉLATIONS NÉCESSAIRES
    # ======================================================================
    def compute_correlations(self):
        specs = self.required_correlations()
        dt = self.tlist[1] - self.tlist[0]

        corr = {}

        for key, (tlist, A, B, rev) in specs.items():
            g = correlation_2op_1t(
                self.H, self.rho_ss, tlist, self.c_ops,
                A, B, reverse=rev
            )
            corr[key] = g
            # FFT → centrée → tronquée à la taille de wlist
            corr[key] = np.fft.fftshift(np.fft.fft(corr[key]) * dt)

        return corr

    # ======================================================================
    # SPECTRES
    # ======================================================================
    def compute_standard_spectrum(self, corr): 
        dt = self.tlist[1] - self.tlist[0] 
        freqs = 2 * np.pi * np.fft.fftfreq(len(self.tlist), d=dt) 
        freqs_shift = np.fft.fftshift(freqs)
        S = np.interp(self.wlist, freqs_shift, np.real(corr["adag_a"])) 
        return np.maximum(S, 0.0)

    def compute_Si_minus(self, corr): 
        dt = self.tlist[1] - self.tlist[0] 
        freqs = 2 * np.pi * np.fft.fftfreq(len(self.tlist), d=dt) 
        freqs_shift = np.fft.fftshift(freqs) 

        S_aa = np.interp(self.wlist, freqs_shift, np.real(corr["a_a_pos"])) 
        S_adagadag = np.interp(self.wlist, freqs_shift, np.real(corr["adag_adag_pos"])) 
        S_adag_a = np.interp(self.wlist, freqs_shift, np.real(corr["adag_a"])) 
        S_adag0_a = np.conj(S_adag_a) 

        phase = physical_params.phi + np.pi/2 
        pref = (physical_params.G * physical_params.e * np.abs(physical_params.beta))**2

        S_i = pref * (np.exp(2j * phase) * S_aa + np.exp(-2j * phase) * S_adagadag + S_adag_a + S_adag0_a ) 
        return np.real(S_i)

    def compute_Si_plus(self, corr): 
        dt = self.tlist[1] - self.tlist[0] 
        freqs = 2 * np.pi * np.fft.fftfreq(len(self.tlist), d=dt) 
        freqs_shift = np.fft.fftshift(freqs) 

        S_aa = np.interp(self.wlist, freqs_shift, corr["a_a_pos"]+corr["a_a_neg"]) 
        S_adagadag = np.interp(self.wlist, freqs_shift, corr["adag_adag_pos"]+corr["adag_adag_neg"]) 
        S_adag_a = np.interp(self.wlist, freqs_shift, 2*np.real(corr["adag_a"])) 
        S_adag_a_rev = np.interp(self.wlist, freqs_shift, 2*np.real(corr["adag_a_rev"])) 
        S_i = S_aa + S_adagadag + S_adag_a + S_adag_a_rev 
        return np.real(S_i)

    # ======================================================================
    # DISPATCH FINAL
    # ======================================================================
    def compute(self):

        dispatcher = {
            "S":        self.compute_standard_spectrum,
            "Si_minus": self.compute_Si_minus,
            "Si_plus":  self.compute_Si_plus,
        }

        corr = self.compute_correlations()
        S = dispatcher[self.which](corr)

        return self.wlist, S
