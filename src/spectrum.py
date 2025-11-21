# src/spectrum.py
import numpy as np
from qutip import correlation_2op_1t, Qobj, expect, qeye
from src.config import physical_params, numerical_params

class SpectrumCalculator:
    """Calcule le spectre S(ω) et optionnellement S_{i_-}(ω) pour un mode Kerr."""

    def __init__(self, H: Qobj, a: Qobj, rho_ss: Qobj, c_ops, 
                 wlist=None, tlist=None, compute_si=False):
        self.H = H
        self.a = a
        self.rho_ss = rho_ss
        self.c_ops = c_ops
        self.compute_si = compute_si

        # Fréquence
        self.wlist = wlist if wlist is not None else np.linspace(
            numerical_params.wmin, numerical_params.wmax, numerical_params.n_w)

        # Temps
        if tlist is None:
            tmax = 50.0 / physical_params.gamma_c
            nt = 2000
            self.tlist = np.linspace(0.0, tmax, nt)
        else:
            self.tlist = tlist

        if not isinstance(rho_ss, Qobj):
            raise TypeError("rho_ss must be a QuTiP Qobj density matrix")

    def compute_fluctuation_operator(self):
        """Opérateur de fluctuation δa = a - <a>."""
        a_mean = expect(self.a, self.rho_ss)
        N = self.a.shape[0]
        return self.a - a_mean * qeye(N)

    def compute_standard_spectrum(self):
        """Calcule S(ω) = Re ∫ <a†(t)a(0)> e^{-i ω t} dt"""
        a_fluct = self.compute_fluctuation_operator()
        corr = correlation_2op_1t(
            H=self.H,
            state0=self.rho_ss,
            taulist=self.tlist,
            c_ops=self.c_ops,
            a_op=a_fluct.dag(),
            b_op=a_fluct
        )
        dt = self.tlist[1] - self.tlist[0]
        G = np.fft.fft(corr) * dt
        freqs = 2 * np.pi * np.fft.fftfreq(len(self.tlist), d=dt)
        Gs = np.fft.fftshift(G)
        freqs_shift = np.fft.fftshift(freqs)
        S = np.interp(self.wlist, freqs_shift, np.real(Gs))
        return np.maximum(S, 0.0)

    def compute_Si(self):
        """Calcule S_{i_-}(ω) selon la formule donnée."""
        a_fluct = self.compute_fluctuation_operator()
        dt = self.tlist[1] - self.tlist[0]
        freqs = 2 * np.pi * np.fft.fftfreq(len(self.tlist), d=dt)
        freqs_shift = np.fft.fftshift(freqs)

        # Correlations nécessaires
        corr_aa = correlation_2op_1t(self.H, self.rho_ss, self.tlist, self.c_ops, a_fluct, a_fluct)
        S_aa = np.fft.fftshift(np.fft.fft(corr_aa) * dt)
        
        corr_adagadag = correlation_2op_1t(self.H, self.rho_ss, self.tlist, self.c_ops, a_fluct.dag(), a_fluct.dag())
        S_adagadag = np.fft.fftshift(np.fft.fft(corr_adagadag) * dt)
        
        corr_adag_a = correlation_2op_1t(self.H, self.rho_ss, self.tlist, self.c_ops, a_fluct.dag(), a_fluct)
        S_adag_a = np.fft.fftshift(np.fft.fft(corr_adag_a) * dt)
        S_adag0_a = np.conj(S_adag_a)

        # Interpolation sur wlist
        S_aa_i = np.maximum(np.interp(self.wlist, freqs_shift, S_aa), 0.0)
        S_aadag_i = np.maximum(np.interp(self.wlist, freqs_shift, S_adagadag), 0.0)
        S_adag_a_i = np.maximum(np.interp(self.wlist, freqs_shift, S_adag_a), 0.0)
        S_adag0_a_i = np.maximum(np.interp(self.wlist, freqs_shift, S_adag0_a), 0.0)

        # Formule finale S_{i_-}(ω)
        S_i_minus = (physical_params.G * physical_params.e * np.abs(physical_params.beta))**2 * (
            np.exp(2j*(physical_params.phi + np.pi/2)) * S_aa_i +
            np.exp(-2j*(physical_params.phi + np.pi/2)) * S_aadag_i +
            S_adag_a_i +
            S_adag0_a_i
        )
        return np.real(S_i_minus)

    def compute(self):
        """Calcule S(ω) et éventuellement S_{i_-}(ω)."""
        S = self.compute_standard_spectrum()
        if self.compute_si:
            S_i_minus = self.compute_Si()
            return self.wlist, S, S_i_minus
        return self.wlist, S
