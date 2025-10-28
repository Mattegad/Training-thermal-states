# spectrum.py
import numpy as np
from qutip import correlation_2op_1t, Qobj
from src.params import wmin, wmax, n_w, gamma_c

def compute_spectrum_via_correlation(H, c_ops, a, rho_ss, wlist=None, tlist=None):
    """
    Compute S(ω) = Re ∫ <a†(t)a(0)> e^{-i ω t} dt
    Compatible with QuTiP 5.x (uses state0 instead of rho0).
    """
    if wlist is None:
        wlist = np.linspace(wmin, wmax, n_w)

    if tlist is None:
        tmax = 50.0 / gamma_c
        nt = 2000
        tlist = np.linspace(0.0, tmax, nt)

    if not isinstance(rho_ss, Qobj):
        raise TypeError("rho_ss must be a QuTiP Qobj density matrix")

    # ✅ QuTiP 5.x syntax
    corr = correlation_2op_1t(
        H=H,
        state0=rho_ss,
        taulist=tlist,
        c_ops=c_ops,
        a_op=a.dag(),
        b_op=a
    )

    # --- FFT pour obtenir S(ω) ---
    dt = tlist[1] - tlist[0]
    G = np.fft.fft(corr) * dt
    freqs = 2 * np.pi * np.fft.fftfreq(len(tlist), d=dt)
    Gs = np.fft.fftshift(G)
    freqs_shift = np.fft.fftshift(freqs)

    S = np.interp(wlist, freqs_shift, np.real(Gs))
    S = np.maximum(S, 0.0)

    return wlist, S, tlist, corr