# spectrum.py
"""
Compute S(omega) = Re \int_0^\infty <a^\dagger(t) a(0)> e^{-i omega t} dt
We use QuTiP two-time correlation function via correlation_2op_1t (qutip.correlation.correlation_2op_1t)
or build evolution by applying superoperator exponentials if needed.
"""
import numpy as np
from qutip import correlation_2op_1t, spectrum_correlation_fft, expect, correlation_3op_1t
from qutip import spre, spost, propagator, liouvillian
from params import wmin, wmax, n_w, gamma_c

def compute_spectrum_via_correlation(H, c_ops, a, rho_ss, wlist=None, tlist=None):
    """
    Use QuTiP's correlation_2op_1t to compute g1(t) = <a^\dagger(t) a(0)>, then FFT -> S(w).
    This is a robust approach, but expensive.
    """
    import numpy as np
    from qutip import correlation_2op_1t

    # time grid for correlations: choose tmax and nt
    if tlist is None:
        tmax = 50.0 / gamma_c    # heuristic; adjust if needed
        nt = 2000
        t = np.linspace(0.0, tmax, nt)
    else:
        t = np.array(tlist)

    # compute two-time correlator: corr = <A(t) B(0)> with A = a.dag(), B = a
    corr = correlation_2op_1t(rho_ss, t, c_ops, [], a.dag(), a)
    # corr is array with shape (len(t),) complex
    # compute Fourier transform S(omega)
    dt = t[1] - t[0]
    G = np.fft.fft(corr) * dt
    freqs = 2 * np.pi * np.fft.fftfreq(len(t), d=dt)
    # shift
    Gs = np.fft.fftshift(G)
    freqs_shift = np.fft.fftshift(freqs)
    # choose desired wlist
    if wlist is None:
        wlist = np.linspace(wmin, wmax, n_w)
    # interpolate real(Re) part
    S = np.interp(wlist, freqs_shift, np.real(Gs))
    # ensure non-negative and return
    S = np.maximum(S, 0.0)
    return wlist, S, t, corr
