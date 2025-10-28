import numpy as np
from qutip import Qobj, steadystate, correlation_2op_1t, qeye, expect
from src.params import wmin, wmax, n_w, gamma_c

def compute_spectrum_via_correlation(H, c_ops, a, rho_ss, wlist=None):
    """
    Compute the emission spectrum S(ω) using correlations + FFT,
    and optionally remove a narrow coherent/delta-like peak around ω=0.

    Parameters
    ----------
    H : Qobj
        Hamiltonian of the system
    c_ops : list of Qobj
        Collapse operators
    a : Qobj
        Annihilation operator
    rho_ss : Qobj
        Steady-state density matrix
    wlist : np.ndarray, optional
        Frequencies at which to sample the spectrum (in eV)
    remove_dc : bool, default True
        Remove narrow peak around ω=0
    dc_width : float, default 1e-5
        Width around ω=0 to remove (in eV)

    Returns
    -------
    wlist_out : np.ndarray
        Frequencies (eV)
    S_w : np.ndarray
        Spectrum values
    """

    N = a.shape[0]

    # --- Operator of fluctuations ---
    a_mean = expect(a, rho_ss)
    a_fluct = a - a_mean * qeye(N)

    # --- Time grid for correlation ---
    dt = 0.005 / gamma_c  # small enough for good temporal resolution
    N_t = 2**14             # power-of-2 for FFT efficiency
    tlist = dt * np.arange(N_t)

    # --- Correlation <a_fluct(t) a_fluct^dag(0)> ---
    corr = correlation_2op_1t(H, rho_ss, tlist, c_ops, a_fluct, a_fluct.dag())

    # --- FFT to get spectrum ---
    S_w_full = np.fft.fftshift(np.fft.fft(corr)) * dt
    freq_full = np.fft.fftshift(np.fft.fftfreq(N_t, dt)) * 2 * np.pi  # in eV

    S_w_full = np.abs(S_w_full)

    # --- Restrict to desired wlist ---
    if wlist is None:
        wlist_out = freq_full
        S_w = S_w_full
    else:
        S_w = np.interp(wlist, freq_full, S_w_full)
        wlist_out = wlist

    return wlist_out, S_w
