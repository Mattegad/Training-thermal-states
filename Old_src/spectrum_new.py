# spectrum_extended.py
import numpy as np
from qutip import correlation_2op_1t, Qobj, expect, qeye
from src.params import wmin, wmax, n_w, gamma_c, G, e, beta, phi

def compute_spectrum_via_correlation(H, c_ops, a, rho_ss, wlist=None, tlist=None, compute_si=False):
    """
    Compute S(ω) = Re ∫ <a†(t)a(0)> e^{-i ω t} dt
    Optionally compute S_{i_-}(ω) as defined in your formula.
    
    Parameters for S_{i_-}:
    - alpha, beta : complex amplitudes
    - zeta, Ge, phi : scalars
    """
    if wlist is None:
        wlist = np.linspace(wmin, wmax, n_w)

    if tlist is None:
        tmax = 50.0 / gamma_c
        nt = 2000
        tlist = np.linspace(0.0, tmax, nt)

    if not isinstance(rho_ss, Qobj):
        raise TypeError("rho_ss must be a QuTiP Qobj density matrix")

    N = a.shape[0]

    # --- Operator of fluctuations ---
    a_mean = expect(a, rho_ss)
    a_fluct = a - a_mean * qeye(N)

    # --- Standard spectrum S(ω) ---
    corr = correlation_2op_1t(
        H=H,
        state0=rho_ss,
        taulist=tlist,
        c_ops=c_ops,
        a_op=a_fluct.dag(),
        b_op=a_fluct
    )

    dt = tlist[1] - tlist[0]
    Ge = np.fft.fft(corr) * dt
    freqs = 2 * np.pi * np.fft.fftfreq(len(tlist), d=dt)
    Gs = np.fft.fftshift(Ge)
    freqs_shift = np.fft.fftshift(freqs)
    S = np.interp(wlist, freqs_shift, np.real(Gs))
    S = np.maximum(S, 0.0)

    # --- Optionally compute S_{i_-}(ω) ---
    if compute_si:

        # Helper: correlation with min/max tau
        tau_m_list = np.minimum(0.0, tlist)
        tau_M_list = np.maximum(0.0, tlist)

        # S_{aa} : <delta a(tau_M) delta a(tau_m)>
        corr_aa = correlation_2op_1t(H, rho_ss, tlist, c_ops, a_fluct, a_fluct)
        S_aa = np.fft.fft(corr_aa) * dt
        S_aa = np.fft.fftshift(S_aa)

        # S_{a†a†} : <delta a†(tau_m) delta a†(tau_M)>
        corr_adagadag = correlation_2op_1t(H, rho_ss, tlist, c_ops, a_fluct.dag(), a_fluct.dag())
        S_adagadag = np.fft.fft(corr_adagadag) * dt
        S_adagadag = np.fft.fftshift(S_adagadag)

        # S_{a†_tau a} : <a†(tau) a(0)>
        corr_adag_a = correlation_2op_1t(H, rho_ss, tlist, c_ops, a_fluct.dag(), a_fluct)
        S_adag_a = np.fft.fft(corr_adag_a) * dt
        S_adag_a = np.fft.fftshift(S_adag_a)

        # S_{a†_0 a} : <a†(0) a(tau)> = conj(<a†(tau) a(0)>)
        S_adag0_a = np.conj(S_adag_a)

        # Interpolation onto wlist
        S_aa_i = np.maximum(np.interp(wlist, freqs_shift, S_aa), 0.0)
        S_aadag_i = np.maximum(np.interp(wlist, freqs_shift, S_adagadag), 0.0)
        S_adag_a_i = np.maximum(np.interp(wlist, freqs_shift, S_adag_a), 0.0)
        S_adag0_a_i = np.maximum(np.interp(wlist, freqs_shift, S_adag0_a), 0.0)

        # δ(ω) term approximation: use first bin (could refine)
        #delta_term = -2 * (G*e)**2 * np.imag(np.conj(alpha)*beta)
        #constant_term = -2 * (G*e**2) * zeta * np.imag(np.conj(alpha)*beta)

        S_i_minus = (G*e * np.abs(beta))**2 * (
                        np.exp(2j*(phi + np.pi/2)) * S_aa_i +
                        np.exp(-2j*(phi + np.pi/2)) * S_aadag_i +
                        S_adag_a_i +
                        S_adag0_a_i
                    )
        S_i_minus = np.real(S_i_minus)  # keep only real part

        return wlist, S, S_i_minus

    return wlist, S
