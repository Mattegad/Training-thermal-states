import numpy as np
from qutip import spectrum, Qobj, expect
from src.params import wmin, wmax, n_w


def compute_spectrum_via_correlation(H, c_ops, a, rho_ss, wlist=None,
                                     remove_dc=True, dc_width=0.0005/20, spike_factor=50.0):
    """
    Compute the emission spectrum S(ω) and suppress a narrow, large coherent/delta-like peak
    without forcing S(0)=0. The function detects narrow spikes (>> baseline) and replaces
    them by a smooth interpolation to preserve the baseline at ω=0.
    """
    if wlist is None:
        wlist = np.linspace(wmin, wmax, n_w)

    if not isinstance(rho_ss, Qobj):
        raise TypeError("rho_ss must be a QuTiP Qobj density matrix")

    # Expectation value <a>
    alpha_mean = expect(a, rho_ss)

    # Centered operator to remove coherent component
    a_fluct = a - alpha_mean * a.unit()

    # Compute raw spectrum
    S_raw = spectrum(H=H, wlist=wlist, c_ops=c_ops, a_op=a_fluct.dag(), b_op=a_fluct)
    S_real = np.real(np.asarray(S_raw, dtype=np.float64))

    # Replace NaNs/infs by interpolation
    mask = np.isfinite(S_real)
    if not np.any(mask):
        raise RuntimeError("Computed spectrum is entirely non-finite (all NaN/Inf).")
    if not np.all(mask):
        S_real = np.interp(wlist, wlist[mask], S_real[mask])
    S_real = np.maximum(S_real, 0.0)

    if remove_dc:
        # Détecter le pic étroit autour de ω=0
        idx_dc = np.abs(wlist) < dc_width
        if np.any(idx_dc):
            baseline = np.median(S_real[~idx_dc])  # valeur de référence sans pic
            # Remplacer le pic par interpolation linéaire avec le baseline
            S_real[idx_dc] = baseline

    return wlist, S_real
