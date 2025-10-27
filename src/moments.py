# moments.py
import numpy as np

def compute_moments(omega, S, max_m=4):
    """
    Compute raw moments Mtilde_m = \int S(omega) * omega^m d omega
    then normalized moments Mm = Mtilde_m / Mtilde_0 for m != 0.
    Return array [M0, M1, M2, ... , M_max_m], where M0 = Mtilde_0.
    """
    domega = omega[1] - omega[0]
    Mtilde = []
    for m in range(max_m+1):
        Mtilde_m = np.trapz(S * (omega**m), omega)
        Mtilde.append(Mtilde_m)
    Mtilde = np.array(Mtilde)
    M0 = Mtilde[0]
    if M0 == 0:
        M = np.zeros_like(Mtilde)
    else:
        M = Mtilde.copy()
        for m in range(1, max_m+1):
            M[m] = Mtilde[m] / M0
    return M  # M[0] = M0, M[1..] = normalized moments
