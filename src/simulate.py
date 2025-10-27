# simulate.py
"""
Construct Hamiltonian, collapse operators for a single Kerr mode coupled
to a squeezed + displaced thermal environment (as in Sec. 4.1 of the paper)
and compute steady state.
"""
import numpy as np
from qutip import destroy, qeye, tensor, steadystate, spre, spost, liouvillian
from qutip import expect
from params import Ncut, U, gamma_c, alphaD_default, theta_default, nbar_default

def create_operators(N=Ncut):
    a = destroy(N)
    Id = qeye(N)
    return a, Id

def bogoliubov_b(a, alpha=alphaD_default, r=0.0, theta=0.0):
    """
    Build operator b = (a - alpha) cosh(r) + e^{2 i theta} (a.dag - alpha*) sinh(r)
    Note: a.dag() returns qutip operator.
    """
    from qutip import qeye
    cosh = np.cosh(r)
    sinh = np.sinh(r)
    b = (a - alpha * qeye(a.shape[0])) * cosh + np.exp(2j*theta) * (a.dag() - np.conjugate(alpha) * qeye(a.shape[0])) * sinh
    return b

def collapse_ops_for_squeezed_env(a, alpha=alphaD_default, r=0.0, theta=0.0, nbar=nbar_default, gamma=gamma_c):
    """
    Following Sec. 4.1, use Lindblad operators sqrt(kappa) b and sqrt(R) b.dag
    with R = nbar/(nbar+1) * kappa. We set kappa = gamma for simplicity.
    """
    from qutip import destroy
    kappa = gamma
    R = (nbar / (nbar + 1.0)) * kappa if nbar > 0 else 0.0
    b = bogoliubov_b(a, alpha=alpha, r=r, theta=theta)
    c_ops = []
    if kappa > 0:
        c_ops.append(np.sqrt(kappa) * b)
    if R > 0:
        c_ops.append(np.sqrt(R) * b.dag())
    # Also include usual photon loss sqrt(gamma) a
    c_ops.append(np.sqrt(gamma) * a)
    return c_ops

def hamiltonian_kerr(a, U_val=U):
    # H = U/2 * a^\dagger a^\dagger a a
    H = 0.5 * U_val * (a.dag() * a.dag() * a * a)
    return H

def compute_steady_state(H, c_ops):
    L = liouvillian(H, c_ops)
    rho_ss = steadystate(H, c_ops)  # QuTiP solves steady state
    return rho_ss
