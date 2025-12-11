# src/dissipation.py
from typing import List
from qutip import Qobj, qeye
import numpy as np


def bogoliubov_b(a, alpha=5.0, r=0.0, theta=0.0):
    """Construire l'opérateur b = (a - alpha) cosh(r) + e^{2iθ} (a† - alpha*) sinh(r)."""
    N = a.shape[0]
    Id = qeye(N)
    cosh = np.cosh(r)
    sinh = np.sinh(r)
    b = (a - alpha * Id) * cosh + np.exp(2j*theta) * (a.dag() - np.conjugate(alpha) * Id) * sinh
    return b


def collapse_ops_for_squeezed_env(a, alpha=5.0, r=0.0, theta=0.0, nbar=0.0, gamma=0.1) -> List[Qobj]:
    """Créer les opérateurs de dissipation pour un environnement squeezé + thermique."""
    kappa = gamma
    R = (nbar / (nbar + 1.0)) * kappa if nbar > 0 else 0.0
    b = bogoliubov_b(a, alpha=alpha, r=r, theta=theta)
    c_ops = []
    if kappa > 0:
        c_ops.append(np.sqrt(kappa) * b)
    if R > 0:
        c_ops.append(np.sqrt(R) * b.dag())
    return c_ops


class DissipationFactory:
    """Factory pour créer les opérateurs de dissipation selon l'environnement."""
    
    @staticmethod
    def make(kind: str, a, **params) -> List[Qobj]:
        if kind == "squeezed":
            alpha = params.get("alpha", 5.0)
            r = params.get("r", 0.0)
            theta = params.get("theta", 0.0)
            nbar = params.get("nbar", 0.0)
            gamma = params.get("gamma", 0.1)
            return collapse_ops_for_squeezed_env(a, alpha, r, theta, nbar, gamma)
        else:
            raise ValueError(f"Environment type '{kind}' non supporté")
