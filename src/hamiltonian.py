# src/hamiltonian.py
from qutip import Qobj, destroy

def create_operators(N=30):
    """Créer l'opérateur annihilation et l'identité."""
    a = destroy(N)
    Id = a * 0 + 1  # équivalent à qeye(N)
    return a, Id


class HamiltonianFactory:
    """Factory pour créer différents Hamiltoniens."""
    
    @staticmethod
    def hamiltonian_kerr(a, U_val):
        """H = U/2 * a† a† a a"""
        return 0.5 * U_val * (a.dag() * a.dag() * a * a)

    @staticmethod
    def make(kind: str, a, **params) -> Qobj:
        if kind == "kerr":
            U_val = params.get("U_val", 12e-6)
            return HamiltonianFactory.hamiltonian_kerr(a, U_val)
        else:
            raise ValueError(f"Hamiltonian type '{kind}' non supporté")


