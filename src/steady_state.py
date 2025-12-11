# src/steady_state.py
from qutip import Qobj, liouvillian, steadystate


class SteadyStateSolver:
    """
    Calcule l'état stationnaire rho_ss pour un Hamiltonien H et des opérateurs de dissipation c_ops.
    """

    def __init__(self, H: Qobj, c_ops: list):
        """
        Args:
            H: Hamiltonien du système (Qobj)
            c_ops: Liste d'opérateurs de dissipation (Lindblad)
        """
        self.H = H
        self.c_ops = c_ops

    def compute(self) -> Qobj:
        """
        Retourne l'état stationnaire rho_ss.
        Utilise QuTiP steadystate.
        """
        return steadystate(self.H, self.c_ops)

    def compute_liouvillian(self):
        """
        Retourne le Liouvillien L = -i[H, .] + sum_i D[c_i]
        Utile si on veut manipuler directement L.
        """
        return liouvillian(self.H, self.c_ops)
