# src/moments.py
import numpy as np

class MomentsCalculator:
    """
    Calcule les moments spectraux à partir d'un spectre S(ω).
    Moments normalisés : M_m = ∫ S(ω) ω^m dω / ∫ S(ω) dω  pour m ≥ 1
    """

    def __init__(self, max_m=4):
        """
        Args:
            max_m: moment maximum à calculer
        """
        self.max_m = max_m

    def compute(self, omega: np.ndarray, S: np.ndarray) -> np.ndarray:
        """
        Calcule les moments normalisés.
        
        Args:
            omega: tableau des fréquences
            S: spectre S(ω)
        
        Returns:
            M: np.ndarray [M0, M1, ..., M_max_m], où M0 = ∫ S(ω) dω, M1..Mmax = normalisés
        """
        Mtilde = np.array([np.trapz(S * omega**m, omega) for m in range(self.max_m+1)])
        M0 = Mtilde[0]

        if M0 == 0:
            M = np.zeros_like(Mtilde)
        else:
            M = Mtilde.copy()
            M[1:] = Mtilde[1:] / M0
        return M
