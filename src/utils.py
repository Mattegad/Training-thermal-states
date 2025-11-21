# src/utils.py
import numpy as np
from scipy.fft import fft, fftfreq, fftshift

def fourier_transform_time_signal(t: np.ndarray, g: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Calcule la transformée de Fourier numérique d'un signal g(t) sur l'axe de temps t.
    
    Args:
        t: array temps, uniforme
        g: array du signal g(t)
        w: array de fréquences souhaitées pour l'évaluation
    
    Returns:
        S(w): transformée de Fourier interpolée aux fréquences w
    """
    dt = t[1] - t[0]
    N = len(t)
    G = fft(g) * dt
    freqs = 2 * np.pi * fftfreq(N, d=dt)
    Gshift = fftshift(G)
    freqs_shift = fftshift(freqs)
    return np.interp(w, freqs_shift, Gshift.real)

def normalize_moments(M: np.ndarray) -> np.ndarray:
    """
    Normalise chaque colonne d'une matrice de moments par sa valeur absolue maximale.
    
    Args:
        M: array shape (n_samples, n_moments)
    
    Returns:
        Mnorm: array normalisé (même shape)
    """
    M = np.asarray(M)
    Mnorm = []
    for col in M.T:
        mx = np.max(np.abs(col))
        if mx == 0:
            Mnorm.append(col)
        else:
            Mnorm.append(col / mx)
    return np.vstack(Mnorm).T
