# utils.py
import numpy as np
from scipy.fft import fft, fftfreq, fftshift

def fourier_transform_time_signal(t, g, w):
    """
    Compute Fourier transform integral int_0^T g(t) e^{-i w t} dt.
    We assume t is 1D array, g same shape. Returns S(w) evaluated at array w.
    Here we use numerical FFT and interpolation.
    """
    # zero-pad / uniform t required. We'll assume t is uniform.
    dt = t[1] - t[0]
    N = len(t)
    G = np.fft.fft(g) * dt
    freqs = 2*np.pi*fftfreq(N, d=dt)
    # arrange as frequency axis symmetric
    Gshift = fftshift(G)
    freqs_shift = fftshift(freqs)
    # interpolate to requested w
    return np.interp(w, freqs_shift, Gshift.real)  # take real part as spectrum

def normalize_moments(M):
    # Rescale each moment vector to its max absolute value (as in figure)
    M = np.asarray(M)
    Mnorm = []
    for col in M.T:
        mx = np.max(np.abs(col))
        if mx == 0:
            Mnorm.append(col)
        else:
            Mnorm.append(col / mx)
    return np.vstack(Mnorm).T
