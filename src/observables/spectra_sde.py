import numpy as np


class SpectraSDE:
    """
    Calcul du spectre d'émission à partir de trajectoires SDE
    (cas OPO réel + cavité Kerr)
    """

    def __init__(self, dt, max_tau, n_bins=1024):
        """
        dt       : pas de temps SDE
        max_tau  : temps maximal pour G1(tau)
        n_bins   : résolution fréquentielle
        """
        self.dt = dt
        self.max_tau = max_tau
        self.n_bins = n_bins

        self.tau = np.arange(0, max_tau, dt)

    # ------------------------------------------------------------------
    # 1) Fonction de cohérence G^(1)
    # ------------------------------------------------------------------
    def first_order_coherence(self, a_traj):
        """
        Calcule G^(1)(tau) = <a*(t+tau)a(t)>
        à partir d'une trajectoire stationnaire
        """
        n_tau = len(self.tau)
        G1 = np.zeros(n_tau, dtype=complex)

        T = len(a_traj)
        # Boucle sur les décalages temporels tau
        for i, tau_idx in enumerate(range(n_tau)):
            shift = tau_idx
            if shift >= T:
                break
            prod = np.conj(a_traj[shift:]) * a_traj[:T-shift]
            G1[i] = np.mean(prod)

        return G1

    # ------------------------------------------------------------------
    # 2) Spectre par transformée de Fourier
    # ------------------------------------------------------------------
    def spectrum_from_G1(self, G1):
        """
        Transformée de Fourier réelle du G1
        """
        S = np.real(np.fft.fft(G1, n=self.n_bins))
        freqs = np.fft.fftfreq(self.n_bins, d=self.dt)
        return freqs, S

    # ------------------------------------------------------------------
    # 3) Spectre direct depuis trajectoire
    # ------------------------------------------------------------------
    def spectrum(self, a_traj):
        """
        Pipeline complet : trajectoire -> spectre
        """
        G1 = self.first_order_coherence(a_traj)
        return self.spectrum_from_G1(G1)

    # ------------------------------------------------------------------
    # 4) Soustraction du spectre de référence (Fig. 6)
    # ------------------------------------------------------------------
    def differential_spectrum(self, a_traj, a_ref):
        """
        S(ω) - S_ref(ω)
        """
        w, S = self.spectrum(a_traj)
        _, Sref = self.spectrum(a_ref)
        return w, S - Sref

    # ------------------------------------------------------------------
    # 5) Moments spectraux
    # ------------------------------------------------------------------
    def spectral_moments(self, w, S, max_order=4):
        """
        Calcule les moments M_0 ... M_max_order
        """
        moments = []

        dw = w[1] - w[0]

        for m in range(max_order + 1):
            M = np.sum(S * (w ** m)) * dw
            moments.append(M)

        moments = np.array(moments)

        # normalisation comme dans l'article
        if moments[0] != 0:
            moments[1:] /= moments[0]

        return moments
