import numpy as np
from numpy.fft import fft, ifft, fftshift
from scipy.optimize import fsolve
from numba import njit
from scipy.linalg import expm
from src.config import NumericalParams



@njit(fastmath=True, parallel=True)
def microcavity_sde_kernel(
    delp_in, delcptil_in,
    Psi, Ufreq, gammacav, eps_p,
    cavtimearray, dt
):
    """
    Intègre les équations stochastiques de la microcavité.
    
    Dimensions :
    - delp_in, delcptil_in : (Ntraj,)
    - sortie : (Ntraj, Nrelt)
    """

    Ntraj = delp_in.size
    Nrelt = cavtimearray.size

    delp = np.zeros(Ntraj, np.complex128) # initialisation des fluctuations  de psi à t=0
    delcptil = np.zeros(Ntraj, np.complex128) # initialisation des fluctuations de psitilde à t=0

    # stockage de psi fluctuations \delta\psi(j,t_n), j=traj, t_n=time
    delpresultarray = np.zeros((Ntraj, Nrelt), np.complex128)
    delcptilresultarray = np.zeros((Ntraj, Nrelt), np.complex128)

    halfdt = 0.5 * dt
    time = 0.0

    for k in range(Nrelt):
        target_time = cavtimearray[k]

        while time < target_time:

            xi = np.random.randn(Ntraj) / np.sqrt(dt)
            xitil = np.random.randn(Ntraj) / np.sqrt(dt)

            # ============================
            # Première demi-étape
            # ============================
            d_delp_lin = (
                0.5 * (-gammacav)
                - 1j * Ufreq * (
                    2*np.abs(Psi)**2
                    + delcptil*delp
                    + 2*Psi*delcptil
                    + np.conj(Psi)*delp
                )
                + np.sqrt(-1j * Ufreq) * xi
                + 0.5j * Ufreq
            )

            d_delp_rest = (
                np.sqrt(eps_p * gammacav) * delp_in
                - 1j * Ufreq * Psi**2 * delcptil
                + np.sqrt(-1j * Ufreq) * Psi * xi
                + 0.5j * Ufreq * Psi
            )

            d_delcptil_lin = (
                0.5 * (-gammacav)
                + 1j * Ufreq * (
                    2*np.abs(Psi)**2
                    + delcptil*delp
                    + 2*np.conj(Psi)*delp
                    + Psi*delcptil
                )
                + np.sqrt(1j * Ufreq) * xitil
                - 0.5j * Ufreq
            )

            d_delcptil_rest = (
                np.sqrt(eps_p * gammacav) * delcptil_in
                + 1j * Ufreq * np.conj(Psi)**2 * delp
                + np.sqrt(1j * Ufreq) * np.conj(Psi) * xitil
                - 0.5j * Ufreq * np.conj(Psi)
            )

            middelp = (
                (delp + d_delp_rest/d_delp_lin)
                * np.exp(d_delp_lin * halfdt)
                - d_delp_rest/d_delp_lin
            )

            middelcptil = (
                (delcptil + d_delcptil_rest/d_delcptil_lin)
                * np.exp(d_delcptil_lin * halfdt)
                - d_delcptil_rest/d_delcptil_lin
            )

            # ============================
            # Seconde demi-étape
            # ============================
            d_delp_lin = (
                0.5 * (-gammacav)
                - 1j * Ufreq * (
                    2*np.abs(Psi)**2
                    + middelcptil*middelp
                    + 2*Psi*middelcptil
                    + np.conj(Psi)*middelp
                )
                + np.sqrt(-1j * Ufreq) * xi
                + 0.5j * Ufreq
            )

            d_delp_rest = (
                np.sqrt(eps_p * gammacav) * delp_in
                - 1j * Ufreq * Psi**2 * middelcptil
                + np.sqrt(-1j * Ufreq) * Psi * xi
                + 0.5j * Ufreq * Psi
            )

            d_delcptil_lin = (
                0.5 * (-gammacav)
                + 1j * Ufreq * (
                    2*np.abs(Psi)**2
                    + middelcptil*middelp
                    + 2*np.conj(Psi)*middelp
                    + Psi*middelcptil
                )
                + np.sqrt(1j * Ufreq) * xitil
                - 0.5j * Ufreq
            )

            d_delcptil_rest = (
                np.sqrt(eps_p * gammacav) * delcptil_in
                + 1j * Ufreq * np.conj(Psi)**2 * middelp
                + np.sqrt(1j * Ufreq) * np.conj(Psi) * xitil
                - 0.5j * Ufreq * np.conj(Psi)
            )

            delp = (
                (delp + d_delp_rest/d_delp_lin)
                * np.exp(d_delp_lin * dt)
                - d_delp_rest/d_delp_lin
            )

            delcptil = (
                (delcptil + d_delcptil_rest/d_delcptil_lin)
                * np.exp(d_delcptil_lin * dt)
                - d_delcptil_rest/d_delcptil_lin
            )

            time += dt

        delpresultarray[:, k] = delp
        delcptilresultarray[:, k] = delcptil

    return delpresultarray, delcptilresultarray


class OPOField:
    """Simule la dynamique de l'OPO."""
    def __init__(self, gammaOPO=10.0, Ntest=10, eps1=0.7, weight=1.0, angle=0.0, seed=NumericalParams.seed):
        self.gammaOPO = gammaOPO
        self.Ntest = Ntest
        self.eps1 = eps1
        self.weight = weight
        self.angle = angle
        self.rng = np.random.default_rng(seed)

    def generate_G_array(self):
        """Génère un tableau de valeurs de G pour la simulation."""
        self.Garr = np.concatenate([np.arange(0.0, 5.0, 0.5),
                                    4.5 * self.rng.random(self.Ntest)])
        self.Nstates = len(self.Garr)
        return self.Garr

    def simulate_trajectories(self, Ntraj=1000, tmax_mult=1020):
        self.Ntraj = Ntraj
        self.tstep = 1 / self.gammaOPO
        tmin = 0.0
        tmax = tmax_mult / self.gammaOPO - self.tstep
        self.timearr = np.arange(tmin, tmax + self.tstep, self.tstep)
        self.relevanttimes = self.timearr >= (20 / self.gammaOPO)  # ignore first 20/gammaOPO, tableau booléen de 0 et 1
        self.Nrelt = np.sum(self.relevanttimes)   # nombre de temps pertinents
        self.omegaarr = np.linspace(
            -np.pi / self.tstep,
            np.pi / self.tstep - 2*np.pi/(self.tstep*self.Nrelt),
            self.Nrelt
        )

    def run_for_G(self, G, y0=None, numstepsperoutput=100):
        # Matrices de drift et diffusion
        A = np.array([
            [-0.5*self.gammaOPO, G],
            [np.conj(G), -0.5*self.gammaOPO]
        ], dtype=np.complex128)
        B = np.array([
            [np.sqrt(G), 0],
            [0, np.sqrt(np.conj(G))]
        ], dtype=np.complex128)

        if y0 is None:
            y0 = np.zeros((2, self.Ntraj), dtype=np.complex128)  # conditions initiales

        dt = self.tstep / numstepsperoutput  # pas de temps interne d'intégration
        yout = np.full((2, self.Ntraj, len(self.timearr)), np.nan, dtype=np.complex128) 
        y = y0.copy()
        yout[:, :, 0] = y  # stocke la sortie initiale (t=0) pour toutes les trajectoires et pour phi et cphitilde

        for tt1 in range(len(self.timearr)-1):
            for _ in range(numstepsperoutput):
                noise = self.rng.standard_normal((2, self.Ntraj)) * np.sqrt(dt)
                y = expm(A*dt) @ y + B @ noise
            yout[:, :, tt1+1] = y  # stocke la sortie tous les 1/gammaOPO

        phis_out = yout[0, :, self.relevanttimes]
        cphitils_out = yout[1, :, self.relevanttimes]
        return phis_out, cphitils_out


class Microcavity:
    """Simule la dynamique d'une microcavité avec entrée OPO."""
    def __init__(self, U=12.0, gammacav=67.0, eps_p=0.45, unitrescaling=1000.0, hbar=0.6582):
        self.U = U
        self.gammacav = gammacav
        self.eps_p = eps_p
        self.unitrescaling = unitrescaling
        self.hbar = hbar
        self.Ufreq = U / hbar

    def simulate(self, psi_in, cpsitil_in, Psi_in, filename="microcavity"):
        # Normalisation des entrées qui vont dépendre des sorties de l'OPO
        psi_in = psi_in / np.sqrt(self.unitrescaling)
        cpsitil_in = cpsitil_in / np.sqrt(self.unitrescaling)
        Psi_in = Psi_in / np.sqrt(self.unitrescaling)

        inshape = psi_in.shape
        delp_in = psi_in.flatten() - Psi_in  # on ne garde que la partie fluctuante de psi_in
        delcptil_in = cpsitil_in.flatten() - np.conj(Psi_in)  # idem pour cpsitil_in
        Ntraj = delp_in.size

        def GPESS(XP):
            """Gross Pitaevskii Equation Steady State Solver."""
            X, Y = XP
            dX = (-0.5*self.gammacav*X + self.Ufreq*(X**2 + Y**2)*Y + np.sqrt(self.gammacav*self.eps_p)*np.real(Psi_in))
            dY = (-0.5*self.gammacav*Y - self.Ufreq*(X**2 + Y**2)*X + np.sqrt(self.gammacav*self.eps_p)*np.imag(Psi_in))
            return [dX, dY]

        XPmf = fsolve(GPESS, [np.real(Psi_in), np.imag(Psi_in)]) # Calcul de l'état stationnaire moyen
        Psi = XPmf[0] + 1j*XPmf[1]  # État stationnaire moyen complexe 

        dt = 1e-2 / self.gammacav  # pas de temps d'intégration
        tstep = 0.5 / self.gammacav  # pas de stockage
        cavtimearray = np.arange(100/self.gammacav, 300/self.gammacav, tstep)  # On stocke entre 100/gammacav et 300/gammacav
        Nrelt = len(cavtimearray)

        delpresultarray, delcptilresultarray = microcavity_sde_kernel(
            delp_in, delcptil_in,
            Psi, self.Ufreq, self.gammacav, self.eps_p,
            cavtimearray, dt
        )
        
        Convratio = np.sum(np.isfinite(delpresultarray[:, -1] + delcptilresultarray[:, -1])) / Ntraj  # ratio de convergence
        # Reconstruction des sorties complètes (moyenne temporelle des fluctuations + état moyen)
        psi_out = (np.mean(delpresultarray, axis=1) + Psi).reshape(inshape) * np.sqrt(self.gammacav*self.unitrescaling)
        cpsitil_out = (np.mean(delcptilresultarray, axis=1) + np.conj(Psi)).reshape(inshape) * np.sqrt(self.gammacav*self.unitrescaling)

        # TF du champ fluctuant pour le calcul du spectre optique
        psi_f = fft(delpresultarray - np.mean(delpresultarray, axis=1, keepdims=True), axis=1)
        psitil_f = fft(delcptilresultarray - np.mean(delcptilresultarray, axis=1, keepdims=True), axis=1)
        # Calcul des corrélations par convolution
        cavg1 = ifft(psi_f * np.conj(psitil_f), axis=1)
        # Moyenne de g1 sur les trajectoires
        meancavg1 = np.nanmean(cavg1, axis=0)
        # Spectre optique pour chaque trajectoire
        opticalspecarr = np.abs(fftshift(ifft(cavg1))) * tstep
        cavomegaarr = np.linspace(-np.pi/tstep, np.pi/tstep - 2*np.pi/(tstep*Nrelt), Nrelt)
        # Moyenne du spectre optique sur les trajectoires
        meanopticalspecarr = np.nanmean(opticalspecarr, axis=0)
        # symétrisation et annulation en omega=0
        meanopticalspecarr[np.abs(cavomegaarr) < 1e-10] = 0.0
        meanopticalspecarr[0] = 0.0

        return psi_out, cpsitil_out, cavtimearray, meancavg1, cavomegaarr, opticalspecarr, meanopticalspecarr, Convratio


class SpectralAnalysis:
    """
    Analyse les spectres et prédit G via ridge regression.

    mode:
        - "raw"      : moments du spectre brut
        - "absdiff"  : moments de S - S_ref
        - "reldiff"  : moments de (S - S_ref) / S_ref
        - "centered" : moments centrés autour de M1
    """
    def __init__(
        self,
        spectra,
        cavomegaarr,
        depvararray,
        Ntrain=10,
        ridgepar=1e-14,
        mode="raw",
        ref_index=0,
    ):
        self.spectra = spectra
        self.cavomegaarr = cavomegaarr
        self.depvararray = depvararray
        self.Ntrain = Ntrain
        self.ridgepar = ridgepar
        self.mode = mode
        self.ref_index = ref_index  # spectre de référence (MATLAB: spectra(1,:))

    def compute_moments(self):
        S = self.spectra.copy()

        # ----- choix du mode -----
        if self.mode == "raw":
            Suse = S

        elif self.mode == "absdiff":
            Sref = S[self.ref_index, :]
            Suse = S - Sref

        elif self.mode == "reldiff":
            Sref = S[self.ref_index, :]
            Suse = (S - Sref[None, :]) / Sref[None, :]
        elif self.mode == "centered":
            # centré autour de la moyenne fréquentielle
            M0_tmp = np.nanmean(S, axis=1)
            M1_tmp = np.nanmean(S * self.cavomegaarr, axis=1) / M0_tmp
            omega_c = self.cavomegaarr[None, :] - M1_tmp[:, None]
            Suse = S
        else:
            raise ValueError(f"mode inconnu: {self.mode}")

        # ----- calcul des moments -----
        M0 = np.nanmean(Suse, axis=1)

        if self.mode == "centered":
            M1 = np.zeros_like(M0)
            M2 = np.nanmean(Suse * omega_c**2, axis=1) / M0
            M3 = np.nanmean(Suse * omega_c**3, axis=1) / M0
            M4 = np.nanmean(Suse * omega_c**4, axis=1) / M0
        else:
            M1 = np.nanmean(Suse * self.cavomegaarr, axis=1) / M0
            M2 = np.nanmean(Suse * self.cavomegaarr**2, axis=1) / M0
            M3 = np.nanmean(Suse * self.cavomegaarr**3, axis=1) / M0
            M4 = np.nanmean(Suse * self.cavomegaarr**4, axis=1) / M0

        self.moments = np.vstack([M0, M1, M2, M3, M4])

    def ridge_regression(self):
        M0, M1, M2, M3, M4 = self.moments

        finite = (
            np.isfinite(M0)
            & np.isfinite(M1)
            & np.isfinite(M2)
            & np.isfinite(M3)
            & np.isfinite(M4)
        )

        idx = np.where(finite)[0]
        idx_train = idx[:self.Ntrain]
        idx_test = idx[self.Ntrain:]

        u = np.vstack([
            np.ones(len(idx_train)),
            M0[idx_train],
            M1[idx_train],
            M2[idx_train],
            M3[idx_train],
            M4[idx_train],
        ])

        y = self.depvararray[idx_train]

        self.W = (
            y @ u.T
            @ np.linalg.inv(u @ u.T + self.ridgepar * np.eye(u.shape[0]))
        )

        self._idx_train = idx_train
        self._idx_test = idx_test

    def predict(self):
        M0, M1, M2, M3, M4 = self.moments
        idx_test = self._idx_test

        v = np.vstack([
            np.ones(len(idx_test)),
            M0[idx_test],
            M1[idx_test],
            M2[idx_test],
            M3[idx_test],
            M4[idx_test],
        ])

        ypred = self.W @ v

        self.prediction = np.full_like(self.depvararray, np.nan, dtype=float)
        self.prediction[idx_test] = ypred

    def compute_nrmse(self):
        """
        Compute NRMSE on test points only (where prediction is finite)
        """
        y_true = self.depvararray
        y_pred = self.prediction
        # masque des points prédits
        mask = np.isfinite(y_pred)

        if np.sum(mask) == 0:
            raise RuntimeError("No valid prediction points to compute NRMSE.")
        err2 = (y_pred[mask] - y_true[mask])**2
        nrmse = np.sqrt(
            np.mean(err2) / np.mean(y_true[mask]**2)
        )

        return nrmse

    def get_results(self):
        return {
            "G_true": self.depvararray,
            "G_pred": self.prediction,
            "moments": self.moments,
            "weights": self.W,
            "mode": self.mode,
        }

