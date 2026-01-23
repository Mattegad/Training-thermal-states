import numpy as np
from numpy.fft import fft, ifft, fftshift
from numpy.linalg import expm
from scipy.optimize import fsolve

class OPOField:
    """Simule la dynamique de l'OPO."""
    def __init__(self, gammaOPO=10.0, Ntest=10, eps1=0.7, weight=1.0, angle=0.0, seed=None):
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
        self.relevanttimes = self.timearr >= (20 / self.gammaOPO)  # ignore first 20/gammaOPO
        self.Nrelt = np.sum(self.relevanttimes)
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

        delp = np.zeros(Ntraj, dtype=np.complex128) # initialisation des fluctuations  de psi à t=0
        delcptil = np.zeros(Ntraj, dtype=np.complex128) # initialisation des fluctuations de psitilde à t=0

        dt = 1e-2 / self.gammacav  # pas de temps d'intégration
        halfdt = 0.5*dt
        tstep = 0.5 / self.gammacav  # pas de stockage
        cavtimearray = np.arange(100/self.gammacav, 300/self.gammacav, tstep)  # On stocke entre 100/gammacav et 300/gammacav
        Nrelt = len(cavtimearray)

        # stockage de psi fluctuations \delta\psi(j,t_n), j=traj, t_n=time  
        delpresultarray = np.zeros((Ntraj, Nrelt), dtype=np.complex128)  
        delcptilresultarray = np.zeros((Ntraj, Nrelt), dtype=np.complex128)

        time = 0.0
        while time < cavtimearray[-1]:
            """Calcul des fluctuations via l'intégration SDE semi-implicite exponentielle."""
            xi = np.random.randn(Ntraj)/np.sqrt(dt)
            xitil = np.random.randn(Ntraj)/np.sqrt(dt)

            # partie linéaire exponentiable
            d_delp_lin = (0.5*(-self.gammacav) - 1j*self.Ufreq*(2*np.abs(Psi)**2 + delcptil*delp + 2*Psi*delcptil + np.conj(Psi)*delp) + np.sqrt(-1j*self.Ufreq)*xi + 0.5j*self.Ufreq)
            # partie restante
            d_delp_rest = (np.sqrt(self.eps_p*self.gammacav)*delp_in - 1j*self.Ufreq*Psi**2*delcptil + np.sqrt(-1j*self.Ufreq)*Psi*xi + 0.5j*self.Ufreq*Psi)
            d_delcptil_lin = (0.5*(-self.gammacav) + 1j*self.Ufreq*(2*np.abs(Psi)**2 + delcptil*delp + 2*np.conj(Psi)*delp + Psi*delcptil) + np.sqrt(1j*self.Ufreq)*xitil - 0.5j*self.Ufreq)
            d_delcptil_rest = (np.sqrt(self.eps_p*self.gammacav)*delcptil_in + 1j*self.Ufreq*np.conj(Psi)**2*delp + np.sqrt(1j*self.Ufreq)*np.conj(Psi)*xitil - 0.5j*self.Ufreq*np.conj(Psi))

            middelp = ((delp + d_delp_rest/d_delp_lin)*np.exp(d_delp_lin*halfdt) - d_delp_rest/d_delp_lin)
            middelcptil = ((delcptil + d_delcptil_rest/d_delcptil_lin)*np.exp(d_delcptil_lin*halfdt) - d_delcptil_rest/d_delcptil_lin)

            d_delp_lin = (0.5*(-self.gammacav) - 1j*self.Ufreq*(2*np.abs(Psi)**2 + middelcptil*middelp + 2*Psi*middelcptil + np.conj(Psi)*middelp) + np.sqrt(-1j*self.Ufreq)*xi + 0.5j*self.Ufreq)
            d_delp_rest = (np.sqrt(self.eps_p*self.gammacav)*delp_in - 1j*self.Ufreq*Psi**2*middelcptil + np.sqrt(-1j*self.Ufreq)*Psi*xi + 0.5j*self.Ufreq*Psi)
            d_delcptil_lin = (0.5*(-self.gammacav) + 1j*self.Ufreq*(2*np.abs(Psi)**2 + middelcptil*middelp + 2*np.conj(Psi)*middelp + Psi*middelcptil) + np.sqrt(1j*self.Ufreq)*xitil - 0.5j*self.Ufreq)
            d_delcptil_rest = (np.sqrt(self.eps_p*self.gammacav)*delcptil_in + 1j*self.Ufreq*np.conj(Psi)**2*middelp + np.sqrt(1j*self.Ufreq)*np.conj(Psi)*xitil - 0.5j*self.Ufreq*np.conj(Psi))

            delp = ((delp + d_delp_rest/d_delp_lin)*np.exp(d_delp_lin*dt) - d_delp_rest/d_delp_lin)
            delcptil = ((delcptil + d_delcptil_rest/d_delcptil_lin)*np.exp(d_delcptil_lin*dt) - d_delcptil_rest/d_delcptil_lin)

            time += dt
            # Stockage des résultats aux temps spécifiés
            idx = np.where(np.abs(cavtimearray - time) < dt/2)[0]
            if idx.size > 0:
                delpresultarray[:, idx[0]] = delp
                delcptilresultarray[:, idx[0]] = delcptil

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
        opticalspecarr = np.abs(fftshift(ifft(cavg1, axis=1), axes=1))
        cavomegaarr = np.linspace(-np.pi/tstep, np.pi/tstep - 2*np.pi/(tstep*Nrelt), Nrelt)
        # Moyenne du spectre optique sur les trajectoires
        meanopticalspecarr = np.nanmean(opticalspecarr, axis=0)

        return psi_out, cpsitil_out, cavtimearray, meancavg1, cavomegaarr, meanopticalspecarr, Convratio


class SpectralAnalysis:
    """Analyse les spectres et prédit G via ridge regression."""
    def __init__(self, spectra, cavomegaarr, depvararray, Ntrain=10, ridgepar=0.01):
        self.spectra = spectra
        self.cavomegaarr = cavomegaarr
        self.depvararray = depvararray
        self.Ntrain = Ntrain
        self.ridgepar = ridgepar

    def compute_moments(self):
        S = self.spectra.copy()
        S[:, np.abs(self.cavomegaarr) < 1e-10] = 0.0
        S[:, 0] = 0.0
        M0 = np.nanmean(S, axis=1)
        M1 = np.nanmean(S*self.cavomegaarr, axis=1)/M0
        M2 = np.nanmean(S*self.cavomegaarr**2, axis=1)/M0
        M3 = np.nanmean(S*self.cavomegaarr**3, axis=1)/M0
        M4 = np.nanmean(S*self.cavomegaarr**4, axis=1)/M0
        self.moments = [M0, M1, M2, M3, M4]

    def ridge_regression(self):
        uraw = np.vstack([np.ones(self.Ntrain)] + [self.moments[m][:self.Ntrain] for m in range(5)])
        self.W = self.depvararray[:self.Ntrain] @ uraw.T @ np.linalg.inv(uraw @ uraw.T + self.ridgepar*np.eye(uraw.shape[0]))

    def predict(self):
        vraw = np.vstack([np.ones(len(self.depvararray))] + [self.moments[m] for m in range(5)])
        self.prediction = self.W @ vraw
        return self.prediction
