import numpy as np
from Old_src.sde_solver import SDESolver


class KerrCavityPositiveP:
    def __init__(self, gamma_c, U, dt, seed=None):
        self.gamma_c = gamma_c
        self.U = U
        self.dt = dt
        self.solver = SDESolver(dt, seed)

    def drift(self, x, a_in, a_in_t):
        a, a_t = x
        return np.array([
            -0.5 * self.gamma_c * a
            - 1j * self.U * a * a * a_t
            + np.sqrt(0.45*self.gamma_c) * a_in,

            -0.5 * self.gamma_c * a_t
            + 1j * self.U * a_t * a_t * a
            + np.sqrt(0.45*self.gamma_c) * a_in_t  # the 0.45 comes from the coupling efficiency
        ])

    def diffusion(self, x):
        a, a_t = x
        return np.array([
            np.sqrt(1j * self.U) * a,
            np.sqrt(-1j * self.U) * a_t
        ])

    def run_stationary(self, a_in, a_in_t, t_relax, t_store, store_period):
        """
        Relaxation + stockage stationnaire
        """
        n_relax = int(t_relax / self.dt)
        n_store = int(t_store / self.dt)
        n_skip = int(store_period / self.dt)

        x = np.zeros(2, dtype=complex)
        output = []

        for i in range(n_relax + n_store):
            x = self.solver.step(
                x,
                drift=lambda y: self.drift(y, a_in, a_in_t),
                diffusion=self.diffusion
            )

            if i >= n_relax and (i - n_relax) % n_skip == 0:  
                # La deuxième condition permet de stocker à intervalles réguliers
                output.append(x[0])  # alpha only

        return np.array(output)
