import numpy as np
from src.simulators.sde_solver import SDESolver


class OPO_SDE:
    def __init__(self, gamma_s, G, dt, seed=None):
        self.gamma_s = gamma_s
        self.G = G
        self.dt = dt
        self.solver = SDESolver(dt, seed)

    def drift(self, x):
        a, a_t = x
        return np.array([
            -self.gamma_s/2 * a + self.G * a_t,
            -self.gamma_s/2 * a_t + np.conjugate(self.G) * a
        ])

    def diffusion(self, x):
        return np.array([
            np.sqrt(self.G),
            np.sqrt(np.conjugate(self.G))
        ])

    def run_and_sample(self, t_relax, t_sample, sample_period):
        """
        Évolution libre + échantillonnage stationnaire
        """
        n_relax = int(t_relax / self.dt)
        n_total = int((t_relax + t_sample) / self.dt)
        n_skip = int(sample_period / self.dt)

        x = np.zeros(2)
        samples = []

        for i in range(n_total):
            x = self.solver.step(x, self.drift, self.diffusion)

            if i >= n_relax and (i - n_relax) % n_skip == 0:
                samples.append(x.copy())

        return np.array(samples)
