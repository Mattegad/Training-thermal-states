import numpy as np


class SDESolver:
    """
    Intégrateur Euler–Maruyama générique
    """

    def __init__(self, dt, seed=None):
        self.dt = dt
        if seed is not None:
            np.random.seed(seed)

    def step(self, x, drift, diffusion):
        """
        Un pas Euler–Maruyama
        """
        dW = np.sqrt(self.dt) * np.random.randn(*x.shape)
        return x + drift(x) * self.dt + diffusion(x) * dW

    def integrate(self, x0, drift, diffusion, n_steps):
        """
        Intègre une trajectoire SDE
        """
        dim = len(x0)
        traj = np.zeros((n_steps, dim))
        x = np.array(x0, dtype=float)

        for i in range(n_steps):
            traj[i] = x
            x = self.step(x, drift, diffusion)

        return traj
