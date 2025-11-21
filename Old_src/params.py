import numpy as np
# Paramètres physiques et numériques utilisés pour reproduire Fig.2

# Conversion eV to s^-1
eV_to_Hz = 2.42e14

# Physical parameters (use values similar to the paper)
U = 12e-6   # nonlinearity (eV) used in paper; here unitless scaling is fine
gamma_c = 67e9/eV_to_Hz/6.28      # decay rate (eV) -> choose units consistent with simulation
hbar = 1.0
phi = -np.pi/2  # homodyne phase
G = 1  # gain factor
eta = 0.4  # detection efficiency
e = 1.6e-19  # electron charge (C)
beta = 1e4  # local oscillator amplitude

# Numerical parameters
Ncut = 30           # truncation of Fock space; increase if needed
wmin = -1e-3        # frequency grid min (eV)
wmax = 1e-3          # frequency grid max (eV)
n_w = 8001           # number of frequency points for S(omega)
tlist = None        # if needed for time evolution (we use Liouvillian exponentials)
max_iter_steady = 10000
n_train = 10         # number of training states
n_test = 10          # number of testing states

# Bogoliubov / squeezed-env parameters default (we will vary r)
alphaD_default = 5.0   # coherent displacement
theta_default = 0.0    # squeezing angle
nbar_default = 0.0     # thermal occupation

# Simulation options
use_sparse = True


