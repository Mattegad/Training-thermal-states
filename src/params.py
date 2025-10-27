# params.py
# Paramètres physiques et numériques utilisés pour reproduire Fig.2
import numpy as np

# Physical parameters (use values similar to the paper)
U = 12e-6           # nonlinearity (eV) used in paper; here unitless scaling is fine
gamma_c = 67.0      # decay rate (ns^-1) -> choose units consistent with simulation
hbar = 1.0

# Numerical parameters
Ncut = 30           # truncation of Fock space; increase if needed
wmin = -10.0        # frequency grid min
wmax = 10.0         # frequency grid max
n_w = 801           # number of frequency points for S(omega)
tlist = None        # if needed for time evolution (we use Liouvillian exponentials)
max_iter_steady = 10000

# Bogoliubov / squeezed-env parameters default (we will vary r)
alphaD_default = 5.0   # coherent displacement
theta_default = 0.0    # squeezing angle
nbar_default = 0.0     # thermal occupation

# Simulation options
use_sparse = True


/Users/gadanimatteo/Training thermal states/src/param.py