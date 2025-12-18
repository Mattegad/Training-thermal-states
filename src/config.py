# src/config.py
import numpy as np
from dataclasses import dataclass
from typing import Literal

# --- Conversion et constantes physiques ---
eV_to_Hz = 2.42e14

@dataclass
class PhysicalParams:
    U: float = 12e-6             # nonlinearity (eV)
    gamma_c: float = 67e9 / eV_to_Hz / 6.28   # cavity decay rate (eV equivalent)
    gamma_s: float = 10e3 / eV_to_Hz / 6.28    # OPO decay rate (eV equivalent)
    hbar: float = 1.0
    phi: float = -np.pi / 2      # homodyne phase
    G: float = 1.0                # gain factor
    eta: float = 0.4              # detection efficiency
    e: float = 1.6e-19            # electron charge (C)
    beta: float = 1e4             # LO amplitude

@dataclass
class NumericalParams:
    Ncut: int = 30                # Fock space truncation
    wmin: float = -1e-3           # frequency grid min (eV)
    wmax: float = 1e-3            # frequency grid max (eV)
    n_w: int = 8001               # number of frequency points
    tlist: np.ndarray = None      # optional, can be generated in simulation
    max_iter_steady: int = 10000
    n_train: int = 10
    n_test: int = 10
    n_traj: int = 1000            # number of stochastic trajectories
    dt_s: float = 1e-2/PhysicalParams.gamma_s     # time step for SDE integration
    dt_c: float = 1e-2/PhysicalParams.gamma_c     # time step for quantum master equation
    dt_stock: float = 0.5/PhysicalParams.gamma_c  # time step for data storage

@dataclass
class SqueezingParams:
    alphaD_default: float = 5.0   # coherent displacement
    theta_default: float = 0.0    # squeezing angle
    nbar_default: float = 0.0     # thermal occupation

@dataclass
class SimulationOptions:
    use_sparse: bool = True

@dataclass
class SpectrumConfig:
    which: Literal["S", "Si_minus", "Si_plus"]  # Type de spectre à calculer      

# --- Instances globales (à importer dans le code) ---


physical_params = PhysicalParams()
numerical_params = NumericalParams()
squeezing_params = SqueezingParams()
sim_options = SimulationOptions()
spec_config = SpectrumConfig(which="Si_plus")
