# src/simulation/__init__.py
"""
Path simulation backends for Monte Carlo pricing.

Each backend exposes a single interface:
    simulate_terminal_prices(S, T, r, sigma, q, n_paths, n_steps, seed) -> np.ndarray
"""

from src.simulation.gbm_numba import NUMBA_AVAILABLE, simulate_gbm_numba
from src.simulation.gbm_numpy import simulate_gbm_numpy, simulate_gbm_numpy_fast
from src.simulation.gbm_qmc import simulate_gbm_qmc, simulate_gbm_qmc_antithetic

__all__ = [
    "simulate_gbm_numpy",
    "simulate_gbm_numpy_fast",
    "simulate_gbm_numba",
    "simulate_gbm_qmc",
    "simulate_gbm_qmc_antithetic",
    "NUMBA_AVAILABLE",
]
