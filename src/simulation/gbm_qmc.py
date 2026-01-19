# src/simulation/gbm_qmc.py
"""
Quasi-Monte Carlo GBM using Sobol sequences.

QMC converges at O(1/N) vs O(1/sqrt(N)) for MC.
For smooth payoffs, this means ~5-10x faster convergence.
"""

import numpy as np
from scipy.stats import norm
from scipy.stats.qmc import Sobol


def simulate_gbm_qmc(
    S: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    n_paths: int,
    n_steps: int,
    seed: int,
) -> np.ndarray:
    """
    Sobol QMC terminal prices.

    Note: n_paths should be power of 2 for optimal Sobol properties.
    """
    # Use min(n_steps, 21201) to stay within Sobol's direction numbers
    effective_steps = min(n_steps, 21201)

    sampler = Sobol(d=effective_steps, scramble=True, seed=seed)
    uniforms = sampler.random(n_paths)

    # Clamp to avoid inf at boundaries
    normals = norm.ppf(np.clip(uniforms, 1e-10, 1 - 1e-10))

    dt = T / effective_steps
    drift = (r - q - 0.5 * sigma * sigma) * dt
    vol = sigma * np.sqrt(dt)
    log_S0 = np.log(S)

    # Sum is sufficient for terminal - no cumsum needed
    log_S_T = log_S0 + drift * effective_steps + vol * np.sum(normals, axis=1)

    return np.exp(log_S_T)


def simulate_gbm_qmc_antithetic(
    S: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    n_paths: int,
    n_steps: int,
    seed: int,
) -> np.ndarray:
    """QMC with antithetic variates for even faster convergence."""
    effective_steps = min(n_steps, 21201)

    sampler = Sobol(d=effective_steps, scramble=True, seed=seed)
    uniforms = sampler.random(n_paths)

    normals = norm.ppf(np.clip(uniforms, 1e-10, 1 - 1e-10))

    dt = T / effective_steps
    drift_total = (r - q - 0.5 * sigma * sigma) * T
    vol = sigma * np.sqrt(dt)
    log_S0 = np.log(S)

    sum_Z = vol * np.sum(normals, axis=1)
    log_S_T_pos = log_S0 + drift_total + sum_Z
    log_S_T_neg = log_S0 + drift_total - sum_Z

    return np.concatenate([np.exp(log_S_T_pos), np.exp(log_S_T_neg)])
