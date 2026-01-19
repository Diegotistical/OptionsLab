# src/simulation/gbm_numpy.py
"""
Ultra-fast NumPy-vectorized GBM simulation.

Optimizations:
    - Antithetic variates (2x variance reduction, no extra cost)
    - Sum instead of cumsum for terminal-only
    - In-place operations where possible
    - Pre-computed constants
"""

import numpy as np


def simulate_gbm_numpy(
    S: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    n_paths: int,
    n_steps: int,
    seed: int,
    antithetic: bool = True,
) -> np.ndarray:
    """
    Fast terminal GBM prices with antithetic variates.

    Returns:
        Array of terminal prices. If antithetic=True, returns 2*n_paths.
    """
    rng = np.random.default_rng(seed)

    # Pre-compute constants
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma * sigma) * dt
    vol = sigma * np.sqrt(dt)
    total_drift = drift * n_steps
    log_S0 = np.log(S)

    # For terminal-only, we just need sum of increments, not cumsum
    # This is O(n_paths * n_steps) but with better cache usage
    Z = rng.standard_normal((n_paths, n_steps))

    # Sum across steps (axis=1) - much faster than cumsum for terminal
    log_S_T = log_S0 + total_drift + vol * np.sum(Z, axis=1)

    if antithetic:
        # Antithetic paths: use -Z for free variance reduction
        log_S_T_anti = log_S0 + total_drift - vol * np.sum(Z, axis=1)
        return np.concatenate([np.exp(log_S_T), np.exp(log_S_T_anti)])

    return np.exp(log_S_T)


def simulate_gbm_numpy_fast(
    S: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    """
    Fastest possible terminal price simulation (single-step).

    For n_steps=1, this is analytically exact and maximally fast.
    Uses the closed-form: S_T = S * exp((r-q-0.5σ²)T + σ√T * Z)
    """
    rng = np.random.default_rng(seed)

    drift = (r - q - 0.5 * sigma * sigma) * T
    vol = sigma * np.sqrt(T)
    log_S0 = np.log(S)

    Z = rng.standard_normal(n_paths)

    # Antithetic: same randoms, doubled output
    log_S_T_pos = log_S0 + drift + vol * Z
    log_S_T_neg = log_S0 + drift - vol * Z

    return np.concatenate([np.exp(log_S_T_pos), np.exp(log_S_T_neg)])


def simulate_gbm_paths(
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
    Simulate full GBM paths (when path-dependent).

    Returns:
        Array of shape (n_paths, n_steps + 1).
    """
    rng = np.random.default_rng(seed)

    dt = T / n_steps
    drift = (r - q - 0.5 * sigma * sigma) * dt
    vol = sigma * np.sqrt(dt)

    Z = rng.standard_normal((n_paths, n_steps))
    log_returns = drift + vol * Z

    # Cumsum needed for full paths
    log_S = np.log(S) + np.cumsum(log_returns, axis=1)

    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S
    paths[:, 1:] = np.exp(log_S)

    return paths
