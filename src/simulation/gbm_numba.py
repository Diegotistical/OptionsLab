# src/simulation/gbm_numba.py
"""
Numba-JIT accelerated GBM simulation.

Optimizations:
    - cache=True for persistent compilation
    - fastmath=True for SIMD vectorization
    - nogil=True for parallel execution
    - Antithetic variates built-in
    - Pre-computed constants outside inner loop
"""

import numpy as np

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator if not args else decorator(args[0])

    def prange(*args):
        return range(*args)


@njit(cache=True, fastmath=True, nogil=True)
def _gbm_terminal_single(S, drift_total, vol_total, n_paths, seed):
    """Ultra-fast single-step terminal prices with antithetic."""
    np.random.seed(seed)
    log_S0 = np.log(S)

    result = np.empty(n_paths * 2)
    for i in range(n_paths):
        z = np.random.randn()
        result[i] = np.exp(log_S0 + drift_total + vol_total * z)
        result[i + n_paths] = np.exp(log_S0 + drift_total - vol_total * z)

    return result


@njit(cache=True, fastmath=True, nogil=True)
def _gbm_terminal_multi_step(S, T, r, sigma, q, n_paths, n_steps, seed):
    """Multi-step terminal prices with antithetic variates."""
    np.random.seed(seed)

    dt = T / n_steps
    drift = (r - q - 0.5 * sigma * sigma) * dt
    vol = sigma * np.sqrt(dt)
    log_S0 = np.log(S)

    result = np.empty(n_paths * 2)

    for i in range(n_paths):
        log_S_pos = log_S0
        log_S_neg = log_S0

        for _ in range(n_steps):
            z = np.random.randn()
            log_S_pos += drift + vol * z
            log_S_neg += drift - vol * z

        result[i] = np.exp(log_S_pos)
        result[i + n_paths] = np.exp(log_S_neg)

    return result


@njit(parallel=True, cache=True, fastmath=True)
def _gbm_terminal_parallel(S, T, r, sigma, q, n_paths, n_steps, seed):
    """Parallel multi-step terminal prices."""
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma * sigma) * dt
    vol = sigma * np.sqrt(dt)
    log_S0 = np.log(S)

    result = np.empty(n_paths * 2)

    for i in prange(n_paths):
        np.random.seed(seed + i)
        log_S_pos = log_S0
        log_S_neg = log_S0

        for _ in range(n_steps):
            z = np.random.randn()
            log_S_pos += drift + vol * z
            log_S_neg += drift - vol * z

        result[i] = np.exp(log_S_pos)
        result[i + n_paths] = np.exp(log_S_neg)

    return result


def simulate_gbm_numba(
    S: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    n_paths: int,
    n_steps: int,
    seed: int,
    parallel: bool = True,
) -> np.ndarray:
    """
    Fast Numba GBM simulation with antithetic variates.

    Returns:
        Array of 2*n_paths terminal prices.
    """
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba not available")

    # For single-step, use the optimized version
    if n_steps == 1:
        drift_total = (r - q - 0.5 * sigma * sigma) * T
        vol_total = sigma * np.sqrt(T)
        return _gbm_terminal_single(S, drift_total, vol_total, n_paths, seed)

    if parallel:
        return _gbm_terminal_parallel(S, T, r, sigma, q, n_paths, n_steps, seed)

    return _gbm_terminal_multi_step(S, T, r, sigma, q, n_paths, n_steps, seed)
