# src / volatility_surface / utils / arbitrage_checks.py

import numpy as np

def check_arbitrage_violations(vol_surface: np.ndarray, S: np.ndarray, K: np.ndarray, T: np.ndarray) -> dict:
    """
    Checks for arbitrage violations:
    - Calendar spread (vol should increase in T)
    - Butterfly spread (convexity in strike)
    
    All inputs must be 3D with shape (S, K, T)
    """
    # Calendar spread: σ(t2) >= σ(t1)
    calendar_violation = np.any(np.diff(vol_surface, axis=2) < -1e-4)

    # Butterfly spread (strike convexity): σ(k) convex in K
    butterfly_violation = False
    for i in range(T.shape[2]):
        for j in range(1, K.shape[1] - 1):
            left = vol_surface[:, j-1, i]
            center = vol_surface[:, j, i]
            right = vol_surface[:, j+1, i]
            convexity = 2 * center - left - right
            if np.any(convexity < -1e-4):
                butterfly_violation = True
                break
        if butterfly_violation:
            break

    return {
        'calendar_spread_violation': calendar_violation,
        'butterfly_spread_violation': butterfly_violation
    }
