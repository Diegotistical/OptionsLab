# File: src/volatility_surface/utils/arbitrage_utils.py

"""
Utilities to perform quick arbitrage-related sanity checks on volatility inputs.
This file intentionally provides conservative placeholder implementations that
avoid import-time failures in environments like Streamlit Cloud. Replace with
stronger financial logic later.
"""

import numpy as np
from typing import Union, Sequence


def validate_domain(x: Union[np.ndarray, Sequence]) -> bool:
    """
    Validate input numeric domain. Placeholder:
    - Converts to numpy array
    - Raises ValueError if any value is NaN or negative
    - Returns True otherwise
    """
    arr = np.asarray(x)
    if np.isnan(arr).any():
        raise ValueError("NaN values found in domain")
    if (arr < 0).any():
        # allow zeros but not negatives
        raise ValueError("Negative values not allowed in domain")
    return True


def check_calendar_spread(vols_by_t: np.ndarray, tol: float = 1e-8) -> bool:
    """
    Check monotonicity in maturity dimension: vols should not decrease materially
    as maturity increases (simple heuristic). vols_by_t shape: (T, N) or (N, T).
    Returns True if passes check.
    """
    arr = np.asarray(vols_by_t)
    # try to orient as (n_points, T)
    if arr.ndim == 1:
        return True
    if arr.ndim == 2:
        # ensure axis 1 is maturity
        diffs = np.diff(arr, axis=1)
        return not np.any(diffs < -tol)
    return True


def check_strike_convexity(vol_surface: np.ndarray, tol: float = 1e-8) -> bool:
    """
    Check convexity in strike direction for a 2D vol surface (K x T).
    Returns True if convex in strike for all maturities.
    """
    arr = np.asarray(vol_surface)
    if arr.ndim != 2:
        return True
    K, T = arr.shape
    # second difference along strike axis
    second_diff = arr[2:, :] - 2 * arr[1:-1, :] + arr[:-2, :]
    return not np.any(second_diff < -tol)
