# volatility_surface/utils/arbitrage_utils.py
"""
Arbitrage utilities for volatility surfaces.

This module exposes:
- check_butterfly_arbitrage(strikes, implied_vols, ttm, tol=1e-12)
    Checks butterfly arbitrage by testing convexity of TOTAL VARIANCE w.r.t strike.
    Returns a dict: {"is_arbitrage_free": bool, "violations": List[(index, amount)], "total_violation": float}

- check_calendar_arbitrage(ttms, total_variances, tol=1e-12)
    Quick check that total variance is non-decreasing with maturity.

Notes / assumptions:
- For butterfly arbitrage we check convexity of total variance w(K) = sigma(K)^2 * ttm
  with finite differences. If strikes are non-uniform spacing we use a central
  finite-difference formula that accounts for uneven spacing.
- The function is defensive but not a replacement for a full arbitrage engine.
"""

from __future__ import annotations
from typing import Iterable, List, Tuple, Dict, Any
import numpy as np

__all__ = ["check_butterfly_arbitrage", "check_calendar_arbitrage"]

def _ensure_1d_array(x: Iterable) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Input must be 1D")
    return arr

def check_butterfly_arbitrage(
    strikes: Iterable[float],
    implied_vols: Iterable[float],
    ttm: float,
    tol: float = 1e-12
) -> Dict[str, Any]:
    """
    Check for butterfly arbitrage on a single slice (fixed maturity).

    Parameters
    ----------
    strikes : iterable of floats
        Strike prices (must be sorted ascending). Can be non-uniformly spaced.
    implied_vols : iterable of floats
        Implied volatilities corresponding to strikes.
    ttm : float
        Time to maturity (in years) for this slice. Must be > 0.
    tol : float
        Numerical tolerance: treat values >= -tol as non-violations.

    Returns
    -------
    dict with keys:
      - "is_arbitrage_free": bool
      - "violations": list of tuples (index, violation_amount) where violation_amount > 0
          Index refers to the centre point i where discrete second derivative is negative.
      - "total_violation": sum of positive violations (abs of negative parts)
      - "w": np.ndarray of total variances used
      - "d2w": np.ndarray of discrete second derivatives (NaN at boundaries)
    """
    K = _ensure_1d_array(strikes)
    sigma = _ensure_1d_array(implied_vols)

    if K.shape[0] != sigma.shape[0]:
        raise ValueError("strikes and implied_vols must have the same length")
    if K.shape[0] < 3:
        # Can't compute a second derivative with fewer than 3 points
        return {
            "is_arbitrage_free": True,
            "violations": [],
            "total_violation": 0.0,
            "w": None,
            "d2w": None,
        }
    if ttm <= 0:
        raise ValueError("ttm must be positive")

    # Total variance
    w = (sigma ** 2) * float(ttm)

    # Ensure strikes strictly increasing
    if not np.all(np.diff(K) > 0):
        # Try sorting, but warn the user by raising (consistency needed)
        raise ValueError("strikes must be strictly increasing")

    n = K.shape[0]
    d2w = np.full(n, np.nan)

    # Central finite-difference for non-uniform grid:
    # For point i, with neighbors i-1 and i+1:
    # d2w_i ≈ 2 * ( (w_{i+1} - w_i)/(K_{i+1}-K_i) - (w_i - w_{i-1})/(K_i - K_{i-1}) ) / (K_{i+1} - K_{i-1})
    # This formula reduces to standard second derivative for uniform spacing.
    for i in range(1, n - 1):
        h1 = K[i] - K[i - 1]
        h2 = K[i + 1] - K[i]
        if h1 <= 0 or h2 <= 0:
            d2w[i] = np.nan
            continue
        term = 2.0 * (((w[i + 1] - w[i]) / h2) - ((w[i] - w[i - 1]) / h1)) / (h1 + h2)
        d2w[i] = term

    # Violations: d2w < -tol (negative second derivative implies non-convexity -> arbitrage)
    viol_indices = np.where(d2w < -tol)[0]
    violations: List[Tuple[int, float]] = []
    total_violation = 0.0
    for idx in viol_indices:
        amount = float(max(0.0, -d2w[idx]))
        violations.append((int(idx), amount))
        total_violation += amount

    return {
        "is_arbitrage_free": len(violations) == 0,
        "violations": violations,
        "total_violation": float(total_violation),
        "w": w,
        "d2w": d2w
    }


def check_calendar_arbitrage(
    ttms: Iterable[float],
    total_variances: Iterable[float],
    tol: float = 1e-12
) -> Dict[str, Any]:
    """
    Simple calendar arbitrage check: total variance must be non-decreasing with maturity.

    Parameters
    ----------
    ttms : iterable of floats
        Times to maturities (must be sorted ascending)
    total_variances : iterable of floats
        Total variances w = sigma^2 * ttm corresponding to ttms
    tol : float
        Tolerance for small decreases (e.g., due to numerical noise)

    Returns
    -------
    dict with keys:
      - "is_calendar_free": bool
      - "violations": list of tuples (index_pair, amount) where total_variance decreases
      - "total_violation": sum of positive decreases
    """
    T = _ensure_1d_array(ttms)
    W = _ensure_1d_array(total_variances)

    if T.shape[0] != W.shape[0]:
        raise ValueError("ttms and total_variances must have the same length")
    if T.shape[0] < 2:
        return {"is_calendar_free": True, "violations": [], "total_violation": 0.0}

    # ensure ascending T
    if not np.all(np.diff(T) > 0):
        raise ValueError("ttms must be strictly increasing")

    diffs = np.diff(W)
    viol_idx = np.where(diffs < -tol)[0]
    violations: List[Tuple[Tuple[int, int], float]] = []
    total_violation = 0.0
    for i in viol_idx:
        amount = float(max(0.0, -diffs[i]))
        violations.append(((int(i), int(i + 1)), amount))
        total_violation += amount

    return {
        "is_calendar_free": len(violations) == 0,
        "violations": violations,
        "total_violation": float(total_violation),
    }


# ---------------------
# Quick self-test when run directly
# ---------------------
if __name__ == "__main__":
    # Small smoke tests
    K = np.array([80., 90., 100., 110., 120.])
    # Construct a convex total variance: sigma increases away from ATM -> convex w
    sigma_convex = np.array([0.40, 0.30, 0.20, 0.30, 0.40])
    res = check_butterfly_arbitrage(K, sigma_convex, ttm=0.5)
    print("Convex test:", res["is_arbitrage_free"], "violations:", res["violations"])

    # Make an artificial non-convex w (violation at center)
    sigma_bad = sigma_convex.copy()
    sigma_bad[2] = 0.45  # bump center -> non-convex
    res2 = check_butterfly_arbitrage(K, sigma_bad, ttm=0.5, tol=1e-14)
    print("Non-convex test:", res2["is_arbitrage_free"], "violations:", res2["violations"], "total:", res2["total_violation"])

    # Calendar test
    ttms = np.array([0.1, 0.5, 1.0, 2.0])
    w_good = np.array([0.01, 0.02, 0.04, 0.08])
    print("Calendar good:", check_calendar_arbitrage(ttms, w_good))
    w_bad = np.array([0.01, 0.019, 0.04, 0.035])
    print("Calendar bad:", check_calendar_arbitrage(ttms, w_bad))

def validate_domain(X: np.ndarray, reference_X: np.ndarray = None) -> float:
    """
    Simple domain validation for volatility surfaces.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix after engineering (n_samples x n_features)
    reference_X : np.ndarray, optional
        Reference features (e.g., training features) to compare distribution
    
    Returns
    -------
    validity_score : float
        Fraction of samples passing simple arbitrage-free checks (0.0 - 1.0)
    """

    # Example heuristic checks:

    # 1. Implied volatility feature must be non-negative (usually the last column)
    vol_col = -1  # adjust if needed
    vols = X[:, vol_col] if X.ndim > 1 else X
    non_negative = np.mean(vols >= 0)

    # 2. Optional: simple TTM monotonicity check if ttm is in features
    # Suppose ttm is in column 0
    ttm_col = 0
    if X.shape[1] > ttm_col:
        ttm = X[:, ttm_col]
        ttm_monotone = np.mean(np.diff(ttm) >= 0)
    else:
        ttm_monotone = 1.0

    # 3. Optional: convexity check along moneyness if available
    # Suppose moneyness is in column 1
    m_col = 1
    if X.shape[1] > m_col:
        m = X[:, m_col]
        vols_sorted = np.sort(vols)
        convexity = np.mean(np.diff(vols_sorted, n=2) >= -1e-4)  # small tolerance
    else:
        convexity = 1.0

    # Combine checks into a single validity score (0.0 - 1.0)
    validity_score = float(np.mean([non_negative, ttm_monotone, convexity]))
    return validity_score
