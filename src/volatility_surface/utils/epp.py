# src/volatility_surface/utils/epp.py
"""
Effective Pricing Penalty (EPP) — Economic Arbitrage Violation Metric.

EPP measures the maximum exploitable arbitrage profit from a volatility surface
by computing the worst-case violation across butterfly and calendar spreads.

Unlike pointwise metrics (RMSE, MAE), EPP captures economically meaningful
arbitrage: if EPP > 0, a trader can construct a portfolio that locks in
risk-free profit from the model's implied prices.

EPP(surface) = max(
    max over (K1,K2,K3,T) of butterfly_profit(K1,K2,K3,T),
    max over (K,T1,T2) of calendar_profit(K,T1,T2)
)

A surface is arbitrage-free iff EPP = 0.

Reference:
    Section 5 of the CINN volatility paper.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import norm


def _bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """Black-Scholes price (vectorizable via numpy)."""
    if T <= 1e-10 or sigma <= 1e-10:
        if option_type == "call":
            return max(S * np.exp(-r * T) - K * np.exp(-r * T), 0.0)
        else:
            return max(K * np.exp(-r * T) - S * np.exp(-r * T), 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _bs_price_vec(
    S: float,
    K: np.ndarray,
    T: float,
    r: float,
    sigma: np.ndarray,
) -> np.ndarray:
    """Vectorized BS call price for arrays of K and sigma."""
    safe_sigma = np.maximum(sigma, 1e-8)
    safe_T = max(T, 1e-10)

    d1 = (np.log(S / K) + (r + 0.5 * safe_sigma**2) * safe_T) / (
        safe_sigma * np.sqrt(safe_T)
    )
    d2 = d1 - safe_sigma * np.sqrt(safe_T)

    return S * norm.cdf(d1) - K * np.exp(-r * safe_T) * norm.cdf(d2)


def compute_butterfly_epp(
    strikes: np.ndarray,
    implied_vols: np.ndarray,
    T: float,
    S: float = 100.0,
    r: float = 0.05,
) -> float:
    """
    Compute maximum butterfly spread arbitrage profit.

    For every triple (K1 < K2 < K3) where K2 = (K1+K3)/2,
    the butterfly payoff should be non-negative in a no-arb world:
        C(K1) - 2*C(K2) + C(K3) >= 0

    Returns the maximum violation (negative = arb profit).
    """
    if len(strikes) < 3:
        return 0.0

    sorted_idx = np.argsort(strikes)
    K = strikes[sorted_idx]
    vols = implied_vols[sorted_idx]

    # Compute all call prices
    prices = _bs_price_vec(S, K, T, r, vols)

    max_violation = 0.0

    # Check all triples (i, j, k) where K_j is interpolated
    for i in range(len(K)):
        for k in range(i + 2, len(K)):
            # Find the midpoint strike
            K_mid = (K[i] + K[k]) / 2.0

            # Find closest actual strike to midpoint
            j = np.argmin(np.abs(K - K_mid))
            if j <= i or j >= k:
                continue

            # Butterfly: long wings, short body
            # Weight by distance for non-equidistant strikes
            w = (K[k] - K[j]) / (K[k] - K[i])
            butterfly_cost = w * prices[i] + (1 - w) * prices[k] - prices[j]

            # Negative cost = arbitrage profit
            if butterfly_cost < -max_violation:
                max_violation = -butterfly_cost

    return max_violation


def compute_calendar_epp(
    strike: float,
    maturities: np.ndarray,
    implied_vols: np.ndarray,
    S: float = 100.0,
    r: float = 0.05,
) -> float:
    """
    Compute maximum calendar spread arbitrage profit at a given strike.

    For T1 < T2: C(K, T2) >= C(K, T1) (no calendar arb).
    Equivalently, total variance w(T) = sigma^2 * T must be non-decreasing.

    Returns maximum violation.
    """
    if len(maturities) < 2:
        return 0.0

    sorted_idx = np.argsort(maturities)
    T = maturities[sorted_idx]
    vols = implied_vols[sorted_idx]

    # Compute call prices at each maturity
    prices = np.array([_bs_price(S, strike, t, r, v) for t, v in zip(T, vols)])

    max_violation = 0.0
    for i in range(len(T) - 1):
        # Calendar: long far, short near should be non-negative
        calendar_cost = prices[i + 1] - prices[i]
        if calendar_cost < -max_violation:
            max_violation = -calendar_cost

    return max_violation


def compute_epp(
    model,
    S: float = 100.0,
    r: float = 0.05,
    strike_range: Tuple[float, float] = (-0.4, 0.4),
    n_strikes: int = 50,
    maturities: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute the full EPP metric for a trained volatility surface model.

    Args:
        model: A trained vol surface model with predict_volatility(df) method.
        S: Spot price.
        r: Risk-free rate.
        strike_range: Range of log-moneyness to test.
        n_strikes: Number of strike points per maturity.
        maturities: Maturities to test. Defaults to [0.1, 0.25, 0.5, 1.0, 2.0].

    Returns:
        Dict with keys:
            - epp_butterfly: max butterfly arbitrage profit
            - epp_calendar: max calendar arbitrage profit
            - epp_total: max(butterfly, calendar) — the EPP metric
            - butterfly_violations: count of violated triples
            - calendar_violations: count of violated pairs
    """
    import pandas as pd

    if maturities is None:
        maturities = np.array([0.1, 0.25, 0.5, 1.0, 2.0])

    log_strikes = np.linspace(strike_range[0], strike_range[1], n_strikes)
    abs_strikes = S * np.exp(log_strikes)

    # Build prediction grid
    rows = []
    for T in maturities:
        for k in log_strikes:
            rows.append({"log_moneyness": k, "T": T})
    grid_df = pd.DataFrame(rows)

    # Get model predictions
    pred_vols = model.predict_volatility(grid_df)
    grid_df["pred_vol"] = pred_vols

    # Butterfly EPP per maturity
    max_butterfly = 0.0
    butterfly_violations = 0

    for T in maturities:
        mask = grid_df["T"] == T
        k_slice = grid_df.loc[mask, "log_moneyness"].values
        vol_slice = grid_df.loc[mask, "pred_vol"].values
        K_slice = S * np.exp(k_slice)

        # Check all butterfly triples
        sorted_idx = np.argsort(K_slice)
        K_sorted = K_slice[sorted_idx]
        vol_sorted = vol_slice[sorted_idx]
        prices = _bs_price_vec(S, K_sorted, T, r, vol_sorted)

        for i in range(len(K_sorted)):
            for k_idx in range(i + 2, len(K_sorted)):
                K_mid = (K_sorted[i] + K_sorted[k_idx]) / 2.0
                j = np.argmin(np.abs(K_sorted - K_mid))
                if j <= i or j >= k_idx:
                    continue
                w = (K_sorted[k_idx] - K_sorted[j]) / (K_sorted[k_idx] - K_sorted[i])
                butterfly_cost = w * prices[i] + (1 - w) * prices[k_idx] - prices[j]
                if butterfly_cost < 0:
                    butterfly_violations += 1
                    max_butterfly = max(max_butterfly, -butterfly_cost)

    # Calendar EPP per strike
    max_calendar = 0.0
    calendar_violations = 0

    for k in log_strikes:
        mask = np.isclose(grid_df["log_moneyness"].values, k)
        T_slice = grid_df.loc[mask, "T"].values
        vol_slice = grid_df.loc[mask, "pred_vol"].values
        K_abs = S * np.exp(k)

        sorted_idx = np.argsort(T_slice)
        T_sorted = T_slice[sorted_idx]
        vol_sorted = vol_slice[sorted_idx]

        prices = np.array(
            [_bs_price(S, K_abs, t, r, v) for t, v in zip(T_sorted, vol_sorted)]
        )

        for i in range(len(T_sorted) - 1):
            calendar_cost = prices[i + 1] - prices[i]
            if calendar_cost < 0:
                calendar_violations += 1
                max_calendar = max(max_calendar, -calendar_cost)

    return {
        "epp_butterfly": max_butterfly,
        "epp_calendar": max_calendar,
        "epp_total": max(max_butterfly, max_calendar),
        "butterfly_violations": butterfly_violations,
        "calendar_violations": calendar_violations,
    }
