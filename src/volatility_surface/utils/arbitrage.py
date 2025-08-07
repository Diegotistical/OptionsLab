# src/volatility_surface/utils/arbitrage.py

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, Tuple

###########################################
# 1. Arbitrage Checks
###########################################

def check_arbitrage_violations(
    vol_surface: np.ndarray,
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray
) -> Dict[str, bool]:
    """
    Check for arbitrage violations in a volatility surface:
      - Calendar spread (volatility must increase with maturity)
      - Butterfly spread (convexity in strike)

    Inputs:
        vol_surface: shape (S, K, T)
        S, K, T: 3D meshgrids corresponding to vol_surface

    Returns:
        Dictionary with violation flags
    """
    # Calendar: σ(t2) >= σ(t1)
    calendar_violation = np.any(np.diff(vol_surface, axis=2) < -1e-4)

    # Butterfly: convexity in K dimension
    butterfly_violation = False
    for ti in range(T.shape[2]):
        for kj in range(1, K.shape[1] - 1):
            left = vol_surface[:, kj-1, ti]
            center = vol_surface[:, kj, ti]
            right = vol_surface[:, kj+1, ti]
            convexity = 2 * center - left - right
            if np.any(convexity < -1e-4):
                butterfly_violation = True
                break
        if butterfly_violation:
            break

    return {
        "calendar_spread_violation": calendar_violation,
        "butterfly_spread_violation": butterfly_violation
    }

###########################################
# 2. Black-Scholes Utilities
###########################################

def black_scholes_call_price(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
    """
    Compute Black-Scholes call price and delta
    """
    if T <= 0 or sigma <= 0:
        return max(S - K, 0), 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    return price, delta

###########################################
# 3. Delta Hedging Simulation
###########################################

def simulate_delta_hedge(
    df: pd.DataFrame,
    model,
    initial_capital: float = 1e6,
    hedge_freq: int = 1,
    tx_cost: float = 0.0005
) -> Dict[str, object]:
    """
    Simulate delta hedging strategy using model-predicted implied vols.

    DataFrame df must have:
        ['timestamp','underlying_price','strike_price','time_to_maturity','risk_free_rate']
    Model must implement:
        predict_volatility(df) -> Tuple[np.ndarray, Optional[np.ndarray]]
    """
    df = df.sort_values('timestamp').reset_index(drop=True)
    cash = initial_capital
    shares = 0.0
    prev_delta = 0.0
    pnl_list = []

    for i in range(0, len(df), hedge_freq):
        row = df.iloc[i]
        spot = row['underlying_price']
        strike = row['strike_price']
        T = row['time_to_maturity']
        r = row['risk_free_rate']

        pred_iv = model.predict_volatility(pd.DataFrame([row]))[0][0]
        option_val, delta = black_scholes_call_price(spot, strike, T, r, pred_iv)

        # Initial hedge
        if i == 0:
            shares = delta
            cash -= shares * spot
        else:
            delta_diff = delta - prev_delta
            cash -= delta_diff * spot + abs(delta_diff * spot) * tx_cost
            shares += delta_diff

        total_value = cash + shares * spot + option_val
        pnl = total_value - initial_capital
        pnl_list.append((row['timestamp'], pnl))
        prev_delta = delta

    pnl_df = pd.DataFrame(pnl_list, columns=['timestamp', 'pnl'])
    sharpe = pnl_df['pnl'].mean() / (pnl_df['pnl'].std() + 1e-6) * np.sqrt(252)
    max_dd = (pnl_df['pnl'].cummax() - pnl_df['pnl']).max()

    return {
        'pnl_df': pnl_df,
        'final_pnl': pnl_df['pnl'].iloc[-1],
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd
    }
