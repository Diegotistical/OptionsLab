# src/pricing_models/black_scholes.py

from typing import Literal

import numpy as np
from scipy.stats import norm


def black_scholes(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
    q: float = 0.0,
) -> float:
    """
    Black-Scholes-Merton model for European option pricing.

    Parameters:
        S: Spot price of the underlying asset
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate (continuously compounded)
        sigma: Volatility of the underlying asset
        option_type: "call" or "put"
        q: Dividend yield (default 0)

    Returns:
        Option price (float)
    """
    if S <= 0 or K <= 0 or T < 0 or sigma < 0:
        raise ValueError(
            "Invalid input: all inputs must be positive, and T, sigma >= 0"
        )

    # Handle immediate expiration
    if T == 0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    else:
        raise ValueError("option_type must be 'call' or 'put'")
