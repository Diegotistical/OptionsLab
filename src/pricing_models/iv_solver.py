# src/pricing_models/iv_solver.py
"""
Implied Volatility Solver Module.

Provides robust methods for computing implied volatility from option prices:
- Newton-Raphson with analytical vega
- Brent's method as fallback
- Vectorized operations for batch processing

Usage:
    from src.pricing_models import implied_volatility

    iv = implied_volatility(
        market_price=10.45,
        S=100, K=100, T=1.0, r=0.05, q=0.0,
        option_type="call"
    )
"""

from typing import Literal, Union

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
    q: float = 0.0,
) -> float:
    """Black-Scholes option price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if option_type == "call" else max(K - S, 0)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def black_scholes_vega(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> float:
    """Black-Scholes vega (sensitivity to volatility)."""
    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: Literal["call", "put"] = "call",
    q: float = 0.0,
    initial_guess: float = 0.2,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
    bounds: tuple = (0.001, 5.0),
) -> float:
    """
    Compute implied volatility using Newton-Raphson with Brent fallback.

    Args:
        market_price: Observed market price of the option.
        S: Spot price.
        K: Strike price.
        T: Time to maturity (years).
        r: Risk-free rate.
        option_type: "call" or "put".
        q: Dividend yield.
        initial_guess: Starting volatility for Newton-Raphson.
        max_iterations: Maximum Newton-Raphson iterations.
        tolerance: Convergence tolerance.
        bounds: (min_vol, max_vol) for Brent's method.

    Returns:
        Implied volatility as a decimal (e.g., 0.20 for 20%).

    Raises:
        ValueError: If no valid IV can be found.
    """
    # Validate inputs
    if market_price <= 0:
        raise ValueError("Market price must be positive")
    if S <= 0 or K <= 0 or T <= 0:
        raise ValueError("S, K, T must be positive")

    # Check for arbitrage bounds
    intrinsic = (
        max(S * np.exp(-q * T) - K * np.exp(-r * T), 0)
        if option_type == "call"
        else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
    )
    if market_price < intrinsic - tolerance:
        raise ValueError(
            f"Price {market_price:.4f} below intrinsic value {intrinsic:.4f}"
        )

    # Try Newton-Raphson first (faster when it works)
    sigma = initial_guess
    for i in range(max_iterations):
        price = black_scholes_price(S, K, T, r, sigma, option_type, q)
        vega = black_scholes_vega(S, K, T, r, sigma, q)

        diff = price - market_price

        if abs(diff) < tolerance:
            return sigma

        if vega < 1e-10:
            break  # Vega too small, switch to Brent

        sigma = sigma - diff / vega

        # Keep sigma in bounds
        sigma = max(bounds[0], min(bounds[1], sigma))

    # Fall back to Brent's method (more robust)
    def objective(sigma):
        return black_scholes_price(S, K, T, r, sigma, option_type, q) - market_price

    try:
        # Check if bounds bracket the root
        f_low = objective(bounds[0])
        f_high = objective(bounds[1])

        if f_low * f_high > 0:
            # Root not bracketed, try to find valid bounds
            if f_low > 0:
                raise ValueError(
                    f"Price too low for IV computation (below {bounds[0]*100:.1f}% vol)"
                )
            else:
                raise ValueError(
                    f"Price too high for IV computation (above {bounds[1]*100:.1f}% vol)"
                )

        return brentq(objective, bounds[0], bounds[1], xtol=tolerance)

    except ValueError as e:
        raise ValueError(f"Could not find implied volatility: {e}")


def implied_volatility_vectorized(
    market_prices: np.ndarray,
    S: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    r: Union[float, np.ndarray],
    option_type: Union[str, np.ndarray] = "call",
    q: Union[float, np.ndarray] = 0.0,
    initial_guess: float = 0.2,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """
    Vectorized implied volatility computation.

    Args:
        market_prices: Array of market prices.
        S, K, T, r, q: Can be scalars or arrays matching market_prices shape.
        option_type: "call", "put", or array of types.
        initial_guess: Starting volatility.
        max_iterations: Maximum iterations per point.
        tolerance: Convergence tolerance.

    Returns:
        Array of implied volatilities (NaN where computation failed).
    """
    market_prices = np.atleast_1d(market_prices)
    n = len(market_prices)

    # Broadcast scalars to arrays
    S = np.broadcast_to(S, n)
    K = np.broadcast_to(K, n)
    T = np.broadcast_to(T, n)
    r = np.broadcast_to(r, n)
    q = np.broadcast_to(q, n)

    if isinstance(option_type, str):
        option_types = [option_type] * n
    else:
        option_types = option_type

    # Initialize result
    ivs = np.full(n, np.nan)

    # Compute IV for each option
    for i in range(n):
        try:
            ivs[i] = implied_volatility(
                market_prices[i],
                S[i],
                K[i],
                T[i],
                r[i],
                option_types[i],
                q[i],
                initial_guess=initial_guess,
                max_iterations=max_iterations,
                tolerance=tolerance,
            )
        except (ValueError, RuntimeError):
            # Keep NaN for failed computations
            pass

    return ivs


def iv_surface_from_prices(
    option_data: dict,
    r: float = 0.05,
    q: float = 0.0,
) -> dict:
    """
    Build implied volatility surface from option prices.

    Args:
        option_data: Dict with keys:
            - spot: Current spot price
            - strikes: Array of strike prices
            - maturities: Array of maturities (years)
            - call_prices: 2D array [n_strikes x n_maturities]
            - put_prices: 2D array [n_strikes x n_maturities] (optional)
        r: Risk-free rate.
        q: Dividend yield.

    Returns:
        Dict with:
            - strikes: Strike prices
            - maturities: Maturities
            - call_iv: 2D IV surface for calls
            - put_iv: 2D IV surface for puts (if provided)
            - moneyness: K/S for each strike
    """
    S = option_data["spot"]
    strikes = np.array(option_data["strikes"])
    maturities = np.array(option_data["maturities"])
    call_prices = np.array(option_data["call_prices"])

    n_strikes = len(strikes)
    n_maturities = len(maturities)

    call_iv = np.full((n_strikes, n_maturities), np.nan)
    put_iv = None

    # Compute call IVs
    for i, K in enumerate(strikes):
        for j, T in enumerate(maturities):
            if call_prices[i, j] > 0:
                try:
                    call_iv[i, j] = implied_volatility(
                        call_prices[i, j], S, K, T, r, "call", q
                    )
                except ValueError:
                    pass

    # Compute put IVs if available
    if "put_prices" in option_data:
        put_prices = np.array(option_data["put_prices"])
        put_iv = np.full((n_strikes, n_maturities), np.nan)

        for i, K in enumerate(strikes):
            for j, T in enumerate(maturities):
                if put_prices[i, j] > 0:
                    try:
                        put_iv[i, j] = implied_volatility(
                            put_prices[i, j], S, K, T, r, "put", q
                        )
                    except ValueError:
                        pass

    return {
        "strikes": strikes,
        "maturities": maturities,
        "call_iv": call_iv,
        "put_iv": put_iv,
        "moneyness": strikes / S,
        "spot": S,
    }
