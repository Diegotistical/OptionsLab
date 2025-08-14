# src / greeks / greeks.py

"""
Greek computation module for option pricing models.

Provides functions to calculate primary Greeks (Delta, Gamma, Vega, Theta, Rho) and
experimental second-order Greeks (Vanna, Charm, Vomma) for European and American options
using a generic Binomial Tree model.

Raises:
    GreeksError: On any failure during Greek computation.
    InputValidationError: If input parameters are invalid.
"""

from typing import Callable, Optional, OrderedDict
from collections import OrderedDict as OD
from enum import IntEnum
import warnings

import numpy as np
from ..pricing_models.binomial_tree import BinomialTree
from ..exceptions.greek_exceptions import GreeksError, InputValidationError

__all__ = ['compute_greeks', 'OptionType', 'ExerciseStyle']

class OptionType(IntEnum):
    """Option type: CALL or PUT."""
    CALL = 0
    PUT = 1

class ExerciseStyle(IntEnum):
    """Option exercise style: EUROPEAN or AMERICAN."""
    EUROPEAN = 0
    AMERICAN = 1

def compute_greeks(
    model: BinomialTree,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN,
    q: float = 0.0,
    h: Optional[float] = None,
) -> OrderedDict[str, float]:
    """
    Compute option price and Greeks using the provided binomial tree model.

    Args:
        model: BinomialTree instance.
        S: Spot price of the underlying.
        K: Strike price.
        T: Time to maturity (in years).
        r: Risk-free interest rate.
        sigma: Volatility of underlying.
        option_type: CALL or PUT.
        exercise_style: EUROPEAN (default) or AMERICAN.
        q: Dividend yield.
        h: Step size for finite differences (optional).

    Returns:
        OrderedDict[str, float]: Contains price, Delta, Gamma, Vega, Theta, Rho,
        and experimental second-order Greeks (Vanna, Charm, Vomma).

    Raises:
        GreeksError: If computation fails.
        InputValidationError: If inputs are invalid.
    """
    try:
        # Step size for finite differences
        h = h or max(1e-4, 0.01 * S)

        # Price evaluation cache
        def price_cache() -> Callable[[float, float, float, float, float, float], float]:
            cache = {}
            def _price(S_: float, K_: float, T_: float, r_: float, sigma_: float, q_: float) -> float:
                key = (S_, K_, T_, r_, sigma_, q_)
                if key not in cache:
                    cache[key] = model.price(S_, K_, T_, r_, sigma_, option_type.name.lower(), exercise_style.name.lower(), q_)
                return cache[key]
            return _price

        P = price_cache()
        base_price = P(S, K, T, r, sigma, q)

        greeks = OD()
        greeks['price'] = base_price
        greeks['delta'] = _delta(model, S, K, T, r, sigma, option_type, exercise_style, q, h, P)
        greeks['gamma'] = _gamma(model, S, K, T, r, sigma, option_type, exercise_style, q, h, P)
        greeks['vega'] = _vega(model, S, K, T, r, sigma, option_type, exercise_style, q, h, P)
        greeks['theta'] = _theta(model, S, K, T, r, sigma, option_type, exercise_style, q, P)
        greeks['rho'] = _rho(model, S, K, T, r, sigma, option_type, exercise_style, q, h, P)

        # Experimental / second-order Greeks
        greeks['vanna'] = _vanna(model, S, K, T, r, sigma, option_type, exercise_style, q, h, P)
        greeks['charm'] = _charm(model, S, K, T, r, sigma, option_type, exercise_style, q, h, P)
        greeks['vomma'] = _vomma(model, S, K, T, r, sigma, option_type, exercise_style, q, h, P)

        return greeks

    except Exception as e:
        raise GreeksError(f"Failed to compute Greeks: {str(e)}") from e

#  Helper Functions 

def _delta(model, S, K, T, r, sigma, option_type, exercise_style, q, h, P) -> float:
    """Compute Delta (first derivative w.r.t. spot price)."""
    if exercise_style == ExerciseStyle.AMERICAN:
        u, d, p, dt = model._compute_params(S, K, T, r, sigma, q)
        asset_prices = model._compute_asset_prices(S, u, d, model.num_steps)
        option_values = model._backward_induction(asset_prices, K, r, dt, p, option_type.value, exercise_style.value)
        return (option_values[1, 1] - option_values[1, 0]) / (S * (u - d))
    return (P(S + h, K, T, r, sigma, q) - P(S - h, K, T, r, sigma, q)) / (2 * h)

def _gamma(model, S, K, T, r, sigma, option_type, exercise_style, q, h, P) -> float:
    """Compute Gamma (second derivative w.r.t. spot price)."""
    if exercise_style == ExerciseStyle.AMERICAN:
        if model.num_steps < 3:
            raise InputValidationError("Gamma for American options requires num_steps >= 3")
        u, d, p, dt = model._compute_params(S, K, T, r, sigma, q)
        asset_prices = model._compute_asset_prices(S, u, d, model.num_steps)
        option_values = model._backward_induction(asset_prices, K, r, dt, p, option_type.value, exercise_style.value)
        delta_up = (option_values[2, 2] - option_values[2, 1]) / (S * u * (u - d))
        delta_down = (option_values[2, 1] - option_values[2, 0]) / (S * d * (u - d))
        return (delta_up - delta_down) / (S * (u - d))
    return (P(S + h, K, T, r, sigma, q) - 2 * P(S, K, T, r, sigma, q) + P(S - h, K, T, r, sigma, q)) / (h ** 2)

def _vega(model, S, K, T, r, sigma, option_type, exercise_style, q, h, P) -> float:
    """Compute Vega (first derivative w.r.t. volatility)."""
    return (P(S, K, T, r, sigma + h, q) - P(S, K, T, r, sigma - h, q)) / (2 * h)

def _theta(model, S, K, T, r, sigma, option_type, exercise_style, q, P) -> float:
    """Compute Theta (first derivative w.r.t. time)."""
    dt = 1 / 365
    if T > dt:
        return (P(S, K, T - dt, r, sigma, q) - P(S, K, T, r, sigma, q)) / dt
    return -P(S, K, T, r, sigma, q) / dt

def _rho(model, S, K, T, r, sigma, option_type, exercise_style, q, h, P) -> float:
    """Compute Rho (first derivative w.r.t. interest rate)."""
    return (P(S, K, T, r + h, sigma, q) - P(S, K, T, r, sigma, q)) / h

def _vanna(model, S, K, T, r, sigma, option_type, exercise_style, q, h, P) -> float:
    """Compute Vanna (cross derivative w.r.t. spot and volatility)."""
    up_up = P(S + h, K, T, r, sigma + h, q)
    up_down = P(S + h, K, T, r, sigma - h, q)
    down_up = P(S - h, K, T, r, sigma + h, q)
    down_down = P(S - h, K, T, r, sigma - h, q)
    return (up_up - up_down - down_up + down_down) / (4 * h ** 2)

def _charm(model, S, K, T, r, sigma, option_type, exercise_style, q, h, P) -> float:
    """Compute Charm (time decay of Delta)."""
    dt = 1 / 365
    delta_now = _delta(model, S, K, T, r, sigma, option_type, exercise_style, q, h, P)
    delta_future = _delta(model, S, K, max(T - dt, 1e-8), r, sigma, option_type, exercise_style, q, h, P)
    return (delta_future - delta_now) / dt

def _vomma(model, S, K, T, r, sigma, option_type, exercise_style, q, h, P) -> float:
    """Compute Vomma (second derivative w.r.t. volatility)."""
    return (P(S, K, T, r, sigma + h, q) - 2 * P(S, K, T, r, sigma, q) + P(S, K, T, r, sigma - h, q)) / (h ** 2)
