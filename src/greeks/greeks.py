# src/greeks/greeks.py

"""
Dynamic Greek computation for binomial tree model.

Implements:
- Delta, Gamma, Vega, Theta, Rho
- Second-order Greeks: Vanna, Charm, Vomma (placeholder)
- Caching for price evaluations
"""

from typing import Literal, Optional, Dict, Any, OrderedDict
import numpy as np
from src.pricing_models.binomial_tree import BinomialTree
from src.exceptions import GreeksError, InputValidationError
from enum import IntEnum
from collections import OrderedDict
import warnings

class OptionType(IntEnum):
    CALL = 0
    PUT = 1

class ExerciseStyle(IntEnum):
    EUROPEAN = 0
    AMERICAN = 1

__all__ = ['compute_greeks', 'delta', 'gamma', 'vega', 'theta', 'rho']

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
    Compute all Greeks using optimal methods for the given binomial tree model.

    Returns OrderedDict with ordered Greeks: price, delta, gamma, vega, theta, rho
    """

    try:
        # Precompute strings and cache prices
        opt_str = option_type.name.lower()
        ex_str = exercise_style.name.lower()
        h = h or max(1e-4, 0.01 * S)

        def price_wrapper():
            cache = {}
            def _p(S_, K_, T_, r_, sigma_, q_):
                key = (S_, K_, T_, r_, sigma_, q_)
                if key not in cache:
                    cache[key] = model.price(S_, K_, T_, r_, sigma_, opt_str, ex_str, q_)
                return cache[key]
            return _p
        P = price_wrapper()

        base_price = P(S, K, T, r, sigma, q)

        greeks = OrderedDict()
        greeks['price'] = base_price
        greeks['delta'] = _delta(model, S, K, T, r, sigma, option_type, exercise_style, q, h, P)
        greeks['gamma'] = _gamma(model, S, K, T, r, sigma, option_type, exercise_style, q, h, P)
        greeks['vega'] = _vega(model, S, K, T, r, sigma, option_type, exercise_style, q, h, P)
        greeks['theta'] = _theta(model, S, K, T, r, sigma, option_type, exercise_style, q, P)
        greeks['rho'] = _rho(model, S, K, T, r, sigma, option_type, exercise_style, q, h, P)

        # Second-order Greeks (placeholders)
        greeks['vanna'] = _vanna(model, S, K, T, r, sigma, option_type, exercise_style, q, h, P)
        greeks['charm'] = _charm(model, S, K, T, r, sigma, option_type, exercise_style, q, h, P)
        greeks['vomma'] = _vomma(model, S, K, T, r, sigma, option_type, exercise_style, q, h, P)

        return greeks

    except Exception as e:
        raise GreeksError(f"Greeks computation failed: {str(e)}") from e

# Helper functions with explicit type hints

def _delta(
    model: BinomialTree,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
    exercise_style: ExerciseStyle,
    q: float,
    h: float,
    P: callable,
) -> float:
    """
    First derivative w.r.t. spot price.
    """
    if exercise_style == ExerciseStyle.AMERICAN:
        u, d, p, dt = model._compute_params(S, K, T, r, sigma, q)
        asset_prices = model._compute_asset_prices(S, u, d, model.num_steps)
        option_values = model._backward_induction(asset_prices, K, r, dt, p, option_type.value, exercise_style.value)
        return (option_values[1, 1] - option_values[1, 0]) / (S * (u - d))
    else:
        return (P(S + h, K, T, r, sigma, q) - P(S - h, K, T, r, sigma, q)) / (2 * h)

def _gamma(
    model: BinomialTree,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
    exercise_style: ExerciseStyle,
    q: float,
    h: float,
    P: callable,
) -> float:
    """
    Second derivative w.r.t. spot price.
    """
    if exercise_style == ExerciseStyle.AMERICAN:
        if model.num_steps < 3:
            raise InputValidationError("American Gamma requires num_steps >= 3")
        u, d, p, dt = model._compute_params(S, K, T, r, sigma, q)
        asset_prices = model._compute_asset_prices(S, u, d, model.num_steps)
        option_values = model._backward_induction(asset_prices, K, r, dt, p, option_type.value, exercise_style.value)
        delta_up = (option_values[2, 2] - option_values[2, 1]) / (S * u * (u - d))
        delta_down = (option_values[2, 1] - option_values[2, 0]) / (S * d * (u - d))
        return (delta_up - delta_down) / (S * (u - d))
    else:
        return (P(S + h, K, T, r, sigma, q) - 2 * P(S, K, T, r, sigma, q) + P(S - h, K, T, r, sigma, q)) / (h * h)

def _vega(
    model: BinomialTree,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
    exercise_style: ExerciseStyle,
    q: float,
    h: float,
    P: callable,
) -> float:
    """
    First derivative w.r.t. volatility.
    """
    return (P(S, K, T, r, sigma + h, q) - P(S, K, T, r, sigma - h, q)) / (2 * h)

def _theta(
    model: BinomialTree,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
    exercise_style: ExerciseStyle,
    q: float,
    P: callable,
) -> float:
    """
    First derivative w.r.t. time (negative means value decays).
    """
    dt = 1 / 365
    if T > dt:
        return (P(S, K, T - dt, r, sigma, q) - P(S, K, T, r, sigma, q)) / dt
    else:
        # Very close to expiry — interpret as max decay
        return -P(S, K, T, r, sigma, q) / dt

def _rho(
    model: BinomialTree,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
    exercise_style: ExerciseStyle,
    q: float,
    h: float,
    P: callable,
) -> float:
    """
    First derivative w.r.t. interest rate.
    """
    return (P(S, K, T, r + h, sigma, q) - P(S, K, T, r, sigma, q)) / h

def _vanna(
    model: BinomialTree,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
    exercise_style: ExerciseStyle,
    q: float,
    h: float,
    P: callable,
) -> float:
    """
    Cross derivative w.r.t. spot and volatility.
    """
    up_up = P(S + h, K, T, r, sigma + h, q)
    up_down = P(S + h, K, T, r, sigma - h, q)
    down_up = P(S - h, K, T, r, sigma + h, q)
    down_down = P(S - h, K, T, r, sigma - h, q)
    return (up_up - up_down - down_up + down_down) / (4 * h * h)

def _charm(
    model: BinomialTree,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
    exercise_style: ExerciseStyle,
    q: float,
    h: float,
    P: callable,
) -> float:
    """
    Time decay of Delta: d(Delta)/dt
    """
    dt = 1 / 365
    delta_now = _delta(model, S, K, T, r, sigma, option_type, exercise_style, q, h, P)
    delta_future = _delta(model, S, K, T - dt, r, sigma, option_type, exercise_style, q, h, P)
    return (delta_future - delta_now) / dt

def _vomma(
    model: BinomialTree,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
    exercise_style: ExerciseStyle,
    q: float,
    h: float,
    P: callable,
) -> float:
    """
    Second derivative w.r.t. volatility (sensitivity of Vega).
    """
    return (P(S, K, T, r, sigma + h, q) - 2 * P(S, K, T, r, sigma, q) + P(S, K, T, r, sigma - h, q)) / (h * h)
