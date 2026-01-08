# src/greeks/greeks.py

"""
Greek computation module for option pricing models.

Optimized to work with the production-grade BinomialTree.
Delegates Delta/Gamma to the model's analytical engine and computes
other Greeks via efficient finite differences.
"""

from collections import OrderedDict as OD
from enum import IntEnum
from typing import Optional, OrderedDict

from src.exceptions.greek_exceptions import GreeksError
from src.pricing_models.binomial_tree import BinomialTree

__all__ = ["compute_greeks", "OptionType", "ExerciseStyle"]


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
    Compute option price and Greeks using the optimized Binomial Tree.
    """
    try:
        # Step size for finite differences (Vega/Theta/Rho)
        h = h or max(1e-4, 0.01 * S)

        # 1. Get analytical Price, Delta, Gamma in O(1) call
        # The new BinomialTree calculates these simultaneously.
        base_metrics = model.calculate_all(
            S, K, T, r, sigma, option_type.name.lower(), exercise_style.name.lower(), q
        )

        greeks = OD()
        greeks["price"] = base_metrics["price"]
        greeks["delta"] = base_metrics["delta"]
        greeks["gamma"] = base_metrics["gamma"]

        # 2. Helper for finite difference Greeks
        # We only strictly need this for Vega, Theta, Rho, etc.
        def get_price(S_, K_, T_, r_, sigma_, q_):
            return model.price(
                S_,
                K_,
                T_,
                r_,
                sigma_,
                option_type.name.lower(),
                exercise_style.name.lower(),
                q_,
            )

        # 3. Calculate other Greeks via Finite Difference
        # Since model.price() is now fast (O(N) + JIT), these are cheap.

        # Vega (dPrice/dSigma)
        p_vega_up = get_price(S, K, T, r, sigma + h, q)
        p_vega_down = get_price(S, K, T, r, sigma - h, q)
        greeks["vega"] = (p_vega_up - p_vega_down) / (2 * h)

        # Theta (dPrice/dTime) - usually negative
        dt = 1 / 365.0
        if T > dt:
            p_theta_minus = get_price(S, K, T - dt, r, sigma, q)
            greeks["theta"] = (p_theta_minus - greeks["price"]) / dt
        else:
            greeks["theta"] = 0.0

        # Rho (dPrice/dRate)
        p_rho_up = get_price(S, K, T, r + 1e-4, sigma, q)
        p_rho_down = get_price(S, K, T, r - 1e-4, sigma, q)
        greeks["rho"] = (p_rho_up - p_rho_down) / (2 * 1e-4)

        # 4. Experimental / Second-order Greeks
        # Vanna (dDelta/dSigma)
        # We can use finite difference on the analytical Deltas for better precision
        delta_sigma_up = model.delta(
            S,
            K,
            T,
            r,
            sigma + h,
            option_type.name.lower(),
            exercise_style.name.lower(),
            q,
        )
        delta_sigma_down = model.delta(
            S,
            K,
            T,
            r,
            sigma - h,
            option_type.name.lower(),
            exercise_style.name.lower(),
            q,
        )
        greeks["vanna"] = (delta_sigma_up - delta_sigma_down) / (2 * h)

        # Charm (dDelta/dTime)
        if T > dt:
            delta_time_minus = model.delta(
                S,
                K,
                T - dt,
                r,
                sigma,
                option_type.name.lower(),
                exercise_style.name.lower(),
                q,
            )
            greeks["charm"] = (delta_time_minus - greeks["delta"]) / dt
        else:
            greeks["charm"] = 0.0

        # Vomma (dVega/dSigma)
        # (Vega_up - Vega_down) / 2h
        # Re-using the prices calculated for Vega to estimate neighbors
        # For true Vomma we need P(sigma+2h)... keeping it simple:
        p_up_up = get_price(S, K, T, r, sigma + h, q)
        p_mid = greeks["price"]
        p_down_down = get_price(S, K, T, r, sigma - h, q)
        # Convexity w.r.t sigma
        greeks["vomma"] = (p_up_up - 2 * p_mid + p_down_down) / (h**2)

        return greeks

    except Exception as e:
        raise GreeksError(f"Failed to compute Greeks: {str(e)}") from e
