# src/pricing_models/__init__.py
"""
Option pricing models for OptionsLab.

This module provides various option pricing implementations including
Black-Scholes analytical formula, binomial tree, Monte Carlo methods,
implied volatility solving, and exotic options.

Available Models:
    - black_scholes: Analytical Black-Scholes pricing
    - BinomialTree: Binomial tree pricing for European/American options
    - MonteCarloPricer: Basic Monte Carlo pricer with Greeks
    - MonteCarloMLSurrogate: ML surrogate for fast pricing
    - MonteCarloPricerUni: Unified CPU/GPU Monte Carlo pricer
    - MLSurrogate: Simplified ML surrogate
    - implied_volatility: Newton-Raphson/Brent IV solver
    - AsianOption, BarrierOption, AmericanOption: Exotic options
"""

from src.pricing_models.binomial_tree import BinomialTree
from src.pricing_models.black_scholes import black_scholes

# Exotic Options
from src.pricing_models.exotic_options import (
    AmericanOption,
    AsianOption,
    BarrierOption,
    price_american,
    price_asian,
    price_barrier,
)

# Implied Volatility
from src.pricing_models.iv_solver import (
    implied_volatility,
    implied_volatility_vectorized,
    iv_surface_from_prices,
)
from src.pricing_models.monte_carlo import NUMBA_AVAILABLE, MonteCarloPricer
from src.pricing_models.monte_carlo_ml import MonteCarloML  # Backward compat alias
from src.pricing_models.monte_carlo_ml import LIGHTGBM_AVAILABLE, MonteCarloMLSurrogate
from src.pricing_models.monte_carlo_unified import (
    GPU_AVAILABLE,
    MLSurrogate,
    MonteCarloPricerUni,
)

__all__ = [
    # Analytical models
    "black_scholes",
    "BinomialTree",
    # Monte Carlo models
    "MonteCarloPricer",
    "MonteCarloMLSurrogate",
    "MonteCarloML",  # Backward compat
    "MonteCarloPricerUni",
    "MLSurrogate",
    # Implied Volatility
    "implied_volatility",
    "implied_volatility_vectorized",
    "iv_surface_from_prices",
    # Exotic Options
    "AsianOption",
    "BarrierOption",
    "AmericanOption",
    "price_asian",
    "price_barrier",
    "price_american",
    # Feature flags
    "NUMBA_AVAILABLE",
    "LIGHTGBM_AVAILABLE",
    "GPU_AVAILABLE",
]
