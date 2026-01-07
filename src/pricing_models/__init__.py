# src/pricing_models/__init__.py
"""
Option pricing models for OptionsLab.

This module provides various option pricing implementations including
Black-Scholes analytical formula, binomial tree, and Monte Carlo methods.

Available Models:
    - black_scholes: Analytical Black-Scholes pricing
    - BinomialTree: Binomial tree pricing for European/American options
    - MonteCarloPricer: Basic Monte Carlo pricer with Greeks
    - MonteCarloMLSurrogate: ML surrogate for fast pricing
    - MonteCarloPricerUni: Unified CPU/GPU Monte Carlo pricer
    - MLSurrogate: Simplified ML surrogate
"""

from src.pricing_models.binomial_tree import BinomialTree
from src.pricing_models.black_scholes import black_scholes
from src.pricing_models.monte_carlo import MonteCarloPricer, NUMBA_AVAILABLE
from src.pricing_models.monte_carlo_ml import (
    MonteCarloMLSurrogate,
    MonteCarloML,  # Backward compat alias
    LIGHTGBM_AVAILABLE,
)
from src.pricing_models.monte_carlo_unified import (
    MonteCarloPricerUni,
    MLSurrogate,
    GPU_AVAILABLE,
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
    # Feature flags
    "NUMBA_AVAILABLE",
    "LIGHTGBM_AVAILABLE",
    "GPU_AVAILABLE",
]