# src/pricing_models/__init__.py
"""
Option pricing models for OptionsLab.

This module provides various option pricing implementations including
Black-Scholes analytical formula, binomial tree, Monte Carlo methods,
implied volatility solving, exotic options, and advanced stochastic
volatility models.

Available Models:
    - black_scholes: Analytical Black-Scholes pricing
    - BinomialTree: Binomial tree pricing for European/American options
    - MonteCarloPricer: Basic Monte Carlo pricer with Greeks, QMC, Control Variates
    - MonteCarloMLSurrogate: ML surrogate for fast pricing
    - MonteCarloPricerUni: Unified CPU/GPU Monte Carlo pricer
    - HestonPricer: Heston stochastic volatility model
    - SABRModel: SABR volatility smile model
    - MertonJumpDiffusion: Jump-diffusion model
    - CrankNicolsonSolver: Finite difference PDE solver
    - AsianOption, BarrierOption, AmericanOption, LookbackOption: Exotic options
"""

from src.pricing_models.binomial_tree import BinomialTree
from src.pricing_models.black_scholes import black_scholes

# Exotic Options
from src.pricing_models.exotic_options import (
    AmericanOption,
    AsianOption,
    AutocallableOption,
    BarrierOption,
    CliquetOption,
    LookbackOption,
    price_american,
    price_asian,
    price_barrier,
)

# Finite Difference Methods
from src.pricing_models.fdm_solver import CrankNicolsonSolver, ExplicitFDMSolver

# Stochastic Volatility Models
from src.pricing_models.heston import HestonPricer, calibrate_heston

# Implied Volatility
from src.pricing_models.iv_solver import (
    implied_volatility,
    implied_volatility_vectorized,
    iv_surface_from_prices,
)
from src.pricing_models.jump_diffusion import KouJumpDiffusion, MertonJumpDiffusion
from src.pricing_models.monte_carlo import NUMBA_AVAILABLE, MonteCarloPricer
from src.pricing_models.monte_carlo_ml import MonteCarloML  # Backward compat alias
from src.pricing_models.monte_carlo_ml import LIGHTGBM_AVAILABLE, MonteCarloMLSurrogate
from src.pricing_models.monte_carlo_unified import (
    GPU_AVAILABLE,
    MLSurrogate,
    MonteCarloPricerUni,
)
from src.pricing_models.sabr import SABRModel, calibrate_sabr

# Validation
from src.pricing_models.validation import (
    validate_arbitrage_bounds,
    validate_greeks_consistency,
    validate_put_call_parity,
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
    # Stochastic Volatility
    "HestonPricer",
    "calibrate_heston",
    "SABRModel",
    "calibrate_sabr",
    # Jump Models
    "MertonJumpDiffusion",
    "KouJumpDiffusion",
    # FDM Solvers
    "CrankNicolsonSolver",
    "ExplicitFDMSolver",
    # Implied Volatility
    "implied_volatility",
    "implied_volatility_vectorized",
    "iv_surface_from_prices",
    # Validation
    "validate_put_call_parity",
    "validate_arbitrage_bounds",
    "validate_greeks_consistency",
    # Exotic Options
    "AsianOption",
    "BarrierOption",
    "AmericanOption",
    "LookbackOption",
    "AutocallableOption",
    "CliquetOption",
    "price_asian",
    "price_barrier",
    "price_american",
    # Feature flags
    "NUMBA_AVAILABLE",
    "LIGHTGBM_AVAILABLE",
    "GPU_AVAILABLE",
]
