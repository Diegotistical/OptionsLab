# src/__init__.py
"""
OptionsLab - Advanced Option Pricing and Risk Analysis Toolkit.

This package provides comprehensive tools for financial derivatives
analysis including pricing models, Greeks computation, risk analysis,
and volatility surface modeling.

Modules:
    - pricing_models: Option pricing implementations (BS, MC, Binomial)
    - greeks: Greeks computation utilities
    - risk_analysis: VaR, ES, and stress testing
    - volatility_surface: IV surface generation and analysis
    - utils: Common utilities and helpers
    - exceptions: Custom exception classes

Example:
    >>> from src.pricing_models import MonteCarloPricer, black_scholes
    >>> from src.risk_analysis import VaRAnalyzer
    >>> # Price an option
    >>> price = black_scholes(100, 100, 1.0, 0.05, 0.2, 'call')
"""

from src import (
    exceptions,
    greeks,
    pricing_models,
    risk_analysis,
    utils,
    volatility_surface,
)

__all__ = [
    "exceptions",
    "greeks",
    "pricing_models",
    "risk_analysis",
    "utils",
    "volatility_surface",
]

__version__ = "1.0.0"