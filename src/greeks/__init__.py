# src/greeks/__init__.py
"""
Greeks computation module.

Provides:
    - compute_greeks: Original BinomialTree-specific Greeks (backward compat)
    - compute_greeks_unified: Universal interface for ANY pricer
    - Adapters: HestonAdapter, SABRAdapter, FDMAdapter for non-conforming models
"""

from src.greeks.greeks import ExerciseStyle, OptionType, compute_greeks
from src.greeks.unified_greeks import (
    ExoticAdapter,
    FDMAdapter,
    HestonAdapter,
    JumpDiffusionAdapter,
    PricerProtocol,
    SABRAdapter,
    compute_greeks_unified,
    greeks_fdm,
    greeks_heston,
    greeks_sabr,
)

__all__ = [
    # Original (BinomialTree)
    "compute_greeks",
    "OptionType",
    "ExerciseStyle",
    # Unified interface (any model)
    "PricerProtocol",
    "compute_greeks_unified",
    # Adapters
    "HestonAdapter",
    "SABRAdapter",
    "FDMAdapter",
    "JumpDiffusionAdapter",
    "ExoticAdapter",
    # Convenience functions
    "greeks_heston",
    "greeks_sabr",
    "greeks_fdm",
]
