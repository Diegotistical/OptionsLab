# src/utils/utils.py

import numpy as np
import logging
from typing import Union, Tuple
from enum import Enum  # Change to StrEnum if Python 3.11+

logger = logging.getLogger(__name__)

# ===== Constants =====
MIN_VOL = 1e-4
MIN_DENOM = 1e-10

# ===== Custom Exceptions =====
class FinancialError(Exception):
    """Base exception for pricing model errors"""
    pass

class NumericalStabilityError(FinancialError):
    """Numerical instability in calculations"""
    pass

# ===== Option Type Enum =====
class OptionType(Enum):  # Use StrEnum if Python 3.11+
    CALL = "call"
    PUT = "put"

    @staticmethod
    def from_string(s: str) -> "OptionType":
        try:
            return OptionType(s.lower())
        except ValueError:
            raise ValueError(f"Invalid option type: '{s}'. Use 'call' or 'put'.")

# ===== Input Validation =====
def validate_inputs(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
    q: float = 0.0
) -> None:
    """Comprehensive financial input validation"""
    if S <= 0:
        raise ValueError(f"Spot price must be positive. Got S={S}")
    if K <= 0:
        raise ValueError(f"Strike price must be positive. Got K={K}")
    if T < 0:
        raise ValueError(f"Time to maturity must be non-negative. Got T={T}")
    if sigma < MIN_VOL:
        raise NumericalStabilityError(f"Volatility too low: {sigma}. Minimum is {MIN_VOL}")
    if not isinstance(option_type, OptionType):
        raise TypeError(f"Invalid option type: {type(option_type)}. Use OptionType enum.")
    if q < 0 or q >= 1:
        raise ValueError(f"Dividend yield must be in [0, 1). Got q={q}")

# ===== Safe Division =====
def safe_division(numerator: float, denominator: float) -> float:
    """Numerically stable division with logging"""
    if abs(denominator) < MIN_DENOM:
        logger.warning("Division by near-zero value: %f", denominator)
        return np.sign(denominator) * np.finfo(float).max
    return numerator / denominator

# ===== d1 and d2 Calculation =====
def calculate_d1_d2(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0
) -> Tuple[float, float]:
    """Numerically stable calculation of d1 and d2"""
    if T == 0:
        raise NumericalStabilityError("Cannot calculate d1/d2 for T=0")

    sqrt_T = np.sqrt(T)
    ratio = safe_division(S, K)

    try:
        log_ratio = np.log(ratio)
    except FloatingPointError:
        logger.error("Log of near-zero ratio: S=%.2f, K=%.2f", S, K)
        raise NumericalStabilityError("Log of extremely small price ratio") from None

    d1_numerator = log_ratio + (r - q + 0.5 * sigma**2) * T
    d1 = safe_division(d1_numerator, sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    return d1, d2

# ===== Moneyness =====
def compute_moneyness(S: float, K: float) -> float:
    """Safe moneyness calculation with sanity checks"""
    if K == 0:
        raise ValueError("Strike price cannot be zero for moneyness calculation.")
    return S / K

# ===== Edge Case Handler =====
def handle_edge_cases(
    S: float,
    K: float,
    T: float,
    option_type: OptionType
) -> Union[float, None]:
    """
    Handle boundary cases before main calculations.
    Assumes validation already handled T < 0.
    """
    if T == 0:
        return max(S - K, 0) if option_type == OptionType.CALL else max(K - S, 0)
    if S == 0:
        return max(K, 0) if option_type == OptionType.PUT else 0.0
    if K == 0:
        return S if option_type == OptionType.CALL else 0.0
    return None

# ===== Export List =====
__all__ = [
    "FinancialError",
    "NumericalStabilityError",
    "OptionType",
    "validate_inputs",
    "safe_division",
    "calculate_d1_d2",
    "compute_moneyness",
    "handle_edge_cases"
]
# This module provides utility functions and classes for financial calculations,including input validation, numerical stability checks, and option type handling.