# src/exceptions/__init__.py
"""
Custom exceptions for OptionsLab.

This module provides a hierarchy of exception classes for handling
errors across the library. Exceptions are organized by category:

    - Data exceptions: Data loading, validation, and format errors
    - Model exceptions: Pricing model and ML model errors
    - Risk exceptions: Risk calculation and portfolio errors
    - Monte Carlo exceptions: Simulation-specific errors
    - Greek exceptions: Greeks calculation errors
    - Pricing exceptions: General pricing errors
"""

from src.exceptions.data_exceptions import (
    DataError,
    DataOutOfRangeError,
    DataSourceConnectionError,
    InvalidDataFormatError,
    MissingDataError,
)
from src.exceptions.model_exceptions import (
    ModelConvergenceError,
    ModelError,
    ModelNotFittedError,
    UnsupportedModelTypeError,
)
from src.exceptions.montecarlo_exceptions import (
    AccelerationError,
    ConvergenceError,
    InputValidationError,
    MonteCarloError,
)
from src.exceptions.risk_exceptions import (
    InsufficientPortfolioDataError,
    InvalidRiskMetricError,
    RiskCalculationError,
    RiskError,
)

__all__ = [
    # Data exceptions
    "DataError",
    "MissingDataError",
    "InvalidDataFormatError",
    "DataSourceConnectionError",
    "DataOutOfRangeError",
    # Model exceptions
    "ModelError",
    "ModelNotFittedError",
    "ModelConvergenceError",
    "UnsupportedModelTypeError",
    # Risk exceptions
    "RiskError",
    "InvalidRiskMetricError",
    "RiskCalculationError",
    "InsufficientPortfolioDataError",
    # Monte Carlo exceptions
    "MonteCarloError",
    "InputValidationError",
    "ConvergenceError",
    "AccelerationError",
]