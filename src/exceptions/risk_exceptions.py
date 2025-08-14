"""
Custom exceptions for risk analysis modules.
"""

class RiskError(Exception):
    """Base class for all risk analysis errors."""
    pass


class InvalidRiskMetricError(RiskError):
    """Raised when an invalid risk metric is requested."""
    pass


class RiskCalculationError(RiskError):
    """Raised when a risk calculation fails."""
    pass


class InsufficientPortfolioDataError(RiskError):
    """Raised when portfolio data is missing or incomplete."""
    pass
