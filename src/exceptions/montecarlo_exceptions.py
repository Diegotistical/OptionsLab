class MonteCarloError(Exception):
    """Base exception for Monte Carlo pricer"""


class InputValidationError(MonteCarloError):
    """Raised for invalid inputs in Monte Carlo calculations"""
