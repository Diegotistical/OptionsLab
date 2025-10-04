# src/exceptions/greek_exceptions.py


class GreeksError(Exception):
    """Base exception class for all Greeks-related errors."""

    def __init__(self, message: str = "An error occurred in Greeks calculations."):
        super().__init__(message)


class InputValidationError(GreeksError):
    """Exception raised when input values for Greeks calculations are invalid."""

    def __init__(self, message: str = "Invalid input provided for Greeks calculation."):
        super().__init__(message)
