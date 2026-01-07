# src/exceptions/montecarlo_exceptions.py
"""
Custom exceptions for Monte Carlo pricing operations.

This module defines exception classes used throughout the Monte Carlo
pricing implementations to provide clear error messages and proper
exception handling.

Classes:
    MonteCarloError: Base exception for Monte Carlo operations.
    InputValidationError: Raised when input parameters are invalid.
    ConvergenceError: Raised when simulation fails to converge.
    AccelerationError: Raised when hardware acceleration fails.
"""

__all__ = [
    "MonteCarloError",
    "InputValidationError",
    "ConvergenceError",
    "AccelerationError",
]


class MonteCarloError(Exception):
    """
    Base exception for Monte Carlo pricer errors.

    This is the parent class for all Monte Carlo related exceptions.
    Catching this exception will catch all Monte Carlo errors.

    Example:
        >>> try:
        ...     pricer.price(-100, 100, 1.0, 0.05, 0.2, 'call')
        ... except MonteCarloError as e:
        ...     print(f"MC Error: {e}")
    """

    def __init__(self, message: str = "Monte Carlo computation error"):
        """
        Initialize MonteCarloError.

        Args:
            message: Error description.
        """
        self.message = message
        super().__init__(self.message)


class InputValidationError(MonteCarloError):
    """
    Raised when input parameters to Monte Carlo methods are invalid.

    This includes cases like:
        - Negative spot or strike prices
        - Zero or negative time to maturity
        - Negative volatility
        - Invalid option type

    Example:
        >>> from src.exceptions.montecarlo_exceptions import InputValidationError
        >>> raise InputValidationError("Spot price must be positive")
    """

    def __init__(self, message: str = "Invalid input parameters"):
        """
        Initialize InputValidationError.

        Args:
            message: Description of what input is invalid.
        """
        super().__init__(f"Input validation failed: {message}")


class ConvergenceError(MonteCarloError):
    """
    Raised when Monte Carlo simulation fails to converge.

    This may occur when:
        - Standard error is too high after maximum iterations
        - Price estimate is unstable
        - Greeks calculation produces NaN values

    Example:
        >>> raise ConvergenceError("Delta estimate unstable", iterations=100000)
    """

    def __init__(
        self,
        message: str = "Simulation did not converge",
        iterations: int = 0,
    ):
        """
        Initialize ConvergenceError.

        Args:
            message: Description of convergence failure.
            iterations: Number of iterations attempted.
        """
        self.iterations = iterations
        full_message = f"{message} (after {iterations} iterations)"
        super().__init__(full_message)


class AccelerationError(MonteCarloError):
    """
    Raised when hardware acceleration (Numba/GPU) fails.

    This includes:
        - Numba JIT compilation failures
        - GPU memory allocation errors
        - Device not available

    Example:
        >>> raise AccelerationError("GPU out of memory", backend="cuda")
    """

    def __init__(
        self,
        message: str = "Hardware acceleration failed",
        backend: str = "unknown",
    ):
        """
        Initialize AccelerationError.

        Args:
            message: Description of acceleration failure.
            backend: Backend that failed ('numba', 'cuda', etc).
        """
        self.backend = backend
        full_message = f"{message} (backend: {backend})"
        super().__init__(full_message)
