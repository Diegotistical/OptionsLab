class PricingError(Exception):
    """Base class for pricing-related errors."""

    pass


class InvalidOptionTypeError(PricingError):
    """Raised when the provided option type is invalid (must be 'call' or 'put')."""

    def __init__(self, option_type):
        super().__init__(
            f"Invalid option type '{option_type}'. Expected 'call' or 'put'."
        )


class NegativeVolatilityError(PricingError):
    """Raised when volatility is negative."""

    def __init__(self, sigma):
        super().__init__(f"Volatility must be non-negative, got {sigma}.")
