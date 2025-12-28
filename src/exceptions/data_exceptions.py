class DataError(Exception):
    """Base class for data-related errors."""

    pass


class MissingDataError(DataError):
    """Raised when required market data is missing."""

    def __init__(self, field: str):
        super().__init__(f"Missing required data field: {field}")


class InvalidDataFormatError(DataError):
    """Raised when input data format is invalid."""

    def __init__(self, message: str):
        super().__init__(f"Invalid data format: {message}")


class DataSourceConnectionError(DataError):
    """Raised when connection to a data source fails."""

    def __init__(self, source: str, details: str = ""):
        msg = f"Failed to connect to data source: {source}"
        if details:
            msg += f" | Details: {details}"
        super().__init__(msg)


class DataOutOfRangeError(DataError):
    """Raised when input data is outside acceptable range."""

    def __init__(self, field: str, value, valid_range: tuple):
        super().__init__(
            f"Value '{value}' for field '{field}' is out of range {valid_range}"
        )
