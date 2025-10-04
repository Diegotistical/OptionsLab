class ModelError(Exception):
    """Base class for model-related errors."""

    pass


class ModelNotFittedError(ModelError):
    """Raised when a model is used before being fitted."""

    def __init__(self):
        super().__init__("Model must be fitted before use.")


class ModelConvergenceError(ModelError):
    """Raised when a model fails to converge."""

    def __init__(self, details: str = ""):
        message = "Model failed to converge."
        if details:
            message += f" Details: {details}"
        super().__init__(message)


class UnsupportedModelTypeError(ModelError):
    """Raised when an unsupported model type is requested."""

    def __init__(self, model_type: str, supported_types: list):
        super().__init__(
            f"Unsupported model type '{model_type}'. Supported types: {', '.join(supported_types)}"
        )
