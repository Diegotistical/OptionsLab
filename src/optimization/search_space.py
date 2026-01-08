# src/optimization/search_space.py
"""
Search space abstractions for hyperparameter optimization.

Provides:
- Protocol for generic search spaces
- Pre-built spaces for LightGBM, MLP, XGBoost
- Rich validation with informative exceptions

Usage:
    from src.optimization.search_space import LightGBMSearchSpace

    space = LightGBMSearchSpace()
    params = space.suggest(trial)
    space.validate(params)  # Raises InvalidSearchSpaceError if invalid
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import optuna


class InvalidSearchSpaceError(Exception):
    """
    Raised when hyperparameters violate search space constraints.

    Provides detailed diagnostics for debugging.
    """

    def __init__(
        self,
        message: str,
        param_name: Optional[str] = None,
        param_value: Any = None,
        constraint: Optional[str] = None,
    ):
        self.param_name = param_name
        self.param_value = param_value
        self.constraint = constraint

        detail = message
        if param_name:
            detail = f"{param_name}={param_value}: {message}"
        if constraint:
            detail += f" (constraint: {constraint})"

        super().__init__(detail)


@runtime_checkable
class SearchSpace(Protocol):
    """Protocol for hyperparameter search spaces."""

    def suggest(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for a trial."""
        ...

    def validate(self, params: Dict[str, Any]) -> None:
        """
        Validate hyperparameters.

        Raises:
            InvalidSearchSpaceError: If params violate constraints.
        """
        ...

    def get_default_params(self) -> Dict[str, Any]:
        """Get default (baseline) parameters."""
        ...


@dataclass
class LightGBMSearchSpace:
    """
    Search space for LightGBM models.

    Attributes:
        n_estimators_range: (min, max) for number of trees.
        max_depth_range: (min, max) for tree depth.
        learning_rate_range: (min, max) for learning rate (log scale).
        num_leaves_range: (min, max) for number of leaves.
        min_child_samples_range: (min, max) for min samples in leaf.
        subsample_range: (min, max) for row sampling.
        colsample_bytree_range: (min, max) for column sampling.
        reg_alpha_range: (min, max) for L1 regularization (log scale).
        reg_lambda_range: (min, max) for L2 regularization (log scale).
    """

    n_estimators_range: Tuple[int, int] = (50, 500)
    max_depth_range: Tuple[int, int] = (3, 12)
    learning_rate_range: Tuple[float, float] = (0.01, 0.3)
    num_leaves_range: Tuple[int, int] = (15, 127)
    min_child_samples_range: Tuple[int, int] = (5, 100)
    subsample_range: Tuple[float, float] = (0.6, 1.0)
    colsample_bytree_range: Tuple[float, float] = (0.6, 1.0)
    reg_alpha_range: Tuple[float, float] = (1e-8, 10.0)
    reg_lambda_range: Tuple[float, float] = (1e-8, 10.0)

    # Fixed params for determinism
    include_deterministic_params: bool = True

    def suggest(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest LightGBM hyperparameters."""
        params = {
            "n_estimators": trial.suggest_int("n_estimators", *self.n_estimators_range),
            "max_depth": trial.suggest_int("max_depth", *self.max_depth_range),
            "learning_rate": trial.suggest_float(
                "learning_rate", *self.learning_rate_range, log=True
            ),
            "num_leaves": trial.suggest_int("num_leaves", *self.num_leaves_range),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", *self.min_child_samples_range
            ),
            "subsample": trial.suggest_float("subsample", *self.subsample_range),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", *self.colsample_bytree_range
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", *self.reg_alpha_range, log=True
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", *self.reg_lambda_range, log=True
            ),
        }

        if self.include_deterministic_params:
            params.update(
                {
                    "deterministic": True,
                    "force_col_wise": True,
                    "verbose": -1,
                }
            )

        return params

    def validate(self, params: Dict[str, Any]) -> None:
        """
        Validate LightGBM hyperparameters.

        Raises:
            InvalidSearchSpaceError: If params violate constraints.
        """
        # num_leaves should be <= 2^max_depth
        if "num_leaves" in params and "max_depth" in params:
            max_leaves = 2 ** params["max_depth"]
            if params["num_leaves"] > max_leaves:
                raise InvalidSearchSpaceError(
                    f"num_leaves ({params['num_leaves']}) exceeds 2^max_depth ({max_leaves})",
                    param_name="num_leaves",
                    param_value=params["num_leaves"],
                    constraint=f"num_leaves <= 2^max_depth = {max_leaves}",
                )

        # learning_rate sanity check
        if params.get("learning_rate", 0.1) <= 0:
            raise InvalidSearchSpaceError(
                "learning_rate must be positive",
                param_name="learning_rate",
                param_value=params.get("learning_rate"),
                constraint="learning_rate > 0",
            )

        # n_estimators sanity check
        if params.get("n_estimators", 100) < 1:
            raise InvalidSearchSpaceError(
                "n_estimators must be at least 1",
                param_name="n_estimators",
                param_value=params.get("n_estimators"),
                constraint="n_estimators >= 1",
            )

    def get_default_params(self) -> Dict[str, Any]:
        """Get default LightGBM parameters."""
        return {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "deterministic": True,
            "force_col_wise": True,
            "verbose": -1,
        }


@dataclass
class MLPSearchSpace:
    """
    Search space for PyTorch MLP models.

    Supports architecture search (layers, units) and training hyperparameters.
    """

    # Architecture
    n_layers_range: Tuple[int, int] = (1, 4)
    units_per_layer_range: Tuple[int, int] = (16, 256)
    activation_choices: List[str] = field(
        default_factory=lambda: ["ReLU", "GELU", "LeakyReLU", "Tanh"]
    )

    # Regularization
    dropout_range: Tuple[float, float] = (0.0, 0.5)
    use_batchnorm: bool = True

    # Training
    learning_rate_range: Tuple[float, float] = (1e-5, 1e-2)
    batch_size_choices: List[int] = field(
        default_factory=lambda: [16, 32, 64, 128, 256]
    )
    weight_decay_range: Tuple[float, float] = (1e-6, 1e-2)

    def suggest(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest MLP hyperparameters."""
        n_layers = trial.suggest_int("n_layers", *self.n_layers_range)

        hidden_layers = []
        for i in range(n_layers):
            units = trial.suggest_int(
                f"units_layer_{i}",
                *self.units_per_layer_range,
                step=16,
            )
            hidden_layers.append(units)

        params = {
            "hidden_layers": hidden_layers,
            "activation": trial.suggest_categorical(
                "activation", self.activation_choices
            ),
            "dropout_rate": trial.suggest_float("dropout_rate", *self.dropout_range),
            "use_batchnorm": (
                trial.suggest_categorical("use_batchnorm", [True, False])
                if not self.use_batchnorm
                else True
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", *self.learning_rate_range, log=True
            ),
            "batch_size": trial.suggest_categorical(
                "batch_size", self.batch_size_choices
            ),
            "weight_decay": trial.suggest_float(
                "weight_decay", *self.weight_decay_range, log=True
            ),
        }

        return params

    def validate(self, params: Dict[str, Any]) -> None:
        """Validate MLP hyperparameters."""
        if "hidden_layers" in params:
            if len(params["hidden_layers"]) == 0:
                raise InvalidSearchSpaceError(
                    "hidden_layers cannot be empty",
                    param_name="hidden_layers",
                    param_value=params["hidden_layers"],
                    constraint="len(hidden_layers) >= 1",
                )

            for i, units in enumerate(params["hidden_layers"]):
                if units < 1:
                    raise InvalidSearchSpaceError(
                        f"Layer {i} units must be positive",
                        param_name=f"hidden_layers[{i}]",
                        param_value=units,
                        constraint="units >= 1",
                    )

        if params.get("dropout_rate", 0.0) < 0 or params.get("dropout_rate", 0.0) >= 1:
            raise InvalidSearchSpaceError(
                "dropout_rate must be in [0, 1)",
                param_name="dropout_rate",
                param_value=params.get("dropout_rate"),
                constraint="0 <= dropout_rate < 1",
            )

    def get_default_params(self) -> Dict[str, Any]:
        """Get default MLP parameters."""
        return {
            "hidden_layers": [64, 32],
            "activation": "GELU",
            "dropout_rate": 0.2,
            "use_batchnorm": True,
            "learning_rate": 1e-3,
            "batch_size": 32,
            "weight_decay": 1e-5,
        }


@dataclass
class XGBoostSearchSpace:
    """Search space for XGBoost models."""

    n_estimators_range: Tuple[int, int] = (50, 500)
    max_depth_range: Tuple[int, int] = (3, 12)
    learning_rate_range: Tuple[float, float] = (0.01, 0.3)
    min_child_weight_range: Tuple[int, int] = (1, 10)
    subsample_range: Tuple[float, float] = (0.6, 1.0)
    colsample_bytree_range: Tuple[float, float] = (0.6, 1.0)
    gamma_range: Tuple[float, float] = (0.0, 5.0)
    reg_alpha_range: Tuple[float, float] = (1e-8, 10.0)
    reg_lambda_range: Tuple[float, float] = (1e-8, 10.0)

    def suggest(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest XGBoost hyperparameters."""
        return {
            "n_estimators": trial.suggest_int("n_estimators", *self.n_estimators_range),
            "max_depth": trial.suggest_int("max_depth", *self.max_depth_range),
            "learning_rate": trial.suggest_float(
                "learning_rate", *self.learning_rate_range, log=True
            ),
            "min_child_weight": trial.suggest_int(
                "min_child_weight", *self.min_child_weight_range
            ),
            "subsample": trial.suggest_float("subsample", *self.subsample_range),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", *self.colsample_bytree_range
            ),
            "gamma": trial.suggest_float("gamma", *self.gamma_range),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", *self.reg_alpha_range, log=True
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", *self.reg_lambda_range, log=True
            ),
            "tree_method": "hist",
            "verbosity": 0,
        }

    def validate(self, params: Dict[str, Any]) -> None:
        """Validate XGBoost hyperparameters."""
        if params.get("max_depth", 6) < 1:
            raise InvalidSearchSpaceError(
                "max_depth must be at least 1",
                param_name="max_depth",
                param_value=params.get("max_depth"),
                constraint="max_depth >= 1",
            )

        if params.get("learning_rate", 0.1) <= 0:
            raise InvalidSearchSpaceError(
                "learning_rate must be positive",
                param_name="learning_rate",
                param_value=params.get("learning_rate"),
                constraint="learning_rate > 0",
            )

    def get_default_params(self) -> Dict[str, Any]:
        """Get default XGBoost parameters."""
        return {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "min_child_weight": 1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "verbosity": 0,
        }
