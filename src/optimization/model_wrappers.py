# src/optimization/model_wrappers.py
"""
Model wrapper utilities for seamless Optuna integration.

Provides factory functions and wrappers that allow existing models
to work with OptunaStudyManager without modifying model internals.

Usage:
    from src.optimization.model_wrappers import create_monte_carlo_ml_optimizer

    result = create_monte_carlo_ml_optimizer(
        X_train, y_train, X_val, y_val,
        n_trials=50,
    )
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd


def create_monte_carlo_ml_optimizer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    study_name: str = "monte_carlo_ml",
    seed: int = 42,
    save_results: bool = True,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run Optuna optimization for MonteCarloMLSurrogate.
    
    This is a convenience wrapper that handles the full optimization flow
    without requiring the user to understand all Optuna internals.
    
    Args:
        X_train: Training features (n_samples, n_features).
        y_train: Training targets (n_samples, n_outputs).
        X_val: Validation features.
        y_val: Validation targets.
        n_trials: Number of Optuna trials.
        study_name: Name for the study.
        seed: Random seed for reproducibility.
        save_results: Whether to save results to disk.
        output_dir: Directory for saving results.
    
    Returns:
        Dict with best_params, best_score, study_result, and trained_model.
    """
    from src.optimization import (
        OptunaStudyManager,
        LightGBMSearchSpace,
        create_lgbm_objective,
    )
    from src.pricing_models import MonteCarloMLSurrogate
    
    output_dir = output_dir or Path("models/optimization_results")
    
    # Create model factory
    def model_factory(**params) -> MonteCarloMLSurrogate:
        # Filter to only valid MonteCarloMLSurrogate params
        valid_params = {
            "n_estimators": params.get("n_estimators", 200),
            "max_depth": params.get("max_depth", 6),
            "learning_rate": params.get("learning_rate", 0.1),
            "seed": seed,
        }
        model = MonteCarloMLSurrogate(**valid_params)
        return model
    
    # Create a wrapper that fits and returns predictions
    class FittableWrapper:
        def __init__(self, **params):
            self.params = params
            self.model = model_factory(**params)
        
        def fit(self, X, y):
            # Since MonteCarloMLSurrogate generates its own data,
            # we'll use sklearn-style fit for the internal model
            if hasattr(self.model, 'model') and self.model.model is not None:
                self.model.model.fit(X, y)
                self.model.trained = True
            return self
        
        def predict(self, X):
            if hasattr(self.model, 'model') and self.model.model is not None:
                return self.model.model.predict(X)
            return np.zeros(len(X))
    
    # Create study manager
    manager = OptunaStudyManager(
        study_name=study_name,
        storage=f"sqlite:///{output_dir / 'optuna_studies.db'}",
        seed=seed,
    )
    
    # Create search space
    search_space = LightGBMSearchSpace(
        n_estimators_range=(100, 500),
        max_depth_range=(4, 10),
        learning_rate_range=(0.01, 0.2),
        num_leaves_range=(15, 63),
    )
    
    # Create objective
    objective = create_lgbm_objective(
        model_factory=lambda **p: FittableWrapper(**p),
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        metric="rmse",
    )
    
    # Run optimization
    result = manager.optimize(
        objective=objective,
        search_space=search_space,
        n_trials=n_trials,
        show_progress_bar=False,
    )
    
    # Train final model with best params
    final_model = MonteCarloMLSurrogate(
        n_estimators=result.best_params.get("n_estimators", 200),
        max_depth=result.best_params.get("max_depth", 6),
        learning_rate=result.best_params.get("learning_rate", 0.1),
        seed=seed,
    )
    
    # Save results
    if save_results:
        output_dir.mkdir(parents=True, exist_ok=True)
        result.save(output_dir / f"{study_name}_result.json")
    
    return {
        "best_params": result.best_params,
        "best_score": result.best_value,
        "study_result": result,
        "model_factory": lambda: MonteCarloMLSurrogate(
            n_estimators=result.best_params.get("n_estimators", 200),
            max_depth=result.best_params.get("max_depth", 6),
            learning_rate=result.best_params.get("learning_rate", 0.1),
            seed=seed,
        ),
        "n_trials": result.n_trials,
        "n_complete": result.n_complete,
        "n_pruned": result.n_pruned,
        "duration_seconds": result.duration_seconds,
    }


def create_mlp_optimizer(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_trials: int = 30,
    study_name: str = "mlp_vol_surface",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run Optuna optimization for MLPModel volatility surface.
    
    Args:
        train_df: Training DataFrame with features and implied_volatility.
        val_df: Validation DataFrame.
        n_trials: Number of Optuna trials.
        study_name: Name for the study.
        seed: Random seed.
    
    Returns:
        Dict with best_params, best_score, and model_factory.
    """
    from src.optimization import (
        OptunaStudyManager,
        MLPSearchSpace,
    )
    import optuna
    
    output_dir = Path("models/optimization_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create study manager
    manager = OptunaStudyManager(
        study_name=study_name,
        storage=f"sqlite:///{output_dir / 'optuna_studies.db'}",
        seed=seed,
    )
    
    search_space = MLPSearchSpace()
    
    def objective(trial: optuna.Trial, trial_seed: int) -> float:
        # Import here to avoid circular imports
        from src.volatility_surface.models.mlp_model import MLPModel
        
        params = {k: v for k, v in trial.params.items()}
        
        # Create model with suggested params
        model = MLPModel(
            hidden_layers=params.get("hidden_layers", [64, 32]),
            activation=params.get("activation", "GELU"),
            dropout_rate=params.get("dropout_rate", 0.2),
            learning_rate=params.get("learning_rate", 1e-3),
            batch_size=params.get("batch_size", 32),
            epochs=50,
            early_stopping_patience=10,
            random_seed=trial_seed,
        )
        
        # Train and get validation loss
        try:
            result = model._train_impl(train_df, val_split=0.2)
            return result["val_loss"]
        except Exception:
            return float("inf")
    
    result = manager.optimize(
        objective=objective,
        search_space=search_space,
        n_trials=n_trials,
        show_progress_bar=False,
    )
    
    result.save(output_dir / f"{study_name}_result.json")
    
    return {
        "best_params": result.best_params,
        "best_score": result.best_value,
        "study_result": result,
    }


def optimize_and_export_onnx(
    model: Any,
    X_test: np.ndarray,
    feature_names: List[str],
    output_path: Path,
    model_type: str = "lightgbm",
) -> Dict[str, Any]:
    """
    Export model to ONNX and validate.
    
    Args:
        model: Trained model to export.
        X_test: Test data for validation.
        feature_names: Ordered feature names.
        output_path: Path for ONNX file.
        model_type: "lightgbm" or "pytorch".
    
    Returns:
        Dict with export_result and validation_result.
    """
    from src.optimization import ONNXExporter, ONNXValidator
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Export
    if model_type == "lightgbm":
        export_result = ONNXExporter.export_lightgbm(
            model=model,
            output_path=output_path,
            feature_names=feature_names,
        )
    else:
        import torch
        dummy_input = torch.randn(1, len(feature_names))
        model.eval()
        export_result = ONNXExporter.export_pytorch(
            model=model,
            dummy_input=dummy_input,
            output_path=output_path,
            feature_names=feature_names,
        )
    
    # Validate
    if export_result.success:
        validator = ONNXValidator(rtol=1e-3, atol=1e-4)
        validation_result = validator.validate(
            native_model=model,
            onnx_path=output_path,
            X_test=X_test,
        )
    else:
        validation_result = None
    
    return {
        "export_result": export_result,
        "validation_result": validation_result,
    }
