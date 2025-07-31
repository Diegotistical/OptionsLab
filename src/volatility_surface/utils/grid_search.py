# src / volatility_surface / utils / grid_search.py
# src/volatility_surface/utils/grid_search.py
import itertools
import logging
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from ..base import VolatilityModelBase

# Configure logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _split_features_target(df: pd.DataFrame, target_col: str):
    """Safe split of features and target without leakage."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    return X, y


def _generate_param_combinations(param_grid: Dict[str, list]) -> List[Dict[str, Any]]:
    """Generate all parameter combinations safely using itertools.product."""
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def _evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard metrics for regression."""
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred)
    }


#  PUBLIC API 

def tune_model(
    model_class: Union[Type, BaseEstimator],
    df: pd.DataFrame,
    param_grid: Dict[str, list],
    target_col: str = 'implied_volatility',
    cv_folds: int = 5,
    n_jobs: int = -1
) -> Dict[str, Any]:
    """
    Universal hyperparameter tuner for:
    - Custom VolatilityModelBase models
    - sklearn-style estimators
    - sklearn Pipelines

    Returns a dict with:
        best_params, best_score (RMSE), cv_scores, cv_mean, cv_std
    """
    X, y = _split_features_target(df, target_col)

    # Determine model type
    is_custom = (
        isinstance(model_class, type) and issubclass(model_class, VolatilityModelBase)
    )
    is_sklearn_estimator = (
        isinstance(model_class, type) and issubclass(model_class, BaseEstimator)
    )

    if is_custom:
        return _tune_custom_model(model_class, df, param_grid, cv_folds)
    elif is_sklearn_estimator:
        return _tune_sklearn_model(model_class, X, y, param_grid, cv_folds, n_jobs)
    else:
        raise TypeError(
            "model_class must be either VolatilityModelBase subclass or sklearn estimator class."
        )


def _tune_custom_model(
    model_class: Type[VolatilityModelBase],
    df: pd.DataFrame,
    param_grid: Dict[str, list],
    cv_folds: int = 5
) -> Dict[str, Any]:
    """Manual grid search for custom models implementing train/evaluate API."""
    best_score = float("inf")
    best_params = {}
    fold_scores = []

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    param_combos = _generate_param_combinations(param_grid)

    logger.info(f"Tuning {model_class.__name__} with {len(param_combos)} param combos...")

    for params in param_combos:
        scores = []
        for train_idx, val_idx in kf.split(df):
            train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
            model = model_class(**params)
            model.train(train_df)
            preds = model.predict(val_df)
            y_true = val_df['implied_volatility'].values
            metrics = _evaluate_predictions(y_true, preds)
            scores.append(metrics["rmse"])

        mean_rmse = np.mean(scores)
        if mean_rmse < best_score:
            best_score = mean_rmse
            best_params = params
        fold_scores.append(mean_rmse)

    return {
        "best_params": best_params,
        "best_score": best_score,
        "cv_scores": fold_scores,
        "cv_mean": np.mean(fold_scores),
        "cv_std": np.std(fold_scores)
    }


def _tune_sklearn_model(
    model_class: Type[BaseEstimator],
    X: np.ndarray,
    y: np.ndarray,
    param_grid: Dict[str, list],
    cv_folds: int = 5,
    n_jobs: int = -1
) -> Dict[str, Any]:
    """Grid search for sklearn estimators with StandardScaler in pipeline."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model_class())
    ])

    prefixed_grid = {f"model__{k}": v for k, v in param_grid.items()}
    grid_search = GridSearchCV(
        pipeline,
        prefixed_grid,
        cv=KFold(cv_folds, shuffle=True, random_state=42),
        scoring="neg_root_mean_squared_error",
        n_jobs=n_jobs,
        verbose=1
    )
    grid_search.fit(X, y)

    return {
        "best_params": grid_search.best_params_,
        "best_score": -grid_search.best_score_,
        "cv_scores": -grid_search.cv_results_['mean_test_score'],
        "cv_mean": np.mean(-grid_search.cv_results_['mean_test_score']),
        "cv_std": np.std(-grid_search.cv_results_['mean_test_score'])
    }


def nested_cross_validate(
    model_class: Type,
    df: pd.DataFrame,
    param_grid: Dict[str, list],
    target_col: str = 'implied_volatility',
    outer_folds: int = 5,
    inner_folds: int = 3,
    n_jobs: int = -1
) -> Dict[str, Any]:
    """
    Nested cross-validation with inner hyperparameter tuning.
    Works for both sklearn estimators and custom VolatilityModelBase models.
    """
    outer_scores = []
    best_params_list = []

    kf_outer = KFold(n_splits=outer_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf_outer.split(df), start=1):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

        tuner_results = tune_model(
            model_class,
            train_df,
            param_grid,
            target_col=target_col,
            cv_folds=inner_folds,
            n_jobs=n_jobs
        )
        best_params = tuner_results["best_params"]
        best_params_list.append(best_params)

        # Fit on full train fold
        if isinstance(model_class, type) and issubclass(model_class, VolatilityModelBase):
            model = model_class(**best_params)
            model.train(train_df)
            preds = model.predict(val_df)
        else:
            X_train, y_train = _split_features_target(train_df, target_col)
            X_val, y_val = _split_features_target(val_df, target_col)
            model = model_class(**best_params)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)

        y_true = val_df[target_col].values
        metrics = _evaluate_predictions(y_true, preds)
        outer_scores.append(metrics["rmse"])
        logger.info(f"[Fold {fold}] RMSE: {metrics['rmse']:.6f}")

    return {
        "nested_scores": outer_scores,
        "nested_mean": np.mean(outer_scores),
        "nested_std": np.std(outer_scores),
        "best_params_per_fold": best_params_list
    }
