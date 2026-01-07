# src/optimization/objectives.py
"""
Thin objective wrappers for Optuna optimization.

Provides factory functions that create objective functions
from model factories and data. Models stay clean - no Optuna imports.

Usage:
    from src.optimization import create_lgbm_objective, LightGBMSearchSpace

    objective = create_lgbm_objective(
        model_factory=lambda **p: MonteCarloMLSurrogate(**p),
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        search_space=LightGBMSearchSpace(),
    )
    
    manager.optimize(objective, search_space, n_trials=100)
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import optuna
from sklearn.model_selection import KFold

from src.optimization.reproducibility import get_cv_split_generator
from src.optimization.search_space import SearchSpace


def create_lgbm_objective(
    model_factory: Callable[..., Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    search_space: Optional[SearchSpace] = None,
    metric: str = "rmse",
    n_cv_folds: int = 0,
    cv_seed: int = 42,
) -> Callable[[optuna.Trial, int], float]:
    """
    Create LightGBM objective function for Optuna.
    
    Args:
        model_factory: Callable that takes params and returns a model with fit/predict.
        X_train: Training features.
        y_train: Training targets.
        X_val: Optional validation features (if not using CV).
        y_val: Optional validation targets (if not using CV).
        search_space: SearchSpace for param suggestions (optional if used in manager).
        metric: Metric to optimize ("rmse", "mae", "mse").
        n_cv_folds: If > 0, use cross-validation instead of holdout.
        cv_seed: Seed for CV splits (ensures determinism).
    
    Returns:
        Objective function compatible with OptunaStudyManager.
    """
    # Validate inputs
    if n_cv_folds == 0 and (X_val is None or y_val is None):
        raise ValueError("Must provide X_val/y_val or set n_cv_folds > 0")
    
    # Create deterministic CV splitter
    cv_rng = get_cv_split_generator(cv_seed)
    
    def objective(trial: optuna.Trial, trial_seed: int) -> float:
        # Get params from trial (already suggested by manager)
        params = {k: v for k, v in trial.params.items()}
        params["seed"] = trial_seed
        params["random_state"] = trial_seed
        
        if n_cv_folds > 0:
            # Cross-validation with deterministic splits
            kf = KFold(
                n_splits=n_cv_folds,
                shuffle=True,
                random_state=cv_seed,
            )
            scores = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                X_fold_train = X_train[train_idx]
                y_fold_train = y_train[train_idx]
                X_fold_val = X_train[val_idx]
                y_fold_val = y_train[val_idx]
                
                # Create and train model
                model = model_factory(**params)
                model.fit(X_fold_train, y_fold_train)
                
                # Predict and score
                y_pred = model.predict(X_fold_val)
                score = _compute_metric(y_fold_val, y_pred, metric)
                scores.append(score)
                
                # Report for pruning
                trial.report(np.mean(scores), fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return np.mean(scores)
        else:
            # Holdout validation
            model = model_factory(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            return _compute_metric(y_val, y_pred, metric)
    
    return objective


def create_sklearn_objective(
    model_class: Type,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    metric: str = "rmse",
    n_cv_folds: int = 0,
    cv_seed: int = 42,
) -> Callable[[optuna.Trial, int], float]:
    """
    Create sklearn-compatible objective function.
    
    Works with any model class that has fit(X, y) and predict(X) methods.
    """
    if n_cv_folds == 0 and (X_val is None or y_val is None):
        raise ValueError("Must provide X_val/y_val or set n_cv_folds > 0")
    
    def objective(trial: optuna.Trial, trial_seed: int) -> float:
        params = {k: v for k, v in trial.params.items()}
        
        # Add random_state if the model supports it
        params["random_state"] = trial_seed
        
        if n_cv_folds > 0:
            kf = KFold(n_splits=n_cv_folds, shuffle=True, random_state=cv_seed)
            scores = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                try:
                    model = model_class(**params)
                except TypeError:
                    # Model doesn't accept random_state
                    params.pop("random_state", None)
                    model = model_class(**params)
                
                model.fit(X_train[train_idx], y_train[train_idx])
                y_pred = model.predict(X_train[val_idx])
                score = _compute_metric(y_train[val_idx], y_pred, metric)
                scores.append(score)
                
                trial.report(np.mean(scores), fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return np.mean(scores)
        else:
            try:
                model = model_class(**params)
            except TypeError:
                params.pop("random_state", None)
                model = model_class(**params)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            return _compute_metric(y_val, y_pred, metric)
    
    return objective


def create_pytorch_objective(
    model_class: Type,
    train_loader,
    val_loader,
    device: str = "cpu",
    epochs: int = 50,
    early_stopping_patience: int = 10,
    metric: str = "mse",
) -> Callable[[optuna.Trial, int], float]:
    """
    Create PyTorch model objective function.
    
    Args:
        model_class: PyTorch nn.Module class.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        device: Device to train on ("cpu" or "cuda").
        epochs: Maximum epochs.
        early_stopping_patience: Epochs without improvement before stopping.
        metric: Loss metric ("mse", "mae").
    
    Returns:
        Objective function for Optuna.
    """
    import torch
    import torch.nn as nn
    
    def objective(trial: optuna.Trial, trial_seed: int) -> float:
        # Set seeds
        torch.manual_seed(trial_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(trial_seed)
        
        # Get hyperparameters
        params = {k: v for k, v in trial.params.items()}
        
        # Extract training params
        lr = params.pop("learning_rate", 1e-3)
        weight_decay = params.pop("weight_decay", 1e-5)
        batch_size = params.pop("batch_size", 32)
        
        # Create model
        model = model_class(**params).to(device)
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        
        # Loss function
        criterion = nn.MSELoss() if metric == "mse" else nn.L1Loss()
        
        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                optimizer.zero_grad()
                output = model(batch_X)
                loss = criterion(output.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
            
            # Validate
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    output = model(batch_X)
                    val_loss = criterion(output.squeeze(), batch_y)
                    val_losses.append(val_loss.item())
            
            avg_val_loss = np.mean(val_losses)
            
            # Report for pruning
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    break
        
        return best_val_loss
    
    return objective


def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
) -> float:
    """Compute evaluation metric."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if metric == "rmse":
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    elif metric == "mse":
        return float(np.mean((y_true - y_pred) ** 2))
    elif metric == "mae":
        return float(np.mean(np.abs(y_true - y_pred)))
    elif metric == "mape":
        return float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
    elif metric == "r2":
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1.0 - (ss_res / (ss_tot + 1e-8))
    else:
        raise ValueError(f"Unknown metric: {metric}")
