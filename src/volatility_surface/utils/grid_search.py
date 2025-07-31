# src / volatility_surface / utils / grid_search.py

from sklearn.model_selection import GridSearchCV, KFold
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Configure logging
logger = logging.getLogger(__name__)

def tune_model(model_class: Any,
              df: pd.DataFrame,
              param_grid: Dict[str, list],
              target_col: str = 'implied_volatility',
              cv_folds: int = 5,
              n_jobs: int = -1) -> Dict[str, Any]:
    """
    Universal hyperparameter tuning for volatility models
    Parameters:
    model_class : class
        Model class with sklearn-style API
    df : pd.DataFrame
        Input data with engineered features
    param_grid : dict
        Dictionary of hyperparameters to search
    target_col : str
        Target column name
    cv_folds : int
        Number of cross-validation folds
    n_jobs : int
        Number of parallel jobs
    Returns:
        Dictionary with best parameters and cross-validation results
    """
    logger.info(f"Tuning {model_class.__name__}")
    
    # Extract features and target
    X = df.values
    y = df[target_col].values
    
    # Create model instance for feature inspection
    base_model = model_class()
    if hasattr(base_model, 'feature_columns'):
        feature_columns = base_model.feature_columns
    else:
        feature_columns = [f"feature_{i}" for i in range(X.shape[1])]
        
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', base_model)
    ])
    
    # Setup grid search with correct parameter prefixes
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=KFold(cv_folds, shuffle=True),
        scoring='neg_mean_squared_error',
        n_jobs=n_jobs,
        verbose=1
    )
    
    # Fit on training data
    grid_search.fit(X, y)
    
    # Update model with best parameters
    best_estimator = grid_search.best_estimator_
    best_params = best_estimator.get_params()
    
    return {
        'best_params': best_params,
        'best_score': -grid_search.best_score_,
        'cv_scores': -grid_search.cv_results_['mean_test_score'],
        'cv_std': grid_search.cv_results_['std_test_score'],
        'feature_columns': feature_columns
    }

def nested_cross_validate(model_class: Any,
                          df: pd.DataFrame,
                          param_grid: Dict[str, list],
                          outer_folds: int = 5,
                          inner_folds: int = 3,
                          n_jobs: int = -1) -> Dict[str, Any]:
    """
    Nested cross-validation for unbiased performance estimation
    Returns:
        Dictionary with cross-validated metrics
    """
    logger.info(f"Running nested cross-validation for {model_class.__name__}")
    
    outer_scores = [] # List to store outer fold scores
    outer_params = [] # List to store best parameters for each outer fold
    
    # Outer cross-validation
    for train_idx, val_idx in KFold(outer_folds, shuffle=True).split(df):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        # Inner tuning
        tuned_params = tune_model(
            model_class,
            train_df,
            param_grid,
            n_jobs=n_jobs
        )
        
        # Create and train best model
        best_model = model_class(**tuned_params['best_params'])
        best_model.fit(train_df)
        
        # Evaluate on validation set
        val_score = best_model.score(val_df)
        outer_scores.append(val_score)
        outer_params.append(tuned_params['best_params'])
        
    return {
        'nested_scores': outer_scores,
        'nested_mean': np.mean(outer_scores),
        'nested_std': np.std(outer_scores),
        'best_params_per_fold': outer_params
    }