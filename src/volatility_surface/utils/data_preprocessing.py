# src / volatility_surface / utils / data_preprocessing.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import logging
import optional
from typing import Tuple, Dict


# Configure logging
logger = logging.getLogger(__name__)

def scale_data(X: pd.DataFrame, scaler: Optional[StandardScaler] = None) -> Tuple[np.ndarray, StandardScaler]:
    """
    Scale features using StandardScaler with leakage prevention
    Returns:
        Scaled features and fitted StandardScaler
    """
    if not isinstance(X, np.ndarray):
        X = X.values
        
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    return X_scaled, scaler

def validate_domain(X: np.ndarray, train_scaler: StandardScaler) -> Dict[str, bool]:
    """
    Validate input is within training domain using StandardScaler
    Returns:
        Dictionary of feature-wise domain validation
    """
    if not hasattr(train_scaler, "data_min_"):
        raise ValueError("Scaler must be trained before domain validation")
        
    train_min = train_scaler.data_min_
    train_max = train_scaler.data_max_
    train_range = train_max - train_min
    
    domain_issues = {}
    for i in range(X.shape[1]):
        feature_name = f"feature_{i}" if i >= len(train_scaler.feature_names_in_) else train_scaler.feature_names_in_[i]
        within_min = np.all(X[:, i] >= train_min[i] - 3 * train_range[i])
        within_max = np.all(X[:, i] <= train_max[i] + 3 * train_range[i])
        domain_issues[feature_name] = within_min and within_max
        
    if not all(domain_issues.values()):
        logger.warning("Input contains out-of-domain values")
    
    return domain_issues

def inverse_transform(X_scaled: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """
    Inverse transform scaled data to original feature space
    """
    if not hasattr(scaler, "data_min_"):
        raise ValueError("Scaler must be trained before inverse transformation")
        
    return scaler.inverse_transform(X_scaled)