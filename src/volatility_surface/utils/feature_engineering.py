# src/volatility_surface/utils/feature_engineering.py

import pandas as pd
import numpy as np
import torch

def engineer_tensor_features(df: pd.DataFrame) -> np.ndarray:
    """
    Feature engineering for volatility surface modeling
    Returns:
        numpy array with engineered features
    """
    d = df.copy()
    d['moneyness'] = d['underlying_price'].clip(1e-6) / d['strike_price'].clip(1e-6)
    d['log_moneyness'] = np.log(d['moneyness'].clip(1e-6))
    d['ttm_squared'] = d['time_to_maturity'].clip(1e-6) ** 2
    
    # Compute volatility skew safely
    d['volatility_skew'] = (
        d['historical_volatility'].clip(1e-6) - 
        d['historical_volatility'].rolling(20).mean().fillna(0)
    )
    
    return d.values