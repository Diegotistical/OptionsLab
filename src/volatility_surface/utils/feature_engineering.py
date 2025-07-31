# src / volatility_surface / utils / feature_engineering.py

import pandas as pd
import numpy as np
import torch
from typing import Union, List
import logging

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

# Centralized feature configuration for both pandas and torch
FEATURE_COLUMNS = [
    'moneyness',        # S / K
    'log_moneyness',    # log(S / K)
    'time_to_maturity', # T
    'ttm_squared',      # T²
    'risk_free_rate',   # r
    'historical_volatility',  # σ_hist
    'volatility_skew'   # Rolling skew / batch deviation
]

# Base columns required in input DataFrame
BASE_COLUMNS = [
    'underlying_price',
    'strike_price',
    'time_to_maturity',
    'risk_free_rate',
    'historical_volatility'
]


def engineer_features(
    data: Union[pd.DataFrame, torch.Tensor]
) -> Union[pd.DataFrame, torch.Tensor]:
    """
    Generate engineered features for volatility surface modeling.
    
    Supports both pandas DataFrame and torch.Tensor inputs.
    - DataFrame: Columns must include S, K, T, r, σ
    - Tensor: Shape [batch_size, 5] -> [S, K, T, r, σ]

    Returns:
        DataFrame or Tensor with 7 engineered features
    """
    if isinstance(data, pd.DataFrame):
        return _engineer_dataframe(data)
    elif isinstance(data, torch.Tensor):
        return _engineer_tensor(data)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def _engineer_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering for pandas DataFrames with NaN‑safe skew."""
    # Validate required columns
    missing = [c for c in BASE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    d = df.copy()

    # 1. Moneyness features
    d['moneyness'] = d['underlying_price'].clip(1e-6) / d['strike_price'].clip(1e-6)
    d['log_moneyness'] = np.log(d['moneyness'].clip(1e-6))

    # 2. Time features
    d['ttm_squared'] = d['time_to_maturity'].clip(1e-6) ** 2

    # 3. Volatility skew (rolling mean, NaN‑safe)
    d['volatility_skew'] = (
        d['historical_volatility'].clip(1e-6)
        - d['historical_volatility'].rolling(window=20, min_periods=1).mean()
    )

    # Return only engineered columns
    return d[FEATURE_COLUMNS]


def _engineer_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Feature engineering for PyTorch tensors.
    
    Expected input shape: [batch_size, 5] (S, K, T, r, σ)
    Output shape: [batch_size, 7]
    """
    if tensor.ndim != 2 or tensor.shape[1] != 5:
        raise ValueError(f"Expected shape [batch_size, 5], got {tuple(tensor.shape)}")

    S = tensor[:, 0].clamp(min=1e-6)
    K = tensor[:, 1].clamp(min=1e-6)
    T = tensor[:, 2].clamp(min=1e-6)
    r = tensor[:, 3]
    vol = tensor[:, 4].clamp(min=1e-6)

    # Compute engineered features
    moneyness = S / K
    log_moneyness = torch.log(moneyness.clamp(min=1e-6))
    ttm_squared = T ** 2

    # Batch-level volatility skew (centered volatility)
    vol_skew = vol - vol.mean()

    # Stack features in correct order
    return torch.stack([
        moneyness,
        log_moneyness,
        T,
        ttm_squared,
        r,
        vol,
        vol_skew
    ], dim=1)


def get_feature_columns() -> List[str]:
    """Return the list of engineered feature names."""
    return FEATURE_COLUMNS


def validate_inputs(
    data: Union[pd.DataFrame, torch.Tensor]
) -> None:
    """
    Validate input contains required base features.
    
    Raises ValueError if required features are missing or shape is invalid.
    """
    if isinstance(data, pd.DataFrame):
        missing = [c for c in BASE_COLUMNS if c not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        logger.info("DataFrame input validated successfully.")

    elif isinstance(data, torch.Tensor):
        if data.ndim != 2 or data.shape[1] < 5:
            raise ValueError("Input tensor must have shape [batch_size, 5] (S, K, T, r, vol)")
        logger.info("Tensor input validated successfully.")

    else:
        raise TypeError(f"Unsupported input type: {type(data)}")
