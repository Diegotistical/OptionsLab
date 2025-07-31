# src / volatility_surface / utils / data_preprocessing.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging
from typing import Optional, Tuple, Dict, Union

# Configure logger
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)


def scale_data(
    data: Union[pd.DataFrame, np.ndarray],
    scaler: Optional[StandardScaler] = None,
    fit: bool = False
) -> Tuple[Union[pd.DataFrame, np.ndarray], StandardScaler]:
    """
    Scale data with strict leakage prevention.

    Args:
        data: Input data (DataFrame or numpy array)
        scaler: Optional pre-fit scaler
        fit: If True, fit a new scaler. If False, use provided scaler.

    Returns:
        Tuple of:
            - Scaled data (same type as input)
            - StandardScaler object
    """
    if fit and scaler is not None:
        raise ValueError("Cannot provide a scaler when fit=True")

    if isinstance(data, pd.DataFrame):
        if fit:
            scaler = StandardScaler()
            scaled_array = scaler.fit_transform(data.values)
            return pd.DataFrame(scaled_array, columns=data.columns), scaler
        else:
            if scaler is None:
                raise ValueError("Scaler must be provided for prediction")
            scaled_array = scaler.transform(data.values)
            return pd.DataFrame(scaled_array, columns=data.columns), scaler

    elif isinstance(data, np.ndarray):
        if fit:
            scaler = StandardScaler()
            scaled_array = scaler.fit_transform(data)
        else:
            if scaler is None:
                raise ValueError("Scaler must be provided for prediction")
            scaled_array = scaler.transform(data)
        return scaled_array, scaler

    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def validate_domain(
    data: Union[pd.DataFrame, np.ndarray],
    train_scaler: StandardScaler
) -> Dict[str, Union[list, np.ndarray, float]]:
    """
    Validate whether samples are within 3σ of the training distribution.

    Args:
        data: Data to validate
        train_scaler: Fitted StandardScaler

    Returns:
        Dictionary with:
            - feature_names: List of feature names
            - domain_mask: Boolean mask of shape (n_samples,)
            - tolerance_ratio: Fraction of samples within domain
    """
    if not hasattr(train_scaler, "mean_"):
        raise ValueError("Scaler must be trained before validation")

    data_array = data.values if isinstance(data, pd.DataFrame) else data

    if data_array.shape[1] != len(train_scaler.mean_):
        raise ValueError("Data shape does not match number of features in scaler")

    train_mean = train_scaler.mean_
    train_std = train_scaler.scale_

    # Determine feature names
    feature_names = getattr(
        train_scaler,
        "feature_names_in_",
        [f"feature_{i}" for i in range(data_array.shape[1])]
    )

    # Compute per-sample domain mask
    lower_bounds = train_mean - 3 * train_std
    upper_bounds = train_mean + 3 * train_std
    domain_mask = np.all((data_array >= lower_bounds) & (data_array <= upper_bounds), axis=1)

    tolerance_ratio = np.mean(domain_mask)

    if not np.all(domain_mask):
        logger.warning(f"{np.sum(~domain_mask)} samples out of domain ({1 - tolerance_ratio:.2%} OOD)")

    return {
        "feature_names": feature_names,
        "domain_mask": domain_mask,
        "tolerance_ratio": tolerance_ratio
    }


def inverse_transform(
    scaled_data: Union[pd.DataFrame, np.ndarray],
    scaler: StandardScaler
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Inverse transform scaled data back to original feature space.

    Args:
        scaled_data: Data to inverse transform
        scaler: Fitted StandardScaler

    Returns:
        Data in original feature space (same type as input)
    """
    if not hasattr(scaler, "mean_"):
        raise ValueError("Scaler must be trained before inverse transformation")

    if isinstance(scaled_data, pd.DataFrame):
        original_array = scaler.inverse_transform(scaled_data.values)
        return pd.DataFrame(original_array, columns=scaled_data.columns)
    elif isinstance(scaled_data, np.ndarray):
        return scaler.inverse_transform(scaled_data)
    else:
        raise TypeError(f"Unsupported data type: {type(scaled_data)}")
