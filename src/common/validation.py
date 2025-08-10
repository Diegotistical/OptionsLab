# src/common/validation.py

from typing import List
import pandas as pd

def check_required_columns(df: pd.DataFrame, required_columns: List[str]) -> None:
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def check_no_nan(df: pd.DataFrame, columns: List[str]) -> None:
    nan_cols = [col for col in columns if df[col].isnull().any()]
    if nan_cols:
        raise ValueError(f"NaNs detected in columns: {nan_cols}")
