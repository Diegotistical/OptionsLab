# common/validation.py

import pandas as pd


def check_required_columns(df: pd.DataFrame, columns: list):
    """
    Ensures all required columns exist in df. Raises ValueError if any missing.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True
