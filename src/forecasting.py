"""Forecasting utilities for horizon definition and splits."""

from __future__ import annotations

import pandas as pd


def make_time_split(
    df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time series DataFrame into train/validation/test by order."""

    if not 0 < train_size < 1:
        raise ValueError("train_size must be between 0 and 1.")
    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1.")
    if train_size + val_size >= 1:
        raise ValueError("train_size + val_size must be less than 1.")

    n = len(df)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


def make_horizon_target(df: pd.DataFrame, target_col: str = "load", horizon: int = 4) -> pd.DataFrame:
    """Shift the target column into the future for supervised learning."""

    out = df.copy()
    out[f"{target_col}_future_{horizon}"] = out[target_col].shift(-horizon)
    return out

