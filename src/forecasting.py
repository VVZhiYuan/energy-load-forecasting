"""Forecasting utilities for horizon definition and splits."""

from __future__ import annotations

import numpy as np
import pandas as pd

MAX_SUPPORTED_HORIZON = 96
EXPECTED_FREQUENCY = pd.Timedelta(minutes=15)


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


def _validate_time_index(index: pd.DatetimeIndex) -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("series must use a DatetimeIndex.")
    if index.hasnans or not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError("timestamps must be present, unique, and increasing.")
    if len(index) > 1 and not index.to_series().diff().dropna().eq(EXPECTED_FREQUENCY).all():
        raise ValueError("timestamps must have a continuous 15-minute frequency.")


def make_multistep_targets(series: pd.Series, horizon: int) -> pd.DataFrame:
    """Create target columns for every future step after each forecast origin."""

    if not 1 <= horizon <= MAX_SUPPORTED_HORIZON:
        raise ValueError("horizon must be between 1 and 96 steps.")
    _validate_time_index(series.index)
    if len(series) <= horizon:
        raise ValueError("series is too short for the requested horizon.")

    return pd.DataFrame(
        {
            f"target_step_{step}": series.shift(-step)
            for step in range(1, horizon + 1)
        },
        index=series.index,
    )


def split_supervised_by_time(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    full_index: pd.DatetimeIndex,
    horizon: int,
    train_size: float = 0.7,
    val_size: float = 0.15,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Split aligned supervised samples while keeping targets inside partitions."""

    if not 1 <= horizon <= MAX_SUPPORTED_HORIZON:
        raise ValueError("horizon must be between 1 and 96 steps.")
    if not 0 < train_size < 1 or not 0 < val_size < 1 or train_size + val_size >= 1:
        raise ValueError("train_size and val_size must define three non-empty partitions.")
    _validate_time_index(full_index)
    if not features.index.equals(targets.index):
        raise ValueError("features and targets must have identical indexes.")

    origin_positions = full_index.get_indexer(features.index)
    if np.any(origin_positions < 0):
        raise ValueError("every supervised origin must exist in full_index.")

    target_end_positions = origin_positions + horizon
    train_end = int(len(full_index) * train_size)
    validation_end = int(len(full_index) * (train_size + val_size))
    masks = {
        "train": target_end_positions < train_end,
        "validation": (origin_positions >= train_end) & (target_end_positions < validation_end),
        "test": (origin_positions >= validation_end) & (target_end_positions < len(full_index)),
    }

    splits = {}
    for name, mask in masks.items():
        if not np.any(mask):
            raise ValueError(f"{name} partition has no supervised samples.")
        splits[name] = (features.loc[mask].copy(), targets.loc[mask].copy())
    return splits
