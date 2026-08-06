"""Feature engineering helpers for time-series forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import holidays
except Exception:  # pragma: no cover - optional dependency guard
    holidays = None


def add_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Add calendar-based features to a time-indexed or timestamp-column DataFrame."""

    out = df.copy()
    ts = pd.Series(pd.to_datetime(out[timestamp_col] if timestamp_col in out.columns else out.index), index=out.index)

    out["hour"] = ts.dt.hour.to_numpy()
    out["day_of_week"] = ts.dt.dayofweek.to_numpy()
    out["month"] = ts.dt.month.to_numpy()
    out["day"] = ts.dt.day.to_numpy()
    out["weekofyear"] = ts.dt.isocalendar().week.astype(int).to_numpy()
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    return out


def add_holiday_feature(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    country: str | None = "PT",
) -> pd.DataFrame:
    """Add a holiday indicator for an explicit country or disable it."""
    out = df.copy()
    ts = pd.Series(
        pd.to_datetime(out[timestamp_col] if timestamp_col in out.columns else out.index),
        index=out.index,
    )
    if country is None:
        out["is_holiday"] = 0
        return out
    if holidays is None:
        raise RuntimeError("python-holidays is required when a holiday country is requested.")
    supported = holidays.list_supported_countries()
    if country.upper() not in supported:
        raise ValueError(f"Unsupported holiday country: {country}")
    calendar = holidays.country_holidays(country.upper())
    out["is_holiday"] = ts.dt.date.map(lambda date: int(date in calendar)).to_numpy()
    return out


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "load",
    lags: list[int] | tuple[int, ...] = (1, 4, 24, 96),
) -> pd.DataFrame:
    """Add lag features for forecasting."""

    out = df.copy()
    for lag in lags:
        out[f"{target_col}_lag_{lag}"] = out[target_col].shift(lag)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "load",
    windows: list[int] | tuple[int, ...] = (4, 24, 96),
) -> pd.DataFrame:
    """Add rolling mean and rolling std features."""

    out = df.copy()
    for window in windows:
        out[f"{target_col}_rolling_mean_{window}"] = out[target_col].shift(1).rolling(window).mean()
        out[f"{target_col}_rolling_std_{window}"] = out[target_col].shift(1).rolling(window).std()
    return out


def build_supervised_frame(
    df: pd.DataFrame,
    target_col: str = "load",
    lags: list[int] | tuple[int, ...] = (1, 4, 24, 96),
    windows: list[int] | tuple[int, ...] = (4, 24, 96),
) -> pd.DataFrame:
    """Create a feature-rich supervised learning table."""

    out = add_time_features(df)
    out = add_holiday_feature(out)
    out = add_lag_features(out, target_col=target_col, lags=lags)
    out = add_rolling_features(out, target_col=target_col, windows=windows)
    out = out.dropna().copy()
    return out


BASELINE_LAGS = (1, 4, 96, 192, 672)
BASELINE_WINDOWS = (4, 96, 672)


def add_cyclical_time_features(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Add continuous cyclical encodings for linear models."""

    out = df.copy()
    ts = pd.Series(
        pd.to_datetime(out[timestamp_col] if timestamp_col in out.columns else out.index),
        index=out.index,
    )
    hour = ts.dt.hour + ts.dt.minute / 60.0
    day_of_week = ts.dt.dayofweek + hour / 24.0
    month = ts.dt.month - 1

    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    out["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    return out


def build_baseline_features(series: pd.Series, country: str | None = "PT") -> pd.DataFrame:
    """Build origin-time features known at or before each forecast origin."""

    if not isinstance(series.index, pd.DatetimeIndex) or not series.index.is_monotonic_increasing:
        raise ValueError("series must use an increasing DatetimeIndex.")
    if len(series) <= max(BASELINE_LAGS):
        raise ValueError("series is too short for the 672-step historical context.")

    out = series.rename("load").to_frame()
    out = add_time_features(out)
    out = add_holiday_feature(out, country=country)
    out = add_cyclical_time_features(out)
    out = add_lag_features(out, lags=BASELINE_LAGS)
    out = add_rolling_features(out, windows=BASELINE_WINDOWS)
    out["current_load"] = out["load"]

    columns = [
        "current_load",
        *[f"load_lag_{lag}" for lag in BASELINE_LAGS],
        *[
            name
            for window in BASELINE_WINDOWS
            for name in (
                f"load_rolling_mean_{window}",
                f"load_rolling_std_{window}",
            )
        ],
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",
        "is_weekend",
        "is_holiday",
    ]
    return out[columns].dropna().copy()
