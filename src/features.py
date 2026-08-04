"""Feature engineering helpers for time-series forecasting."""

from __future__ import annotations

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
    country: str = "PT",
) -> pd.DataFrame:
    """Add a holiday indicator using the dataset's country by default.

    If the holidays package is unavailable or the region is unsupported,
    the function returns the input with a zero holiday flag.
    """

    out = df.copy()
    ts = pd.Series(pd.to_datetime(out[timestamp_col] if timestamp_col in out.columns else out.index), index=out.index)
    holiday_flag = pd.Series(0, index=out.index, dtype=int)

    if holidays is not None:
        try:
            holiday_calendar = holidays.country_holidays(country)
            holiday_flag = ts.dt.date.map(lambda x: 1 if x in holiday_calendar else 0).astype(int)
        except Exception:
            holiday_flag = pd.Series(0, index=out.index, dtype=int)

    out["is_holiday"] = holiday_flag.values if isinstance(holiday_flag, pd.Series) else holiday_flag
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
