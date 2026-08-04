import numpy as np
import pandas as pd

from src.features import add_cyclical_time_features, build_baseline_features


def make_series(length: int = 800) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=length, freq="15min")
    values = 100 + 10 * np.sin(np.arange(length) * 2 * np.pi / 96)
    return pd.Series(values, index=index, name="load")


def test_cyclical_features_wrap_at_daily_boundary():
    frame = make_series(97).to_frame()
    result = add_cyclical_time_features(frame)

    assert np.isclose(result.iloc[0]["hour_sin"], result.iloc[96]["hour_sin"])
    assert np.isclose(result.iloc[0]["hour_cos"], result.iloc[96]["hour_cos"])


def test_baseline_features_start_after_full_historical_context():
    series = make_series()
    features = build_baseline_features(series)

    assert features.index[0] == series.index[672]
    assert features.loc[series.index[672], "current_load"] == series.iloc[672]
    assert features.loc[series.index[672], "load_lag_672"] == series.iloc[0]
    assert features.loc[series.index[672], "load_rolling_mean_4"] == series.iloc[668:672].mean()


def test_baseline_features_contain_only_expected_columns():
    features = build_baseline_features(make_series())
    assert list(features.columns) == [
        "current_load",
        "load_lag_1",
        "load_lag_4",
        "load_lag_96",
        "load_lag_192",
        "load_lag_672",
        "load_rolling_mean_4",
        "load_rolling_std_4",
        "load_rolling_mean_96",
        "load_rolling_std_96",
        "load_rolling_mean_672",
        "load_rolling_std_672",
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",
        "is_weekend",
        "is_holiday",
    ]
    assert not features.isna().any().any()
