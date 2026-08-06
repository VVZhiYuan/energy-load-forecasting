import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_forecast_series


def test_loads_two_column_forecast_csv(tmp_path):
    path = tmp_path / "building.csv"
    pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=8, freq="15min"),
        "load": np.arange(8, dtype=float),
    }).to_csv(path, index=False)

    loaded = load_forecast_series(path)

    assert loaded.input_format == "long"
    assert loaded.meter is None
    assert loaded.source_label == "building"
    assert loaded.series.name == "load"


def test_loads_uci_style_wide_file(tmp_path):
    path = tmp_path / "meters.txt"
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=8, freq="15min"),
        "MT_001": np.arange(8, dtype=float) + 0.5,
        "MT_002": np.arange(8, dtype=float) + 2.5,
    })
    frame.to_csv(path, sep=";", decimal=",", index=False)

    loaded = load_forecast_series(path, meter="MT_002")

    assert loaded.input_format == "wide"
    assert loaded.meter == "MT_002"
    np.testing.assert_allclose(loaded.series, frame["MT_002"])


def test_rejects_missing_interval(tmp_path):
    path = tmp_path / "bad.csv"
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=8, freq="15min"),
        "load": np.arange(8, dtype=float),
    }).drop(index=3)
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="continuous 15-minute"):
        load_forecast_series(path)


def test_rejects_unknown_wide_meter(tmp_path):
    path = tmp_path / "meters.csv"
    pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=8, freq="15min"),
        "MT_001": np.arange(8, dtype=float),
    }).to_csv(path, index=False)

    with pytest.raises(ValueError, match="MT_999"):
        load_forecast_series(path, meter="MT_999")
