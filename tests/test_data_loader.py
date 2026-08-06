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


def test_rejects_ambiguous_long_input_with_extra_columns(tmp_path):
    path = tmp_path / "ambiguous.csv"
    pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=8, freq="15min"),
        "load": np.arange(8, dtype=float),
        "metadata": np.arange(8, dtype=float) + 100,
    }).to_csv(path, index=False)

    with pytest.raises(ValueError, match="ambiguous"):
        load_forecast_series(path, meter="metadata")


def test_rejects_empty_input_with_controlled_error(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("timestamp,load\n", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        load_forecast_series(path)


def test_records_negative_load_count(tmp_path):
    path = tmp_path / "net-load.csv"
    pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=8, freq="15min"),
        "load": [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    }).to_csv(path, index=False)

    loaded = load_forecast_series(path)

    assert loaded.negative_load_count == 1


@pytest.mark.parametrize("bad_value", [np.inf, np.nan])
def test_rejects_non_finite_or_missing_load(tmp_path, bad_value):
    path = tmp_path / "invalid-load.csv"
    values = np.arange(8, dtype=float)
    values[3] = bad_value
    pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=8, freq="15min"),
        "load": values,
    }).to_csv(path, index=False)

    with pytest.raises(ValueError, match="finite numeric"):
        load_forecast_series(path)
