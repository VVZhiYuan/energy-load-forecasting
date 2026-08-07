import numpy as np
import pandas as pd
import pytest

from src.deep_learning import (
    GRUConfig,
    LoadScaler,
    make_sequence_windows,
    split_sequence_windows,
)


def make_series(length: int = 1000) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=length, freq="15min")
    return pd.Series(np.arange(length, dtype=float), index=index, name="load")


def test_sequence_windows_keep_historical_context_and_future_targets():
    series = make_series(240)

    windows = make_sequence_windows(series, horizon=4, context_steps=96)

    assert windows.inputs.dtype == np.float32
    assert windows.inputs.shape[1:] == (96, 1)
    assert windows.index[0] == series.index[95]
    np.testing.assert_allclose(windows.inputs[0, :, 0], series.iloc[:96])
    np.testing.assert_allclose(windows.targets.iloc[0], series.iloc[96:100])


def test_96_step_partition_keeps_complete_targets_in_each_time_boundary():
    series = make_series(1500)
    windows = make_sequence_windows(series, horizon=96, context_steps=96)

    partitions = split_sequence_windows(windows)

    train_end = int(len(series) * 0.7)
    validation_end = int(len(series) * 0.85)
    positions = {
        name: series.index.get_indexer(partition.index)
        for name, partition in partitions.items()
    }

    assert set(partitions) == {"train", "validation", "test"}
    assert all(len(partition.index) > 0 for partition in partitions.values())
    assert np.all(positions["train"] + 96 < train_end)
    assert np.all(positions["validation"] >= train_end)
    assert np.all(positions["validation"] + 96 < validation_end)
    assert np.all(positions["test"] >= validation_end)
    assert np.all(positions["test"] + 96 < len(series))


def test_load_scaler_fits_training_inputs_only_and_preserves_shapes():
    train_inputs = np.array([[[1.0]], [[3.0]]], dtype=np.float32)
    validation_inputs = np.array([[[100.0]]], dtype=np.float32)

    scaler = LoadScaler().fit(train_inputs)
    transformed_validation = scaler.transform(validation_inputs)

    assert scaler.mean == 2.0
    assert scaler.std == 1.0
    assert transformed_validation.shape == validation_inputs.shape
    np.testing.assert_allclose(transformed_validation, [[[98.0]]])
    np.testing.assert_allclose(scaler.inverse_transform(transformed_validation), validation_inputs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"context_steps": 0},
        {"hidden_size": 0},
        {"num_layers": 0},
        {"batch_size": 0},
        {"epochs": 0},
        {"learning_rate": 0.0},
        {"learning_rate": np.nan},
        {"learning_rate": np.inf},
        {"learning_rate": -np.inf},
        {"patience": 0},
        {"epochs": 3, "patience": 3},
    ],
)
def test_gru_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        GRUConfig(**kwargs)


@pytest.mark.parametrize(
    "series",
    [
        np.arange(240, dtype=float),
        pd.Series(np.arange(240, dtype=float)),
        pd.Series(
            [0.0, np.nan] + list(np.arange(238, dtype=float)),
            index=pd.date_range("2024-01-01", periods=240, freq="15min"),
        ),
    ],
)
def test_sequence_windows_reject_malformed_series(series):
    with pytest.raises(ValueError):
        make_sequence_windows(series, horizon=4, context_steps=96)


def test_sequence_windows_rejects_series_without_a_complete_window():
    with pytest.raises(ValueError, match="too short"):
        make_sequence_windows(make_series(99), horizon=4, context_steps=96)


def test_split_sequence_windows_rejects_mismatched_horizon():
    windows = make_sequence_windows(make_series(1500), horizon=96, context_steps=96)

    with pytest.raises(ValueError, match="horizon"):
        split_sequence_windows(windows, horizon=1)
