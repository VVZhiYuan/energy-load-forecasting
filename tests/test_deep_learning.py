import numpy as np
import pandas as pd
import pytest

import src.deep_learning as deep_learning
from src.deep_learning import (
    DirectGRU,
    GRUConfig,
    LoadScaler,
    calibrate_residual_intervals,
    fit_direct_gru,
    make_sequence_windows,
    require_torch,
    run_gru_benchmark,
    split_sequence_windows,
)
from src.evaluate import evaluate_multistep


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


def make_small_partitions():
    index = pd.date_range("2024-01-01", periods=300, freq="15min")
    values = 20 + np.sin(np.arange(300) * 2 * np.pi / 24) + np.arange(300) * 0.01
    windows = make_sequence_windows(pd.Series(values, index=index), horizon=3, context_steps=8)
    return split_sequence_windows(windows)


@pytest.mark.skipif(deep_learning.torch is None, reason="PyTorch is not installed")
def test_direct_gru_is_deterministic_and_restores_validation_best_checkpoint():
    partitions = make_small_partitions()
    config = GRUConfig(
        context_steps=8,
        hidden_size=8,
        batch_size=16,
        epochs=3,
        patience=1,
    )

    first = fit_direct_gru(partitions["train"], partitions["validation"], config)
    second = fit_direct_gru(partitions["train"], partitions["validation"], config)

    assert isinstance(first.model, DirectGRU)
    assert first.validation_prediction.shape == (
        len(partitions["validation"].targets),
        3,
    )
    assert first.best_epoch > 0
    assert first.device == ("cuda" if deep_learning.torch.cuda.is_available() else "cpu")
    assert next(first.model.parameters()).is_cuda == deep_learning.torch.cuda.is_available()
    assert first.best_epoch == int(first.history["epoch"].iloc[first.history["validation_mae"].argmin()])
    assert first.validation_metrics["MAE"] == pytest.approx(
        evaluate_multistep(
            partitions["validation"].targets.to_numpy(), first.validation_prediction
        )["MAE"]
    )
    np.testing.assert_allclose(first.validation_prediction, second.validation_prediction, atol=1e-6)


@pytest.mark.skipif(deep_learning.torch is None, reason="PyTorch is not installed")
def test_run_gru_benchmark_scores_test_data_and_returns_calibrated_latest_forecast():
    series = pd.Series(
        25 + np.sin(np.arange(300) * 2 * np.pi / 24) + np.arange(300) * 0.01,
        index=pd.date_range("2024-01-01", periods=300, freq="15min"),
    )
    config = GRUConfig(
        context_steps=8,
        hidden_size=8,
        batch_size=16,
        epochs=2,
        patience=1,
    )

    run = run_gru_benchmark(series, horizon=3, config=config)

    assert run.test_prediction.shape == (len(run.partitions["test"].targets), 3)
    assert run.test_metrics == pytest.approx(
        evaluate_multistep(run.partitions["test"].targets.to_numpy(), run.test_prediction)
    )
    assert run.latest_forecast.index[0] == series.index[-1] + pd.Timedelta(minutes=15)
    assert run.latest_forecast["prediction"].shape == (3,)
    assert np.all(np.diff(run.latest_forecast[["p10", "p50", "p90"]], axis=1) >= 0)
    assert (run.latest_forecast[["p10", "p50", "p90"]].to_numpy() >= 0).all()


def test_residual_calibration_clips_negative_values_and_orders_quantiles():
    point_predictions = np.array([[1.0, 3.0], [0.0, 2.0]])
    validation_residuals = np.array(
        [[-10.0, 3.0], [-5.0, -4.0], [2.0, 1.0]], dtype=float
    )

    intervals = calibrate_residual_intervals(point_predictions, validation_residuals)

    assert intervals.shape == (2, 2, 3)
    assert (intervals >= 0).all()
    assert np.all(np.diff(intervals, axis=2) >= 0)


def test_require_torch_raises_a_concise_installation_error(monkeypatch):
    monkeypatch.setattr(deep_learning, "torch", None)

    with pytest.raises(RuntimeError, match="PyTorch is required"):
        require_torch()
