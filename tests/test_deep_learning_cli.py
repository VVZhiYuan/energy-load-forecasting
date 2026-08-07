import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import deep_learning_benchmark
import src.deep_learning_reporting as deep_learning_reporting


def make_fake_run(horizon: int):
    index = pd.date_range("2025-01-01", periods=3, freq="15min")
    forecast_index = pd.date_range("2025-01-02", periods=horizon, freq="15min")
    training = SimpleNamespace(
        device="cpu",
        best_epoch=2,
        history=pd.DataFrame(
            {"epoch": [1, 2], "train_loss": [1.0, 0.5], "validation_mae": [2.0, 1.0]}
        ),
        scaler=SimpleNamespace(mean=10.0, std=2.0),
        validation_metrics={"MAE": 1.0, "RMSE": 1.2},
    )
    partition = SimpleNamespace(inputs=np.zeros((3, 96, 1)), targets=pd.DataFrame(np.zeros((3, horizon))))
    return SimpleNamespace(
        horizon=horizon,
        config=SimpleNamespace(
            context_steps=96,
            hidden_size=8,
            num_layers=1,
            batch_size=16,
            epochs=3,
            learning_rate=0.001,
            patience=1,
            seed=42,
        ),
        partitions={"train": partition, "validation": partition, "test": partition},
        training=training,
        test_prediction=np.full((3, horizon), 11.0),
        test_metrics={"MAE": 1.5, "RMSE": 1.8},
        latest_forecast=pd.DataFrame(
            {
                "step": np.arange(1, horizon + 1),
                "prediction": np.full(horizon, 11.0),
                "p10": np.full(horizon, 9.0),
                "p50": np.full(horizon, 11.0),
                "p90": np.full(horizon, 13.0),
                "interval_method": "validation_residual_calibration",
            },
            index=forecast_index,
        ),
        validation_residual_quantiles=np.tile(np.array([-2.0, 0.0, 2.0]), (horizon, 1)),
    )


def test_parser_supports_both_horizons_and_gru_overrides():
    args = deep_learning_benchmark.build_parser().parse_args(
        ["--horizon", "both", "--hidden-size", "8", "--epochs", "3"]
    )

    assert args.horizon == "both"
    assert args.hidden_size == 8
    assert args.epochs == 3
    assert args.context_steps == 96


def test_main_returns_two_when_pytorch_is_unavailable(monkeypatch, capsys):
    monkeypatch.setattr(
        deep_learning_benchmark,
        "run_gru_benchmark",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("PyTorch is required")),
    )
    monkeypatch.setattr(
        deep_learning_benchmark,
        "load_forecast_series",
        lambda *args, **kwargs: SimpleNamespace(
            series=pd.Series(
                np.arange(200.0), index=pd.date_range("2025-01-01", periods=200, freq="15min")
            ),
            source_label="fixture",
        ),
    )

    assert deep_learning_benchmark.main(["--input", "fixture.csv", "--horizon", "1h"]) == 2
    assert "Error: PyTorch is required" in capsys.readouterr().err


def test_main_publishes_complete_gru_report(monkeypatch, tmp_path):
    captured_horizons = []

    def fake_loaded(path, meter):
        return SimpleNamespace(
            series=pd.Series(
                np.arange(200.0), index=pd.date_range("2025-01-01", periods=200, freq="15min")
            ),
            source_label="fixture_meter",
            input_format="long",
            meter=meter,
            negative_load_count=0,
        )

    def fake_run(series, horizon, config):
        captured_horizons.append(horizon)
        return make_fake_run(horizon)

    monkeypatch.setattr(deep_learning_benchmark, "load_forecast_series", fake_loaded)
    monkeypatch.setattr(deep_learning_benchmark, "run_gru_benchmark", fake_run)
    output = tmp_path / "report"

    assert (
        deep_learning_benchmark.main(
            ["--input", "fixture.csv", "--horizon", "both", "--output-dir", str(output)]
        )
        == 0
    )

    assert captured_horizons == [4, 96]
    for label in ("1h", "24h"):
        report_dir = output / label
        assert (report_dir / "metrics.json").is_file()
        assert (report_dir / "comparison.csv").is_file()
        assert (report_dir / "forecast.csv").is_file()
        assert (report_dir / "forecast.png").stat().st_size > 0
        assert (report_dir / "training_history.csv").is_file()
        assert json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))["horizon"] == label


def test_main_forwards_all_gru_config_overrides(monkeypatch, tmp_path):
    captured = []
    loaded = SimpleNamespace(
        series=pd.Series(
            np.arange(200.0), index=pd.date_range("2025-01-01", periods=200, freq="15min")
        ),
        source_label="fixture",
    )

    def fake_run(series, horizon, config):
        captured.append((horizon, config))
        return make_fake_run(horizon)

    monkeypatch.setattr(
        deep_learning_benchmark, "load_forecast_series", lambda *args, **kwargs: loaded
    )
    monkeypatch.setattr(deep_learning_benchmark, "run_gru_benchmark", fake_run)

    assert (
        deep_learning_benchmark.main(
            [
                "--input",
                "fixture.csv",
                "--horizon",
                "1h",
                "--output-dir",
                str(tmp_path / "report"),
                "--context-steps",
                "8",
                "--hidden-size",
                "12",
                "--num-layers",
                "2",
                "--batch-size",
                "32",
                "--epochs",
                "5",
                "--learning-rate",
                "0.02",
                "--patience",
                "2",
                "--seed",
                "7",
            ]
        )
        == 0
    )

    assert len(captured) == 1
    horizon, config = captured[0]
    assert horizon == 4
    assert vars(config) == {
        "context_steps": 8,
        "hidden_size": 12,
        "num_layers": 2,
        "batch_size": 32,
        "epochs": 5,
        "learning_rate": 0.02,
        "patience": 2,
        "seed": 7,
    }


def test_publish_failure_restores_the_prior_complete_report(monkeypatch, tmp_path):
    output = tmp_path / "report"
    observed = pd.Series(
        np.arange(200.0), index=pd.date_range("2025-01-01", periods=200, freq="15min")
    )
    deep_learning_reporting.write_gru_artifacts(
        make_fake_run(4),
        observed,
        output,
        source_label="fixture",
        horizon_label="1h",
        runtime_seconds=1.0,
    )
    prior_files = {
        path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
    }
    original_replace = Path.replace

    def fail_staged_directory_replace(path, target):
        if path.name.startswith(".report.staging-") and Path(target) == output:
            raise OSError("simulated directory publish failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_staged_directory_replace)

    with pytest.raises(OSError, match="simulated directory publish failure"):
        deep_learning_reporting.write_gru_artifacts(
            make_fake_run(4),
            observed,
            output,
            source_label="fixture",
            horizon_label="1h",
            runtime_seconds=2.0,
        )

    assert {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()} == prior_files
