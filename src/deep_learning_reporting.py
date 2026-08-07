"""Publish auditable report artifacts for a completed GRU benchmark."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.deep_learning import GRUBenchmarkResult


ARTIFACT_NAMES = {
    "metrics_json": "metrics.json",
    "comparison_csv": "comparison.csv",
    "forecast_csv": "forecast.csv",
    "png": "forecast.png",
    "training_history_csv": "training_history.csv",
}
FORECAST_COLUMNS = ("step", "prediction", "p10", "p50", "p90", "interval_method")
COMPARISON_COLUMNS = (
    "model",
    "configuration",
    "validation_mae",
    "validation_rmse",
    "test_mae",
    "test_rmse",
    "selected",
    "training_seconds",
)


def read_saved_comparison(report_dir: str | Path) -> pd.DataFrame:
    """Read an existing classical comparison table when one has been published."""

    path = Path(report_dir) / "model_comparison.csv"
    if not path.is_file():
        return pd.DataFrame(columns=COMPARISON_COLUMNS)
    return pd.read_csv(path)


def write_gru_artifacts(
    result: GRUBenchmarkResult,
    observed: pd.Series,
    output_dir: str | Path,
    *,
    source_label: str,
    horizon_label: str,
    runtime_seconds: float,
    classical_comparison: pd.DataFrame | None = None,
    source_metadata: dict[str, object] | None = None,
) -> dict[str, Path]:
    """Stage, validate, and atomically publish five GRU benchmark artifacts."""

    output_dir = Path(output_dir)
    forecast = _validate_forecast(result.latest_forecast)
    history = result.training.history.copy()
    if history.empty:
        raise ValueError("GRU training history must contain at least one epoch.")
    comparison = _comparison_frame(result, runtime_seconds, classical_comparison)
    metrics = _metrics_document(
        result,
        source_label=source_label,
        horizon_label=horizon_label,
        runtime_seconds=runtime_seconds,
        source_metadata=source_metadata or {},
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        forecast.to_csv(
            staging_dir / ARTIFACT_NAMES["forecast_csv"],
            index=True,
            index_label=forecast.index.name,
        )
        comparison.to_csv(staging_dir / ARTIFACT_NAMES["comparison_csv"], index=False)
        history.to_csv(staging_dir / ARTIFACT_NAMES["training_history_csv"], index=False)
        _render_png(
            result,
            forecast,
            source_label,
            horizon_label,
            staging_dir / ARTIFACT_NAMES["png"],
        )
        (staging_dir / ARTIFACT_NAMES["metrics_json"]).write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _validate_staged_artifacts(staging_dir, forecast, comparison, history, metrics)

        published = {key: output_dir / name for key, name in ARTIFACT_NAMES.items()}
        _publish_directory(staging_dir, output_dir)
        return published
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def _publish_directory(staging_dir: Path, output_dir: Path) -> None:
    """Replace a complete report directory, restoring the old one on failure."""

    if not output_dir.exists():
        staging_dir.replace(output_dir)
        return
    if not output_dir.is_dir():
        raise ValueError(f"report output path is not a directory: {output_dir}")

    backup_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.backup-", dir=output_dir.parent)
    )
    backup_dir.rmdir()
    output_dir.replace(backup_dir)
    try:
        staging_dir.replace(output_dir)
    except OSError:
        backup_dir.replace(output_dir)
        raise
    else:
        shutil.rmtree(backup_dir)


def _validate_forecast(forecast: pd.DataFrame) -> pd.DataFrame:
    forecast = forecast.copy()
    missing = [column for column in FORECAST_COLUMNS if column not in forecast]
    if missing:
        raise ValueError(f"latest forecast is missing required columns: {', '.join(missing)}")
    if forecast.empty:
        raise ValueError("latest forecast must contain at least one row.")
    values = forecast.loc[:, ("prediction", "p10", "p50", "p90")].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("latest forecast values must be finite.")
    if not np.all(values[:, 1] <= values[:, 2]) or not np.all(values[:, 2] <= values[:, 3]):
        raise ValueError("latest forecast must satisfy p10 <= p50 <= p90.")
    if forecast.index.name is None:
        forecast.index.name = "forecast_timestamp"
    return forecast


def _comparison_frame(
    result: GRUBenchmarkResult,
    runtime_seconds: float,
    classical_comparison: pd.DataFrame | None,
) -> pd.DataFrame:
    configuration = _object_values(result.config)
    gru_row = {
        "model": "GRU",
        "configuration": ", ".join(f"{key}={value}" for key, value in configuration.items()),
        "validation_mae": result.training.validation_metrics["MAE"],
        "validation_rmse": result.training.validation_metrics["RMSE"],
        "test_mae": result.test_metrics["MAE"],
        "test_rmse": result.test_metrics["RMSE"],
        "selected": False,
        "training_seconds": runtime_seconds,
    }
    classical = (
        classical_comparison.copy()
        if classical_comparison is not None
        else pd.DataFrame(columns=COMPARISON_COLUMNS)
    )
    return pd.concat([classical, pd.DataFrame([gru_row])], ignore_index=True, sort=False)


def _metrics_document(
    result: GRUBenchmarkResult,
    *,
    source_label: str,
    horizon_label: str,
    runtime_seconds: float,
    source_metadata: dict[str, object],
) -> dict[str, object]:
    partition_sizes = {
        name: len(partition.targets) for name, partition in result.partitions.items()
    }
    return {
        "artifacts": dict(ARTIFACT_NAMES),
        "best_epoch": int(result.training.best_epoch),
        "config": _object_values(result.config),
        "device": result.training.device,
        "horizon": horizon_label,
        "horizon_steps": int(result.horizon),
        "runtime_seconds": float(runtime_seconds),
        "scaler": {
            "mean": float(result.training.scaler.mean),
            "std": float(result.training.scaler.std),
        },
        "source_label": source_label,
        "source_metadata": _json_value(source_metadata),
        "split_sizes": partition_sizes,
        "test_metrics": _json_value(result.test_metrics),
        "validation_metrics": _json_value(result.training.validation_metrics),
        "validation_residual_quantiles": _json_value(
            result.validation_residual_quantiles
        ),
    }


def _object_values(value: object) -> dict[str, object]:
    if is_dataclass(value):
        return _json_value(asdict(value))
    return _json_value(vars(value))


def _json_value(value: object):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    return value


def _render_png(
    result: GRUBenchmarkResult,
    forecast: pd.DataFrame,
    source_label: str,
    horizon_label: str,
    destination: Path,
) -> None:
    actual = result.partitions["test"].targets.to_numpy(dtype=float)
    prediction = np.asarray(result.test_prediction, dtype=float)
    if actual.shape != prediction.shape or actual.ndim != 2 or actual.shape[0] == 0:
        raise ValueError("test targets and predictions must be aligned non-empty matrices.")

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    test_x = np.arange(actual.shape[0])
    axes[0].plot(test_x, actual[:, 0], label="Actual test load")
    axes[0].plot(test_x, prediction[:, 0], label="GRU prediction")
    axes[0].set_title("Test forecast, first lead")
    axes[0].set_xlabel("Test sample")
    axes[0].set_ylabel("Load")
    axes[0].legend()

    axes[1].plot(forecast.index, forecast["p50"], label="P50 forecast")
    axes[1].fill_between(
        forecast.index,
        forecast["p10"].to_numpy(dtype=float),
        forecast["p90"].to_numpy(dtype=float),
        alpha=0.25,
        label="P10-P90 interval",
    )
    axes[1].set_title("Latest GRU forecast")
    axes[1].set_xlabel("Timestamp")
    axes[1].set_ylabel("Load")
    axes[1].legend()
    figure.suptitle(f"{source_label} {horizon_label} GRU Benchmark")
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def _validate_staged_artifacts(
    staging_dir: Path,
    forecast: pd.DataFrame,
    comparison: pd.DataFrame,
    history: pd.DataFrame,
    metrics: dict[str, object],
) -> None:
    forecast_readback = pd.read_csv(
        staging_dir / ARTIFACT_NAMES["forecast_csv"], index_col=0, parse_dates=[0]
    )
    comparison_readback = pd.read_csv(staging_dir / ARTIFACT_NAMES["comparison_csv"])
    history_readback = pd.read_csv(staging_dir / ARTIFACT_NAMES["training_history_csv"])
    if len(forecast_readback) != len(forecast) or list(forecast_readback.columns) != list(forecast.columns):
        raise ValueError("forecast.csv failed readback validation.")
    if len(comparison_readback) != len(comparison) or list(comparison_readback.columns) != list(comparison.columns):
        raise ValueError("comparison.csv failed readback validation.")
    if len(history_readback) != len(history) or list(history_readback.columns) != list(history.columns):
        raise ValueError("training_history.csv failed readback validation.")
    if json.loads((staging_dir / ARTIFACT_NAMES["metrics_json"]).read_text(encoding="utf-8")) != metrics:
        raise ValueError("metrics.json failed readback validation.")
    image = mpimg.imread(staging_dir / ARTIFACT_NAMES["png"])
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("forecast.png failed readback validation.")
    if any(
        not (staging_dir / filename).is_file()
        or (staging_dir / filename).stat().st_size == 0
        for filename in ARTIFACT_NAMES.values()
    ):
        raise ValueError("all GRU report artifacts must be non-empty before publication.")
