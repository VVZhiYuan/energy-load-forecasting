"""Cached, validated readers for dashboard report artifacts."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
import pandas as pd

from src.config import REPORTS_DIR


class DashboardReportError(RuntimeError):
    """Raised when a dashboard report artifact cannot be read safely."""


F = TypeVar("F", bound=Callable[..., Any])


def _cache_data(func: F) -> F:
    """Apply Streamlit caching only when Streamlit is available."""

    try:
        import streamlit as st
    except ImportError:
        return func
    return st.cache_data(func)


def _report_error(path: Path, exc: Exception) -> DashboardReportError:
    return DashboardReportError(f"Unable to load dashboard artifact {path.name}: {exc}")


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, TypeError) as exc:
        raise _report_error(path, exc) from exc


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        raise _report_error(path, exc) from exc
    if not isinstance(value, dict):
        raise _report_error(path, ValueError("JSON document must be an object"))
    return value


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], path: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise _report_error(path, ValueError(f"missing required columns: {', '.join(missing)}"))
    if frame.empty:
        raise _report_error(path, ValueError("report must contain at least one row"))


def _require_finite_numbers(
    frame: pd.DataFrame, columns: tuple[str, ...], path: Path
) -> None:
    try:
        values = frame.loc[:, columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise _report_error(path, exc) from exc
    if not np.isfinite(values).all():
        raise _report_error(path, ValueError("numeric values must be finite"))
    frame.loc[:, columns] = values


def _timestamp_index(frame: pd.DataFrame, column: str, path: Path) -> pd.DataFrame:
    try:
        timestamps = pd.to_datetime(frame[column], errors="raise")
    except (ValueError, TypeError) as exc:
        raise _report_error(path, exc) from exc
    if timestamps.isna().any():
        raise _report_error(path, ValueError(f"{column} must not contain null timestamps"))
    indexed = frame.drop(columns=column).copy()
    indexed.index = pd.DatetimeIndex(timestamps, name=column)
    return indexed


def _forecast_dir(reports_dir: Path, meter: str, horizon: str) -> Path:
    return reports_dir / "predictions" / meter / horizon


@_cache_data
def _load_forecast_report(reports_dir: Path, meter: str, horizon: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    directory = _forecast_dir(reports_dir, meter, horizon)
    forecast_path = directory / "forecast.csv"
    forecast = _read_csv(forecast_path)
    required = ("forecast_timestamp", "step", "prediction", "p10", "p50", "p90")
    _require_columns(forecast, required, forecast_path)
    _require_finite_numbers(forecast, ("step", "prediction", "p10", "p50", "p90"), forecast_path)
    if not (forecast["p10"] <= forecast["p50"]).all() or not (
        forecast["p50"] <= forecast["p90"]
    ).all():
        raise _report_error(forecast_path, ValueError("intervals must satisfy p10 <= p50 <= p90"))
    return _timestamp_index(forecast, "forecast_timestamp", forecast_path), _read_json_object(
        directory / "summary.json"
    )


def load_forecast_report(
    meter: str, horizon: str, *, reports_dir: str | Path = REPORTS_DIR
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load one classical forecast trajectory and its summary metadata."""

    forecast, summary = _load_forecast_report(Path(reports_dir), meter, horizon)
    return forecast.copy(), deepcopy(summary)


@_cache_data
def _load_model_comparison(reports_dir: Path, meter: str, horizon: str) -> pd.DataFrame:
    path = _forecast_dir(reports_dir, meter, horizon) / "model_comparison.csv"
    comparison = _read_csv(path)
    required = (
        "model",
        "configuration",
        "validation_mae",
        "validation_rmse",
        "test_mae",
        "test_rmse",
        "selected",
        "training_seconds",
    )
    _require_columns(comparison, required, path)
    _require_finite_numbers(
        comparison,
        ("validation_mae", "validation_rmse", "test_mae", "test_rmse", "training_seconds"),
        path,
    )
    return comparison


def load_model_comparison(meter: str, horizon: str, *, reports_dir: str | Path = REPORTS_DIR) -> pd.DataFrame:
    """Load the classical model comparison table for one report horizon."""

    return _load_model_comparison(Path(reports_dir), meter, horizon).copy()


@_cache_data
def _load_robustness_report(reports_dir: Path, meter: str, horizon: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    directory = reports_dir / "robustness" / meter / horizon
    metrics_path = directory / "robustness_metrics.csv"
    metrics = _read_csv(metrics_path)
    required = (
        "scenario",
        "scenario_kind",
        "forecast_origin",
        "horizon",
        "horizon_steps",
        "selected_model",
        "affected_points",
        "imputed_points",
        "mae",
        "rmse",
        "mape",
        "clean_mae",
        "mae_delta",
        "mae_degradation_pct",
    )
    numeric = (
        "horizon_steps",
        "affected_points",
        "imputed_points",
        "mae",
        "rmse",
        "mape",
        "clean_mae",
        "mae_delta",
        "mae_degradation_pct",
    )
    _require_columns(metrics, required, metrics_path)
    _require_finite_numbers(metrics, numeric, metrics_path)
    try:
        metrics["forecast_origin"] = pd.to_datetime(metrics["forecast_origin"], errors="raise")
    except (ValueError, TypeError) as exc:
        raise _report_error(metrics_path, exc) from exc
    return metrics, _read_json_object(directory / "robustness_summary.json")


def load_robustness_report(
    meter: str, horizon: str, *, reports_dir: str | Path = REPORTS_DIR
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load robustness scenario metrics and their metadata."""

    metrics, summary = _load_robustness_report(Path(reports_dir), meter, horizon)
    return metrics.copy(), deepcopy(summary)


@_cache_data
def _load_storage_report(
    reports_dir: Path, meter: str, horizon: str, scenario: str, strategy: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    directory = reports_dir / "optimization" / meter / horizon
    dispatch_path = directory / "dispatch.csv"
    dispatch = _read_csv(dispatch_path)
    required = (
        "forecast_timestamp",
        "scenario",
        "strategy",
        "forecast_load_kw",
        "grid_import_kw",
        "charge_kw",
        "discharge_kw",
        "battery_energy_kwh",
        "soc",
        "tariff_period",
        "energy_price",
        "interval_energy_cost",
    )
    numeric = (
        "forecast_load_kw",
        "grid_import_kw",
        "charge_kw",
        "discharge_kw",
        "battery_energy_kwh",
        "soc",
        "energy_price",
        "interval_energy_cost",
    )
    _require_columns(dispatch, required, dispatch_path)
    _require_finite_numbers(dispatch, numeric, dispatch_path)
    selected = dispatch.loc[
        (dispatch["scenario"] == scenario) & (dispatch["strategy"] == strategy)
    ]
    if selected.empty:
        raise _report_error(
            dispatch_path,
            ValueError(f"missing dispatch rows for scenario={scenario}, strategy={strategy}"),
        )
    return _timestamp_index(selected, "forecast_timestamp", dispatch_path), _read_json_object(
        directory / "optimization_summary.json"
    )


def load_storage_report(
    meter: str,
    horizon: str,
    *,
    scenario: str = "p50",
    strategy: str = "optimized",
    reports_dir: str | Path = REPORTS_DIR,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load one storage dispatch scenario and strategy with its summary."""

    dispatch, summary = _load_storage_report(
        Path(reports_dir), meter, horizon, scenario, strategy
    )
    return dispatch.copy(), deepcopy(summary)


@_cache_data
def _load_gru_metrics(reports_dir: Path, meter: str, horizon: str) -> dict[str, Any]:
    path = reports_dir / "deep_learning" / meter / horizon / "metrics.json"
    metrics = _read_json_object(path)
    for section in ("validation_metrics", "test_metrics"):
        section_metrics = metrics.get(section)
        if not isinstance(section_metrics, dict):
            raise _report_error(path, ValueError(f"missing required object: {section}"))
        for metric in ("MAE", "RMSE"):
            value = section_metrics.get(metric)
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise _report_error(path, ValueError(f"{section}.{metric} must be numeric")) from exc
            if not np.isfinite(numeric_value):
                raise _report_error(path, ValueError(f"{section}.{metric} must be finite"))
    return metrics


def load_gru_metrics(meter: str, horizon: str, *, reports_dir: str | Path = REPORTS_DIR) -> dict[str, Any]:
    """Load validated GRU benchmark metrics metadata."""

    return deepcopy(_load_gru_metrics(Path(reports_dir), meter, horizon))
