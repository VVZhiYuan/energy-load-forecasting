"""Forecast agent context construction and provider orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from src.ai_config import AISettings
from src.ai_provider import AgentContext, AgentResponse, build_provider
from src.agent_contract import MAX_CONTEXT_ROWS
from src.inference import ForecastRun


SUMMARY_KEYS = (
    "meter",
    "horizon",
    "selected_model",
    "selected_configuration",
    "forecast_origin",
    "observed_start",
    "observed_end",
)
FORECAST_ROW_KEYS = (
    "forecast_timestamp",
    "step",
    "prediction",
    "p10",
    "p50",
    "p90",
    "point_model",
    "interval_method",
)
COMPARISON_ROW_KEYS = (
    "model",
    "configuration",
    "validation_mae",
    "validation_rmse",
    "test_mae",
    "test_rmse",
    "selected",
    "training_seconds",
)
RECENT_LOAD_ROW_KEYS = ("timestamp", "load")


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return value.isoformat()
    if isinstance(value, np.generic):
        item = value.item()
        return None if pd.isna(item) else item
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def _to_records(
    frame: pd.DataFrame, allowed_keys: tuple[str, ...]
) -> list[dict[str, object]]:
    rows = frame.copy().reset_index()
    if rows.columns[0] != "forecast_timestamp":
        rows = rows.rename(columns={rows.columns[0]: "forecast_timestamp"})
    return [
        {key: _json_safe(record[key]) for key in allowed_keys if key in record}
        for record in rows.to_dict(orient="records")
    ]


def _recent_load_rows(run: ForecastRun, recent_points: int) -> list[dict[str, object]]:
    if recent_points == 0:
        return []
    recent = run.observed.iloc[-min(recent_points, MAX_CONTEXT_ROWS) :].copy()
    return [
        {
            "timestamp": timestamp.isoformat(),
            "load": _json_safe(value),
        }
        for timestamp, value in recent.items()
    ]


def build_agent_context_from_frames(
    summary_data: Mapping[str, object],
    forecast_data: pd.DataFrame,
    comparison_data: pd.DataFrame,
    recent_load_rows: list[dict[str, object]] | None = None,
) -> AgentContext:
    """Build the shared Agent contract from in-memory or saved report data."""

    forecast = forecast_data.copy()
    comparison = comparison_data.copy()
    required_columns = {"step", "prediction", "p10", "p90"}
    missing = sorted(required_columns.difference(forecast.columns))
    if missing:
        raise ValueError(
            f"forecast is missing required Agent columns: {', '.join(missing)}"
        )

    if forecast.empty:
        raise ValueError("forecast must contain at least one row")
    if not isinstance(forecast.index, pd.DatetimeIndex):
        raise ValueError("forecast must use a DatetimeIndex")
    if len(forecast) > MAX_CONTEXT_ROWS:
        raise ValueError(f"forecast must contain at most {MAX_CONTEXT_ROWS} rows")

    peak_position = int(forecast["prediction"].astype(float).to_numpy().argmax())
    peak_row = forecast.iloc[peak_position]
    p10 = forecast["p10"].astype(float).to_numpy()
    p90 = forecast["p90"].astype(float).to_numpy()
    interval_width = p90 - p10

    safe_recent = [
        {
            key: _json_safe(row.get(key))
            for key in RECENT_LOAD_ROW_KEYS
        }
        for row in (recent_load_rows or [])[-MAX_CONTEXT_ROWS:]
    ]
    summary = {
        key: _json_safe(summary_data[key])
        for key in SUMMARY_KEYS
        if key in summary_data
    }
    summary.update(
        {
            "selected_model": summary.get("selected_model"),
            "selected_configuration": summary.get("selected_configuration"),
            "horizon": summary.get("horizon"),
            "horizon_steps": int(len(forecast)),
            "forecast_steps": int(len(forecast)),
            "forecast_origin": summary.get("forecast_origin"),
            "observed_start": summary.get("observed_start"),
            "observed_end": summary.get("observed_end"),
            "recent_points": len(safe_recent),
            "peak_step": _json_safe(peak_row["step"]),
            "peak_timestamp": forecast.index[peak_position].isoformat(),
            "peak_prediction": _json_safe(peak_row["prediction"]),
            "interval_width_at_peak": float(interval_width[peak_position]),
            "peak_interval_width": float(interval_width[peak_position]),
            "mean_interval_width": float(interval_width.mean()),
        }
    )

    return AgentContext(
        summary=summary,
        forecast_rows=_to_records(forecast, FORECAST_ROW_KEYS),
        comparison_rows=_to_records(comparison, COMPARISON_ROW_KEYS),
        recent_load_rows=safe_recent,
    )


def build_agent_context(run: ForecastRun, recent_points: int = 96) -> AgentContext:
    if recent_points < 0:
        raise ValueError("recent_points must be non-negative")
    recent_rows = _recent_load_rows(run, recent_points)
    return build_agent_context_from_frames(
        run.summary,
        run.forecast,
        run.model_comparison,
        recent_rows,
    )


def analyze_forecast(
    run: ForecastRun, settings: AISettings | None = None
) -> AgentResponse:
    chosen_settings = settings or AISettings.from_env()
    provider = build_provider(chosen_settings)
    context = build_agent_context(run)
    return provider.analyze(context)
