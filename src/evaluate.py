"""Evaluation metrics for forecasting models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def evaluate_forecast(y_true, y_pred) -> dict[str, float]:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }


def _validated_forecast_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.ndim != 2 or pred.ndim != 2 or true.shape != pred.shape:
        raise ValueError("y_true and y_pred must be 2D arrays with identical shapes.")
    return true, pred


def evaluate_multistep(y_true, y_pred) -> dict[str, float]:
    """Summarize a complete multi-step trajectory forecast."""

    true, pred = _validated_forecast_arrays(y_true, y_pred)
    metrics = evaluate_forecast(true, pred)
    metrics["Endpoint_MAE"] = mae(true[:, -1], pred[:, -1])
    return metrics


def per_step_metrics(y_true, y_pred) -> pd.DataFrame:
    """Return one metric row for each future 15-minute step."""

    true, pred = _validated_forecast_arrays(y_true, y_pred)
    rows = []
    for index in range(true.shape[1]):
        metrics = evaluate_forecast(true[:, index], pred[:, index])
        rows.append(
            {
                "forecast_step": index + 1,
                "lead_minutes": (index + 1) * 15,
                **metrics,
            }
        )
    return pd.DataFrame(rows)
