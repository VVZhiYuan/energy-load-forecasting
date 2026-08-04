"""Transparent baseline models for multi-step load forecasting."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate import mae


def naive_forecast(current_load, horizon: int) -> np.ndarray:
    """Repeat the latest observed load across the forecast horizon."""

    if horizon < 1:
        raise ValueError("horizon must be at least 1.")
    values = np.asarray(current_load, dtype=float).reshape(-1, 1)
    return np.repeat(values, horizon, axis=1)


def seasonal_naive_forecast(
    series: pd.Series,
    origins: pd.DatetimeIndex,
    horizon: int,
    season_length: int = 96,
) -> np.ndarray:
    """Use the previous day's values aligned to every future step."""

    if not 1 <= horizon <= season_length:
        raise ValueError("horizon must be between 1 and season_length.")
    origin_positions = series.index.get_indexer(origins)
    if np.any(origin_positions < 0):
        raise ValueError("every origin must exist in the source series.")

    steps = np.arange(1, horizon + 1)
    seasonal_positions = origin_positions[:, None] + steps[None, :] - season_length
    if seasonal_positions.min() < 0:
        raise ValueError("origins do not have a complete previous-day season.")
    if np.any(seasonal_positions > origin_positions[:, None]):
        raise ValueError("seasonal forecast attempted to use future observations.")
    return series.to_numpy(dtype=float)[seasonal_positions]


def select_ridge_alpha(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    alphas: Iterable[float] = (0.1, 1.0, 10.0, 100.0),
) -> tuple[Pipeline, pd.DataFrame]:
    """Select a direct multi-output Ridge pipeline by validation MAE."""

    candidates = sorted(set(float(alpha) for alpha in alphas))
    if not candidates or candidates[0] <= 0:
        raise ValueError("alphas must contain positive values.")

    best_model = None
    best_score = np.inf
    rows = []
    for alpha in candidates:
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )
        model.fit(X_train, y_train)
        score = mae(y_val, model.predict(X_val))
        rows.append({"alpha": alpha, "validation_MAE": score})
        if score < best_score:
            best_model = model
            best_score = score

    return best_model, pd.DataFrame(rows)
