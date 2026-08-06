"""Leakage-safe model selection and latest load forecasting."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from src.baselines import naive_forecast, seasonal_naive_forecast, select_ridge_alpha
from src.config import ONE_HOUR_CONFIG, TWENTY_FOUR_HOUR_CONFIG
from src.data_loader import LoadedLoadSeries
from src.evaluate import mae, rmse
from src.features import build_baseline_features
from src.forecasting import make_multistep_targets, split_supervised_by_time
from src.ml_models import (
    DEFAULT_LIGHTGBM_CANDIDATES,
    fit_direct_lightgbm,
    fit_direct_quantile_lightgbm,
    refit_direct_lightgbm,
    select_lightgbm_candidate,
)


MODEL_ORDER = {"Naive": 0, "Seasonal Naive": 1, "Ridge": 2, "LightGBM": 3}
STANDARD_CANDIDATE = {
    "1h": next(
        candidate
        for candidate in DEFAULT_LIGHTGBM_CANDIDATES
        if candidate.name == "medium"
    ),
    "24h": next(
        candidate
        for candidate in DEFAULT_LIGHTGBM_CANDIDATES
        if candidate.name == "small"
    ),
}
HORIZON_CONFIG = {
    ONE_HOUR_CONFIG.forecast_label: ONE_HOUR_CONFIG,
    TWENTY_FOUR_HOUR_CONFIG.forecast_label: TWENTY_FOUR_HOUR_CONFIG,
}
QUANTILES = (0.1, 0.5, 0.9)


@dataclass(frozen=True)
class ForecastRun:
    observed: pd.Series
    forecast: pd.DataFrame
    model_comparison: pd.DataFrame
    summary: dict[str, object]


def ordered_quantiles(values: np.ndarray) -> tuple[np.ndarray, int]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("quantile values must be a finite n-by-3 array.")
    crossing = np.any(np.diff(values, axis=1) < 0, axis=1)
    return np.sort(values, axis=1), int(crossing.sum())


def _score_row(
    model,
    configuration,
    y_val,
    val_prediction,
    y_test,
    test_prediction,
    seconds,
):
    return {
        "model": model,
        "configuration": configuration,
        "validation_mae": mae(y_val, val_prediction),
        "validation_rmse": rmse(y_val, val_prediction),
        "test_mae": mae(y_test, test_prediction),
        "test_rmse": rmse(y_test, test_prediction),
        "selected": False,
        "training_seconds": float(seconds),
    }


def _select_row(table: pd.DataFrame) -> pd.DataFrame:
    ranked = table.assign(_order=table["model"].map(MODEL_ORDER)).sort_values(
        ["validation_mae", "_order"], kind="stable"
    )
    selected_index = ranked.index[0]
    result = table.copy()
    result["selected"] = result.index == selected_index
    return result


def _notify(progress: Callable[[str], object] | None, message: str) -> None:
    if progress is not None:
        progress(message)


def run_latest_forecast(
    loaded: LoadedLoadSeries,
    horizon_label: str,
    holiday_country: str | None,
    search: bool = False,
    parallel_jobs: int = -1,
    progress: Callable[[str], object] | None = None,
) -> ForecastRun:
    """Compare model families, refit the validation winner, and forecast ahead."""

    started = perf_counter()
    try:
        config = HORIZON_CONFIG[horizon_label]
    except KeyError as exc:
        raise ValueError("horizon_label must be '1h' or '24h'.") from exc
    horizon = config.target_horizon_steps

    features = build_baseline_features(loaded.series, country=holiday_country)
    targets = make_multistep_targets(loaded.series, horizon)
    complete_targets = targets.dropna()
    all_index = features.index.intersection(complete_targets.index)
    X_all = features.loc[all_index]
    y_all = complete_targets.loc[all_index]
    splits = split_supervised_by_time(
        X_all,
        y_all,
        full_index=loaded.series.index,
        horizon=horizon,
    )
    X_train, y_train = splits["train"]
    X_validation, y_validation = splits["validation"]
    X_test, y_test = splits["test"]

    rows = []
    validation_predictions = {}

    _notify(progress, "Evaluating Naive")
    naive_validation = naive_forecast(
        loaded.series.loc[X_validation.index], horizon
    )
    naive_test = naive_forecast(loaded.series.loc[X_test.index], horizon)
    validation_predictions["Naive"] = naive_validation
    rows.append(
        _score_row(
            "Naive",
            "persistence",
            y_validation,
            naive_validation,
            y_test,
            naive_test,
            0.0,
        )
    )

    _notify(progress, "Evaluating Seasonal Naive")
    seasonal_validation = seasonal_naive_forecast(
        loaded.series, X_validation.index, horizon
    )
    seasonal_test = seasonal_naive_forecast(loaded.series, X_test.index, horizon)
    validation_predictions["Seasonal Naive"] = seasonal_validation
    rows.append(
        _score_row(
            "Seasonal Naive",
            "season_length=96",
            y_validation,
            seasonal_validation,
            y_test,
            seasonal_test,
            0.0,
        )
    )

    _notify(progress, "Evaluating Ridge")
    family_started = perf_counter()
    ridge_model, _ = select_ridge_alpha(
        X_train, y_train, X_validation, y_validation
    )
    ridge_seconds = perf_counter() - family_started
    ridge_validation = ridge_model.predict(X_validation)
    ridge_test = ridge_model.predict(X_test)
    validation_predictions["Ridge"] = ridge_validation
    ridge_alpha = float(ridge_model.named_steps["ridge"].alpha)
    rows.append(
        _score_row(
            "Ridge",
            f"alpha={ridge_alpha:g}",
            y_validation,
            ridge_validation,
            y_test,
            ridge_test,
            ridge_seconds,
        )
    )

    _notify(progress, "Evaluating LightGBM")
    family_started = perf_counter()
    if search:
        lightgbm_model, search_results = select_lightgbm_candidate(
            X_train,
            y_train,
            X_validation,
            y_validation,
            candidates=DEFAULT_LIGHTGBM_CANDIDATES,
            parallel_jobs=parallel_jobs,
        )
    else:
        lightgbm_model = fit_direct_lightgbm(
            X_train,
            y_train,
            X_validation,
            y_validation,
            STANDARD_CANDIDATE[horizon_label],
            parallel_jobs=parallel_jobs,
        )
        search_results = pd.DataFrame()
    lightgbm_seconds = perf_counter() - family_started
    lightgbm_validation = lightgbm_model.predict(X_validation)
    lightgbm_test = lightgbm_model.predict(X_test)
    validation_predictions["LightGBM"] = lightgbm_validation
    rows.append(
        _score_row(
            "LightGBM",
            lightgbm_model.candidate.name,
            y_validation,
            lightgbm_validation,
            y_test,
            lightgbm_test,
            lightgbm_seconds,
        )
    )

    comparison = _select_row(pd.DataFrame(rows))
    selected_row = comparison.loc[comparison["selected"]].iloc[0]
    winner = str(selected_row["model"])
    selected_configuration = str(selected_row["configuration"])
    latest_X = features.iloc[[-1]]

    quantile_models = None
    if winner == "LightGBM":
        _notify(progress, "Fitting LightGBM quantile intervals")
        quantile_models = fit_direct_quantile_lightgbm(
            X_train,
            y_train,
            X_validation,
            y_validation,
            lightgbm_model.candidate,
            quantiles=QUANTILES,
            parallel_jobs=parallel_jobs,
        )

    _notify(progress, "Refitting selected model on all labeled origins")
    if winner == "Naive":
        prediction = naive_forecast([loaded.series.iloc[-1]], horizon)[0]
    elif winner == "Seasonal Naive":
        prediction = seasonal_naive_forecast(
            loaded.series,
            pd.DatetimeIndex([loaded.series.index[-1]]),
            horizon,
        )[0]
    elif winner == "Ridge":
        ridge_model.fit(X_all, y_all)
        prediction = ridge_model.predict(latest_X)[0]
    else:
        point_model = refit_direct_lightgbm(
            X_all, y_all, lightgbm_model, parallel_jobs
        )
        prediction = point_model.predict(latest_X)[0]

    if winner == "LightGBM":
        refitted_quantiles = {
            quantile: refit_direct_lightgbm(
                X_all, y_all, quantile_models[quantile], parallel_jobs
            )
            for quantile in QUANTILES
        }
        quantile_values = np.column_stack(
            [
                refitted_quantiles[quantile].predict(latest_X)[0]
                for quantile in QUANTILES
            ]
        )
        interval_method = "lightgbm_quantile"
    else:
        residuals = (
            y_validation.to_numpy()
            - np.asarray(validation_predictions[winner], dtype=float)
        )
        residual_quantiles = np.quantile(residuals, QUANTILES, axis=0).T
        quantile_values = np.asarray(prediction, dtype=float)[:, None] + residual_quantiles
        interval_method = "residual_calibration"

    prediction = np.asarray(prediction, dtype=float)
    if prediction.shape != (horizon,) or not np.isfinite(prediction).all():
        raise ValueError("point forecast must contain one finite value per horizon step.")
    quantile_values, correction_count = ordered_quantiles(quantile_values)

    future_index = pd.date_range(
        loaded.series.index[-1] + pd.Timedelta(minutes=15),
        periods=horizon,
        freq="15min",
        name="forecast_timestamp",
    )
    forecast = pd.DataFrame(
        {
            "step": np.arange(1, horizon + 1),
            "prediction": prediction,
            "p10": quantile_values[:, 0],
            "p50": quantile_values[:, 1],
            "p90": quantile_values[:, 2],
            "point_model": winner,
            "interval_method": interval_method,
        },
        index=future_index,
    )

    summary = {
        "source_label": loaded.source_label,
        "input_format": loaded.input_format,
        "meter": loaded.meter,
        "observed_start": loaded.series.index[0].isoformat(),
        "observed_end": loaded.series.index[-1].isoformat(),
        "forecast_origin": loaded.series.index[-1].isoformat(),
        "horizon": horizon_label,
        "horizon_steps": horizon,
        "holiday_country": holiday_country,
        "search": bool(search),
        "selected_model": winner,
        "selected_configuration": selected_configuration,
        "split_rows": {
            "train": len(X_train),
            "validation": len(X_validation),
            "test": len(X_test),
            "all_labeled": len(X_all),
        },
        "quantile_correction_count": correction_count,
        "interval_method": interval_method,
        "negative_load_count": int(loaded.negative_load_count),
        "random_seed": 42,
        "runtime_seconds": float(perf_counter() - started),
        "lightgbm_search": search_results.to_dict(orient="records"),
    }
    return ForecastRun(
        observed=loaded.series.copy(),
        forecast=forecast,
        model_comparison=comparison,
        summary=summary,
    )
