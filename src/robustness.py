"""Deterministic data-quality and distribution-shift scenarios."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from math import ceil

import numpy as np
import pandas as pd

from src.data_loader import LoadedLoadSeries
from src.evaluate import evaluate_forecast
from src.inference import HORIZON_CONFIG, run_latest_forecast


@dataclass(frozen=True)
class RobustnessScenario:
    """One reproducible perturbation applied before a forecast origin."""

    name: str
    kind: str
    parameters: dict[str, float | int]


@dataclass(frozen=True)
class PerturbationResult:
    """Perturbed observations plus audit counts."""

    series: pd.Series
    affected_points: int
    imputed_points: int


def scenario_catalog() -> tuple[RobustnessScenario, ...]:
    return (
        RobustnessScenario("clean", "clean", {}),
        RobustnessScenario(
            "sensor_noise_5pct",
            "sensor_noise",
            {"std_fraction": 0.05},
        ),
        RobustnessScenario(
            "missing_blocks_1pct",
            "missing_blocks",
            {"fraction": 0.01, "block_length": 4},
        ),
        RobustnessScenario(
            "spikes_1pct",
            "spikes",
            {"fraction": 0.01, "std_multiplier": 3.0},
        ),
        RobustnessScenario(
            "distribution_shift_10pct",
            "distribution_shift",
            {"tail_fraction": 0.20, "shift_fraction": 0.10},
        ),
    )


def _validate_series(series: pd.Series) -> pd.Series:
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("robustness scenarios require a DatetimeIndex")
    if series.empty or series.index.hasnans or not series.index.is_unique:
        raise ValueError("robustness input must have a non-empty unique timestamp index")
    values = pd.to_numeric(series, errors="coerce").astype(float)
    if values.isna().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("robustness input must contain only finite numeric values")
    values.name = series.name
    return values


def _missing_mask(length: int, fraction: float, block_length: int, rng) -> np.ndarray:
    requested = max(1, ceil(length * fraction))
    block_length = max(1, min(length, block_length))
    block_count = max(1, ceil(requested / block_length))
    possible_starts = np.arange(length - block_length + 1)
    starts = rng.choice(
        possible_starts,
        size=min(block_count, len(possible_starts)),
        replace=False,
    )
    mask = np.zeros(length, dtype=bool)
    for start in np.atleast_1d(starts):
        mask[start : start + block_length] = True
    return mask


def apply_scenario(
    series: pd.Series,
    scenario: RobustnessScenario,
    seed: int,
) -> PerturbationResult:
    """Apply one scenario without modifying the input series."""

    clean = _validate_series(series)
    if scenario.kind == "clean":
        return PerturbationResult(clean.copy(deep=True), 0, 0)

    rng = np.random.default_rng(seed)
    values = clean.to_numpy(dtype=float, copy=True)
    scale = max(float(clean.std(ddof=0)), 1e-9)
    affected = 0
    imputed = 0

    if scenario.kind == "sensor_noise":
        noise_std = scale * float(scenario.parameters["std_fraction"])
        values += rng.normal(0.0, noise_std, size=len(values))
        affected = len(values)
    elif scenario.kind == "missing_blocks":
        mask = _missing_mask(
            len(values),
            float(scenario.parameters["fraction"]),
            int(scenario.parameters["block_length"]),
            rng,
        )
        values[mask] = np.nan
        affected = int(mask.sum())
        repaired = pd.Series(values, index=clean.index).interpolate(
            method="time", limit_direction="both"
        )
        values = repaired.to_numpy(dtype=float)
        imputed = affected
    elif scenario.kind == "spikes":
        count = max(1, ceil(len(values) * float(scenario.parameters["fraction"])))
        indices = rng.choice(len(values), size=min(count, len(values)), replace=False)
        values[indices] += scale * float(scenario.parameters["std_multiplier"])
        affected = len(indices)
    elif scenario.kind == "distribution_shift":
        tail = max(1, ceil(len(values) * float(scenario.parameters["tail_fraction"])))
        values[-tail:] *= 1.0 + float(scenario.parameters["shift_fraction"])
        affected = tail
    else:
        raise ValueError(f"unsupported robustness scenario kind: {scenario.kind}")

    values = np.maximum(values, 0.0)
    perturbed = pd.Series(values, index=clean.index, name=clean.name)
    if not np.isfinite(perturbed.to_numpy()).all():
        raise ValueError("robustness scenario produced non-finite values")
    return PerturbationResult(perturbed, affected, imputed)


def run_robustness_experiment(
    loaded: LoadedLoadSeries,
    horizon_label: str,
    holiday_country: str | None,
    scenarios: Sequence[str] | None = None,
    seed: int = 42,
    search: bool = False,
    parallel_jobs: int = -1,
    progress: Callable[[str], object] | None = None,
) -> pd.DataFrame:
    """Evaluate forecast degradation against an untouched historical horizon."""

    try:
        horizon = HORIZON_CONFIG[horizon_label].target_horizon_steps
    except KeyError as exc:
        raise ValueError("horizon_label must be '1h' or '24h'.") from exc

    origin_position = len(loaded.series) - horizon - 1
    if origin_position < 0:
        raise ValueError("series is too short for the requested robustness horizon")

    catalog = {scenario.name: scenario for scenario in scenario_catalog()}
    requested_names = list(scenarios) if scenarios is not None else list(catalog)
    unknown = [name for name in requested_names if name not in catalog]
    if unknown:
        raise ValueError(f"unknown robustness scenario: {', '.join(unknown)}")
    ordered_names = ["clean"] + [
        name for name in requested_names if name != "clean"
    ]
    observed = loaded.series.iloc[: origin_position + 1].copy()
    future = loaded.series.iloc[origin_position + 1 : origin_position + 1 + horizon]
    if len(future) != horizon:
        raise ValueError("historical future horizon is incomplete")

    rows: list[dict[str, object]] = []
    for scenario_name in ordered_names:
        scenario = catalog[scenario_name]
        if progress is not None:
            progress(f"Evaluating robustness scenario: {scenario.name}")
        perturbation = apply_scenario(observed, scenario, seed=seed)
        scenario_loaded = replace(
            loaded,
            series=perturbation.series,
            negative_load_count=int((perturbation.series < 0).sum()),
        )
        run = run_latest_forecast(
            scenario_loaded,
            horizon_label=horizon_label,
            holiday_country=holiday_country,
            search=search,
            parallel_jobs=parallel_jobs,
        )
        prediction = np.asarray(run.forecast["prediction"], dtype=float)
        if prediction.shape != (horizon,) or not np.isfinite(prediction).all():
            raise ValueError(
                f"scenario {scenario.name} produced an invalid point forecast"
            )
        metrics = evaluate_forecast(future.to_numpy(dtype=float), prediction)
        rows.append(
            {
                "scenario": scenario.name,
                "scenario_kind": scenario.kind,
                "forecast_origin": observed.index[-1].isoformat(),
                "horizon": horizon_label,
                "horizon_steps": horizon,
                "selected_model": run.summary.get("selected_model"),
                "affected_points": perturbation.affected_points,
                "imputed_points": perturbation.imputed_points,
                "mae": metrics["MAE"],
                "rmse": metrics["RMSE"],
                "mape": metrics["MAPE"],
            }
        )

    result = pd.DataFrame(rows)
    clean_mae = float(result.loc[result["scenario"].eq("clean"), "mae"].iloc[0])
    result["clean_mae"] = clean_mae
    result["mae_delta"] = result["mae"] - clean_mae
    if clean_mae == 0.0:
        result["mae_degradation_pct"] = 0.0
    else:
        result["mae_degradation_pct"] = result["mae_delta"] / clean_mae * 100.0
    return result
