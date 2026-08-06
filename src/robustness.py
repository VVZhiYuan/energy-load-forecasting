"""Deterministic data-quality and distribution-shift scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
import pandas as pd


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

