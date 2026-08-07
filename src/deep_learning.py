"""Leakage-safe sequence preparation for the optional GRU benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral, Real

import numpy as np
import pandas as pd

from src.forecasting import make_multistep_targets, split_supervised_by_time

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


def require_torch() -> None:
    """Raise a concise error when a GRU entry point needs PyTorch."""

    if torch is None:
        raise RuntimeError(
            "PyTorch is required for GRU benchmarking. Install it with: "
            "python -m pip install -r requirements-deep-learning.txt"
        )


@dataclass(frozen=True)
class GRUConfig:
    """Configuration shared by the direct GRU benchmark."""

    context_steps: int = 96
    hidden_size: int = 64
    num_layers: int = 1
    batch_size: int = 256
    epochs: int = 15
    learning_rate: float = 0.001
    patience: int = 3
    seed: int = 42

    def __post_init__(self) -> None:
        integer_fields = (
            "context_steps",
            "hidden_size",
            "num_layers",
            "batch_size",
            "epochs",
            "patience",
            "seed",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")

        if (
            not isinstance(self.learning_rate, Real)
            or isinstance(self.learning_rate, bool)
            or not np.isfinite(float(self.learning_rate))
            or self.learning_rate <= 0
        ):
            raise ValueError("learning_rate must be positive.")
        if self.patience >= self.epochs:
            raise ValueError("patience must be less than epochs.")


@dataclass(frozen=True)
class SequenceWindows:
    """Historical input windows and their complete future targets."""

    inputs: np.ndarray
    targets: pd.DataFrame
    index: pd.DatetimeIndex
    full_index: pd.DatetimeIndex | None = None
    horizon: int | None = None


@dataclass(frozen=True)
class SequencePartition:
    """One chronological partition of sequence windows."""

    inputs: np.ndarray
    targets: pd.DataFrame
    index: pd.DatetimeIndex


@dataclass
class LoadScaler:
    """Standardize load inputs using statistics fitted on training inputs."""

    mean_: float | None = field(default=None, init=False)
    std_: float | None = field(default=None, init=False)

    @property
    def mean(self) -> float:
        self._require_fitted()
        return self.mean_  # type: ignore[return-value]

    @property
    def std(self) -> float:
        self._require_fitted()
        return self.std_  # type: ignore[return-value]

    def fit(self, inputs: np.ndarray) -> LoadScaler:
        values = _as_finite_float_array(inputs, "inputs")
        if values.size == 0:
            raise ValueError("inputs must not be empty.")

        self.mean_ = float(np.mean(values, dtype=np.float64))
        standard_deviation = float(np.std(values, dtype=np.float64))
        self.std_ = standard_deviation if standard_deviation > 0 else 1.0
        return self

    def transform(self, inputs: np.ndarray) -> np.ndarray:
        self._require_fitted()
        values = _as_finite_float_array(inputs, "inputs")
        return ((values - self.mean) / self.std).astype(np.float32, copy=False)

    def inverse_transform(self, inputs: np.ndarray) -> np.ndarray:
        self._require_fitted()
        values = _as_finite_float_array(inputs, "inputs")
        return (values * self.std + self.mean).astype(np.float32, copy=False)

    def _require_fitted(self) -> None:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("LoadScaler must be fitted before transforming values.")


def make_sequence_windows(
    series: pd.Series,
    horizon: int,
    context_steps: int = 96,
) -> SequenceWindows:
    """Create one load-only historical window for every complete target."""

    if not isinstance(series, pd.Series):
        raise ValueError("series must be a pandas Series.")
    context_steps = _positive_integer(context_steps, "context_steps")
    horizon = _positive_integer(horizon, "horizon")

    try:
        values = series.to_numpy(dtype=np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("series values must be numeric.") from exc
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("series values must be finite and non-empty.")

    targets = make_multistep_targets(series, horizon).dropna()
    first_origin_position = context_steps - 1
    if len(series) < context_steps + horizon or len(targets) <= first_origin_position:
        raise ValueError("series is too short for the requested context and horizon.")

    targets = targets.iloc[first_origin_position:].copy()
    origin_positions = series.index.get_indexer(targets.index)
    if np.any(origin_positions < first_origin_position):
        raise ValueError("series does not contain complete historical contexts.")

    offsets = np.arange(context_steps, dtype=np.int64)
    starts = origin_positions - first_origin_position
    inputs = values[starts[:, None] + offsets[None, :], None]
    inputs = np.asarray(inputs, dtype=np.float32)

    return SequenceWindows(
        inputs=inputs,
        targets=targets,
        index=targets.index.copy(),
        full_index=series.index.copy(),
        horizon=horizon,
    )


def split_sequence_windows(
    windows: SequenceWindows,
    full_index: pd.DatetimeIndex | None = None,
    horizon: int | None = None,
    train_size: float = 0.7,
    val_size: float = 0.15,
) -> dict[str, SequencePartition]:
    """Split windows chronologically while keeping every target in its split."""

    if not isinstance(windows, SequenceWindows):
        raise ValueError("windows must be a SequenceWindows instance.")
    if len(windows.inputs) != len(windows.targets) or len(windows.index) != len(windows.targets):
        raise ValueError("windows inputs, targets, and index must be aligned.")
    if not windows.targets.index.equals(windows.index):
        raise ValueError("windows targets and index must be aligned.")

    source_index = full_index if full_index is not None else windows.full_index
    if source_index is None:
        raise ValueError("full_index is required when windows has no source index.")
    if not isinstance(source_index, pd.DatetimeIndex):
        raise ValueError("full_index must be a DatetimeIndex.")

    if horizon is not None:
        target_horizon = _positive_integer(horizon, "horizon")
        if (
            target_horizon != windows.targets.shape[1]
            or (windows.horizon is not None and target_horizon != windows.horizon)
        ):
            raise ValueError("horizon must match the sequence windows targets.")
    else:
        target_horizon = windows.horizon
        if target_horizon is None:
            target_horizon = windows.targets.shape[1]
        target_horizon = _positive_integer(target_horizon, "horizon")

    aligned_features = pd.DataFrame(
        {"window_row": np.arange(len(windows.index), dtype=np.int64)},
        index=windows.index,
    )
    splits = split_supervised_by_time(
        aligned_features,
        windows.targets,
        full_index=source_index,
        horizon=target_horizon,
        train_size=train_size,
        val_size=val_size,
    )

    output: dict[str, SequencePartition] = {}
    for name, (partition_features, partition_targets) in splits.items():
        row_positions = windows.index.get_indexer(partition_features.index)
        if np.any(row_positions < 0):
            raise ValueError("every partition origin must exist in windows.index.")
        output[name] = SequencePartition(
            inputs=windows.inputs[row_positions].copy(),
            targets=partition_targets.copy(),
            index=partition_targets.index.copy(),
        )
    return output


def _positive_integer(value: int, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _as_finite_float_array(values: np.ndarray, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values.")
    return array
