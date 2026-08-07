"""Leakage-safe sequence preparation for the optional GRU benchmark."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from numbers import Integral, Real
import os
import random

import numpy as np
import pandas as pd

from src.evaluate import evaluate_multistep
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


if nn is not None:

    class DirectGRU(nn.Module):
        """A GRU encoder with one direct multi-step trajectory head."""

        def __init__(self, horizon: int, config: GRUConfig) -> None:
            super().__init__()
            self.horizon = _positive_integer(horizon, "horizon")
            self.encoder = nn.GRU(
                input_size=1,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                batch_first=True,
            )
            self.head = nn.Linear(config.hidden_size, self.horizon)

        def forward(self, inputs):
            encoded, _ = self.encoder(inputs)
            return self.head(encoded[:, -1, :])

else:  # pragma: no cover - exercised only in environments without PyTorch

    class DirectGRU:
        """Dependency-safe stand-in that fails only when instantiated."""

        def __init__(self, horizon: int, config: GRUConfig) -> None:
            require_torch()


@dataclass(frozen=True)
class GRUTrainingResult:
    """The validation-selected direct GRU and its training evidence."""

    model: DirectGRU
    scaler: LoadScaler
    best_epoch: int
    validation_prediction: np.ndarray
    validation_metrics: dict[str, float]
    history: pd.DataFrame
    device: str


@dataclass(frozen=True)
class GRUBenchmarkResult:
    """Untouched-test metrics and calibrated latest forecast for one horizon."""

    config: GRUConfig
    horizon: int
    partitions: dict[str, SequencePartition]
    training: GRUTrainingResult
    test_prediction: np.ndarray
    test_metrics: dict[str, float]
    latest_forecast: pd.DataFrame
    validation_residual_quantiles: np.ndarray


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


def fit_direct_gru(
    train: SequencePartition,
    validation: SequencePartition,
    config: GRUConfig,
) -> GRUTrainingResult:
    """Train a direct GRU and restore the lowest-validation-MAE checkpoint."""

    require_torch()
    _validate_partition(train, "train", config.context_steps)
    _validate_partition(validation, "validation", config.context_steps)
    if train.targets.shape[1] != validation.targets.shape[1]:
        raise ValueError("train and validation targets must use the same horizon.")

    _set_torch_seed(config.seed)
    device = _torch_device()
    horizon = train.targets.shape[1]
    scaler = LoadScaler().fit(train.inputs)
    train_inputs = torch.as_tensor(scaler.transform(train.inputs), dtype=torch.float32)
    train_targets = torch.as_tensor(
        _normalize_targets(train.targets.to_numpy(dtype=np.float32), scaler),
        dtype=torch.float32,
    )
    validation_inputs = torch.as_tensor(
        scaler.transform(validation.inputs), dtype=torch.float32, device=device
    )
    validation_targets = validation.targets.to_numpy(dtype=np.float32)

    dataset = torch.utils.data.TensorDataset(train_inputs, train_targets)
    generator = torch.Generator().manual_seed(config.seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    model = DirectGRU(horizon, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.learning_rate))
    loss_function = nn.MSELoss()

    best_mae = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for batch_inputs, batch_targets in loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_inputs), batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_inputs)
            total_rows += len(batch_inputs)

        validation_prediction = _predict_normalized(model, validation_inputs, scaler)
        validation_metrics = evaluate_multistep(validation_targets, validation_prediction)
        validation_mae = validation_metrics["MAE"]
        history.append(
            {
                "epoch": epoch,
                "train_mse": total_loss / total_rows,
                "validation_mae": validation_mae,
            }
        )
        if validation_mae < best_mae:
            best_mae = validation_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    if best_state is None:  # pragma: no cover - non-empty validation is validated above
        raise RuntimeError("GRU training did not produce a validation checkpoint.")
    model.load_state_dict(best_state)
    validation_prediction = _predict_normalized(model, validation_inputs, scaler)
    return GRUTrainingResult(
        model=model,
        scaler=scaler,
        best_epoch=best_epoch,
        validation_prediction=validation_prediction,
        validation_metrics=evaluate_multistep(validation_targets, validation_prediction),
        history=pd.DataFrame(history),
        device=str(device),
    )


def predict_gru(
    model: DirectGRU,
    inputs: np.ndarray,
    scaler: LoadScaler,
) -> np.ndarray:
    """Return raw-unit direct trajectories from a validation-selected GRU."""

    require_torch()
    values = _as_finite_float_array(inputs, "inputs")
    if values.ndim != 3 or values.shape[0] == 0 or values.shape[2] != 1:
        raise ValueError("inputs must have shape (samples, context_steps, 1).")
    try:
        device = next(model.parameters()).device
    except (AttributeError, StopIteration) as exc:
        raise ValueError("model must be an initialized DirectGRU.") from exc
    normalized_inputs = torch.as_tensor(
        scaler.transform(values), dtype=torch.float32, device=device
    )
    return _predict_normalized(model, normalized_inputs, scaler)


def calibrate_residual_intervals(
    point_predictions: np.ndarray,
    validation_residuals: np.ndarray,
) -> np.ndarray:
    """Calibrate non-negative, ordered P10/P50/P90 trajectories by lead time."""

    points = _as_finite_float_array(point_predictions, "point_predictions")
    residuals = _as_finite_float_array(validation_residuals, "validation_residuals")
    if points.ndim != 2 or residuals.ndim != 2 or points.shape[1] != residuals.shape[1]:
        raise ValueError("predictions and residuals must be 2D with the same horizon.")
    if residuals.shape[0] == 0:
        raise ValueError("validation_residuals must not be empty.")

    lower_residual, upper_residual = np.quantile(residuals, [0.1, 0.9], axis=0)
    p50 = np.maximum(points, 0.0)
    p10 = np.minimum(np.maximum(points + lower_residual, 0.0), p50)
    p90 = np.maximum(np.maximum(points + upper_residual, 0.0), p50)
    return np.stack((p10, p50, p90), axis=2).astype(np.float32, copy=False)


def run_gru_benchmark(
    series: pd.Series,
    horizon: int,
    config: GRUConfig | None = None,
) -> GRUBenchmarkResult:
    """Train, test, and calibrate one direct multi-step GRU benchmark."""

    require_torch()
    config = config or GRUConfig()
    if not isinstance(config, GRUConfig):
        raise ValueError("config must be a GRUConfig instance.")
    horizon = _positive_integer(horizon, "horizon")
    windows = make_sequence_windows(
        series, horizon=horizon, context_steps=config.context_steps
    )
    partitions = split_sequence_windows(windows, horizon=horizon)
    training = fit_direct_gru(partitions["train"], partitions["validation"], config)
    test_prediction = predict_gru(
        training.model, partitions["test"].inputs, training.scaler
    )
    validation_residuals = (
        partitions["validation"].targets.to_numpy(dtype=np.float32)
        - training.validation_prediction
    )
    latest_inputs = series.to_numpy(dtype=np.float32)[-config.context_steps:].reshape(
        1, config.context_steps, 1
    )
    latest_prediction = predict_gru(training.model, latest_inputs, training.scaler)
    latest_intervals = calibrate_residual_intervals(
        latest_prediction, validation_residuals
    )[0]
    future_index = pd.date_range(
        series.index[-1] + pd.Timedelta(minutes=15),
        periods=horizon,
        freq="15min",
        name="forecast_timestamp",
    )
    latest_forecast = pd.DataFrame(
        {
            "step": np.arange(1, horizon + 1),
            "prediction": latest_prediction[0],
            "p10": latest_intervals[:, 0],
            "p50": latest_intervals[:, 1],
            "p90": latest_intervals[:, 2],
            "interval_method": "validation_residual_calibration",
        },
        index=future_index,
    )
    return GRUBenchmarkResult(
        config=config,
        horizon=horizon,
        partitions=partitions,
        training=training,
        test_prediction=test_prediction,
        test_metrics=evaluate_multistep(
            partitions["test"].targets.to_numpy(dtype=np.float32), test_prediction
        ),
        latest_forecast=latest_forecast,
        validation_residual_quantiles=np.quantile(
            validation_residuals, [0.1, 0.5, 0.9], axis=0
        ).T,
    )


def _set_torch_seed(seed: int) -> None:
    """Configure deterministic random streams for a repeatable training run."""

    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _torch_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _normalize_targets(targets: np.ndarray, scaler: LoadScaler) -> np.ndarray:
    values = _as_finite_float_array(targets, "targets")
    return ((values - scaler.mean) / scaler.std).astype(np.float32, copy=False)


def _predict_normalized(model: DirectGRU, inputs, scaler: LoadScaler) -> np.ndarray:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        normalized_prediction = model(inputs).detach().cpu().numpy()
    if was_training:
        model.train()
    return scaler.inverse_transform(normalized_prediction)


def _validate_partition(
    partition: SequencePartition, name: str, context_steps: int
) -> None:
    if not isinstance(partition, SequencePartition):
        raise ValueError(f"{name} must be a SequencePartition instance.")
    inputs = _as_finite_float_array(partition.inputs, f"{name}.inputs")
    targets = _as_finite_float_array(
        partition.targets.to_numpy(dtype=np.float32), f"{name}.targets"
    )
    if inputs.ndim != 3 or inputs.shape[0] == 0 or inputs.shape[1:] != (context_steps, 1):
        raise ValueError(
            f"{name}.inputs must have shape (samples, {context_steps}, 1)."
        )
    if targets.ndim != 2 or targets.shape[0] != inputs.shape[0] or targets.shape[1] == 0:
        raise ValueError(f"{name} inputs and targets must be non-empty and aligned.")


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
