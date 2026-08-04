# Multi-Step Forecasting Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate 4-step and 96-step load trajectory baselines for `MT_252` using naive, seasonal-naive, and direct multi-output Ridge models.

**Architecture:** Convert the ordered load series into origin-indexed feature and target matrices, split samples chronologically without allowing targets to cross boundaries, and evaluate three models through focused utility modules. A thin executed notebook orchestrates the real UCI experiment and writes portfolio tables and figures.

**Tech Stack:** Python 3.12, pandas, NumPy, scikit-learn, matplotlib, seaborn, pytest, nbconvert.

## Global Constraints

- Use only meter `MT_252`.
- One raw step equals 15 minutes.
- The 1-hour task outputs 4 future values; the 24-hour task outputs 96 future values.
- Features at forecast origin `t` may use only observations at or before `t`.
- Use lags `1, 4, 96, 192, 672` and shifted rolling windows `4, 96, 672`.
- Use cyclical hour, day-of-week, and month features, plus weekend and Portugal holiday flags.
- Split timestamps chronologically into 70% train, 15% validation, and 15% test.
- No target vector may cross a partition boundary.
- Select Ridge alpha from `[0.1, 1.0, 10.0, 100.0]` by validation overall MAE; ties select the smaller alpha.
- Evaluate the selected model once on test data without refitting on validation data.
- MAE is the primary metric; also report RMSE, MAPE, endpoint MAE, and per-step MAE.
- Do not add multi-meter, recursive, probabilistic, tree, deep-learning, robustness, or dashboard functionality.

---

## File Map

- `src/forecasting.py`: multi-step targets and leakage-safe chronological splits.
- `src/features.py`: cyclical calendar values and baseline feature matrix.
- `src/baselines.py`: naive forecasts, seasonal-naive forecasts, and Ridge alpha selection.
- `src/evaluate.py`: multi-step summary and per-step metrics.
- `tests/test_forecasting.py`: target and split boundary tests.
- `tests/test_features.py`: feature availability and anti-leakage tests.
- `tests/test_baselines.py`: model shape and alpha-selection tests.
- `tests/test_evaluate.py`: hand-calculated metric tests.
- `notebooks/02_baseline_models.ipynb`: real-data experiment and artifact generation.
- `notebooks/README.md`: notebook execution order.
- `README.md`: Week 2 results and figures.
- `requirements.txt`: pytest dependency.

---

### Task 1: Multi-Step Targets And Leakage-Safe Splits

**Files:**
- Modify: `requirements.txt`
- Modify: `src/forecasting.py`
- Create: `tests/test_forecasting.py`

**Interfaces:**
- Consumes: `pd.Series` with a unique, increasing 15-minute `DatetimeIndex`.
- Produces: `make_multistep_targets(series, horizon) -> pd.DataFrame`.
- Produces: `split_supervised_by_time(features, targets, full_index, horizon, train_size=0.7, val_size=0.15) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]`.

- [ ] **Step 1: Add pytest and write failing forecasting tests**

Add `pytest>=8.0.0` to `requirements.txt`, then create `tests/test_forecasting.py`:

```python
import numpy as np
import pandas as pd
import pytest

from src.forecasting import make_multistep_targets, split_supervised_by_time


def make_series(length: int = 1000) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=length, freq="15min")
    return pd.Series(np.arange(length, dtype=float), index=index, name="load")


def test_make_multistep_targets_contains_ordered_future_values():
    series = make_series(20)
    targets = make_multistep_targets(series, horizon=4)

    assert list(targets.columns) == [
        "target_step_1",
        "target_step_2",
        "target_step_3",
        "target_step_4",
    ]
    assert targets.iloc[5].tolist() == [6.0, 7.0, 8.0, 9.0]
    assert targets.iloc[-1].isna().all()


@pytest.mark.parametrize("horizon", [0, 97])
def test_make_multistep_targets_rejects_unsupported_horizon(horizon):
    with pytest.raises(ValueError, match="between 1 and 96"):
        make_multistep_targets(make_series(), horizon=horizon)


def test_make_multistep_targets_rejects_irregular_index():
    series = make_series(20).drop(make_series(20).index[5])
    with pytest.raises(ValueError, match="15-minute"):
        make_multistep_targets(series, horizon=4)


def test_split_keeps_every_target_inside_its_partition():
    series = make_series(1000)
    targets = make_multistep_targets(series, horizon=96).dropna()
    features = pd.DataFrame({"current_load": series}, index=series.index).loc[targets.index]

    splits = split_supervised_by_time(
        features,
        targets,
        full_index=series.index,
        horizon=96,
    )

    train_end_position = int(len(series) * 0.7)
    validation_end_position = int(len(series) * 0.85)
    positions = {name: series.index.get_indexer(X.index) for name, (X, _) in splits.items()}

    assert np.all(positions["train"] + 96 < train_end_position)
    assert np.all(positions["validation"] >= train_end_position)
    assert np.all(positions["validation"] + 96 < validation_end_position)
    assert np.all(positions["test"] >= validation_end_position)
    assert np.all(positions["test"] + 96 < len(series))


def test_validation_features_can_use_pre_boundary_history():
    series = make_series(1000)
    targets = make_multistep_targets(series, horizon=4).dropna()
    features = pd.DataFrame({"lag_672": series.shift(672)}, index=series.index).dropna()
    common_index = features.index.intersection(targets.index)

    splits = split_supervised_by_time(
        features.loc[common_index],
        targets.loc[common_index],
        full_index=series.index,
        horizon=4,
    )

    X_validation, _ = splits["validation"]
    first_validation_position = series.index.get_loc(X_validation.index[0])
    assert first_validation_position == int(len(series) * 0.7)
    assert X_validation.iloc[0, 0] == series.iloc[first_validation_position - 672]
```

- [ ] **Step 2: Run tests and verify the new interfaces are missing**

Run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest tests\test_forecasting.py -v
```

Expected: collection fails because `make_multistep_targets` and
`split_supervised_by_time` are not defined.

- [ ] **Step 3: Implement targets and splits in `src/forecasting.py`**

Keep `make_time_split` and `make_horizon_target` for backward compatibility,
then append:

```python
import numpy as np


MAX_SUPPORTED_HORIZON = 96
EXPECTED_FREQUENCY = pd.Timedelta(minutes=15)


def _validate_time_index(index: pd.DatetimeIndex) -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("series must use a DatetimeIndex.")
    if index.hasnans or not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError("timestamps must be present, unique, and increasing.")
    if len(index) > 1 and not index.to_series().diff().dropna().eq(EXPECTED_FREQUENCY).all():
        raise ValueError("timestamps must have a continuous 15-minute frequency.")


def make_multistep_targets(series: pd.Series, horizon: int) -> pd.DataFrame:
    """Create target columns for every future step after each forecast origin."""

    if not 1 <= horizon <= MAX_SUPPORTED_HORIZON:
        raise ValueError("horizon must be between 1 and 96 steps.")
    _validate_time_index(series.index)
    if len(series) <= horizon:
        raise ValueError("series is too short for the requested horizon.")

    return pd.DataFrame(
        {
            f"target_step_{step}": series.shift(-step)
            for step in range(1, horizon + 1)
        },
        index=series.index,
    )


def split_supervised_by_time(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    full_index: pd.DatetimeIndex,
    horizon: int,
    train_size: float = 0.7,
    val_size: float = 0.15,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Split aligned supervised samples while keeping targets inside partitions."""

    if not 1 <= horizon <= MAX_SUPPORTED_HORIZON:
        raise ValueError("horizon must be between 1 and 96 steps.")
    if not 0 < train_size < 1 or not 0 < val_size < 1 or train_size + val_size >= 1:
        raise ValueError("train_size and val_size must define three non-empty partitions.")
    _validate_time_index(full_index)
    if not features.index.equals(targets.index):
        raise ValueError("features and targets must have identical indexes.")

    origin_positions = full_index.get_indexer(features.index)
    if np.any(origin_positions < 0):
        raise ValueError("every supervised origin must exist in full_index.")

    target_end_positions = origin_positions + horizon
    train_end = int(len(full_index) * train_size)
    validation_end = int(len(full_index) * (train_size + val_size))
    masks = {
        "train": target_end_positions < train_end,
        "validation": (origin_positions >= train_end) & (target_end_positions < validation_end),
        "test": (origin_positions >= validation_end) & (target_end_positions < len(full_index)),
    }

    splits = {}
    for name, mask in masks.items():
        if not np.any(mask):
            raise ValueError(f"{name} partition has no supervised samples.")
        splits[name] = (features.loc[mask].copy(), targets.loc[mask].copy())
    return splits
```

- [ ] **Step 4: Run forecasting tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_forecasting.py -v
```

Expected: 6 tests pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add requirements.txt src\forecasting.py tests\test_forecasting.py
git commit -m "feat: add leakage-safe multi-step forecast windows"
```

---

### Task 2: Baseline Feature Matrix

**Files:**
- Modify: `src/features.py`
- Create: `tests/test_features.py`

**Interfaces:**
- Consumes: the `load` series validated by Task 1.
- Produces: `add_cyclical_time_features(df, timestamp_col="timestamp") -> pd.DataFrame`.
- Produces: `build_baseline_features(series, country="PT") -> pd.DataFrame`.

- [ ] **Step 1: Write failing feature tests**

Create `tests/test_features.py`:

```python
import numpy as np
import pandas as pd

from src.features import add_cyclical_time_features, build_baseline_features


def make_series(length: int = 800) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=length, freq="15min")
    values = 100 + 10 * np.sin(np.arange(length) * 2 * np.pi / 96)
    return pd.Series(values, index=index, name="load")


def test_cyclical_features_wrap_at_daily_boundary():
    frame = make_series(97).to_frame()
    result = add_cyclical_time_features(frame)

    assert np.isclose(result.iloc[0]["hour_sin"], result.iloc[96]["hour_sin"])
    assert np.isclose(result.iloc[0]["hour_cos"], result.iloc[96]["hour_cos"])


def test_baseline_features_start_after_full_historical_context():
    series = make_series()
    features = build_baseline_features(series)

    assert features.index[0] == series.index[672]
    assert features.loc[series.index[672], "current_load"] == series.iloc[672]
    assert features.loc[series.index[672], "load_lag_672"] == series.iloc[0]
    assert features.loc[series.index[672], "load_rolling_mean_4"] == series.iloc[668:672].mean()


def test_baseline_features_contain_only_expected_columns():
    features = build_baseline_features(make_series())
    assert list(features.columns) == [
        "current_load",
        "load_lag_1",
        "load_lag_4",
        "load_lag_96",
        "load_lag_192",
        "load_lag_672",
        "load_rolling_mean_4",
        "load_rolling_std_4",
        "load_rolling_mean_96",
        "load_rolling_std_96",
        "load_rolling_mean_672",
        "load_rolling_std_672",
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",
        "is_weekend",
        "is_holiday",
    ]
    assert not features.isna().any().any()
```

- [ ] **Step 2: Run tests and verify feature functions are missing**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_features.py -v
```

Expected: collection fails because the two new feature functions are absent.

- [ ] **Step 3: Implement cyclical and baseline features**

Add `import numpy as np` to `src/features.py`, then append:

```python
BASELINE_LAGS = (1, 4, 96, 192, 672)
BASELINE_WINDOWS = (4, 96, 672)


def add_cyclical_time_features(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Add continuous cyclical encodings for linear models."""

    out = df.copy()
    ts = pd.Series(
        pd.to_datetime(out[timestamp_col] if timestamp_col in out.columns else out.index),
        index=out.index,
    )
    hour = ts.dt.hour + ts.dt.minute / 60.0
    day_of_week = ts.dt.dayofweek + hour / 24.0
    month = ts.dt.month - 1

    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    out["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    return out


def build_baseline_features(series: pd.Series, country: str = "PT") -> pd.DataFrame:
    """Build origin-time features known at or before each forecast origin."""

    if not isinstance(series.index, pd.DatetimeIndex) or not series.index.is_monotonic_increasing:
        raise ValueError("series must use an increasing DatetimeIndex.")
    if len(series) <= max(BASELINE_LAGS):
        raise ValueError("series is too short for the 672-step historical context.")

    out = series.rename("load").to_frame()
    out = add_time_features(out)
    out = add_holiday_feature(out, country=country)
    out = add_cyclical_time_features(out)
    out = add_lag_features(out, lags=BASELINE_LAGS)
    out = add_rolling_features(out, windows=BASELINE_WINDOWS)
    out["current_load"] = out["load"]

    columns = [
        "current_load",
        *[f"load_lag_{lag}" for lag in BASELINE_LAGS],
        *[
            name
            for window in BASELINE_WINDOWS
            for name in (
                f"load_rolling_mean_{window}",
                f"load_rolling_std_{window}",
            )
        ],
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",
        "is_weekend",
        "is_holiday",
    ]
    return out[columns].dropna().copy()
```

- [ ] **Step 4: Run feature tests and the existing module compile check**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_features.py -v
.\.venv\Scripts\python.exe -m compileall src
```

Expected: 3 tests pass and all source modules compile.

- [ ] **Step 5: Commit Task 2**

```powershell
git add src\features.py tests\test_features.py
git commit -m "feat: add cyclical load forecasting features"
```

---

### Task 3: Forecast Baseline Models

**Files:**
- Create: `src/baselines.py`
- Create: `tests/test_baselines.py`

**Interfaces:**
- Consumes: current-load vectors, the full observed series, forecast origins, and Task 1 split matrices.
- Produces: `naive_forecast(current_load, horizon) -> np.ndarray`.
- Produces: `seasonal_naive_forecast(series, origins, horizon, season_length=96) -> np.ndarray`.
- Produces: `select_ridge_alpha(X_train, y_train, X_val, y_val, alphas=(0.1, 1.0, 10.0, 100.0)) -> tuple[Pipeline, pd.DataFrame]`.

- [ ] **Step 1: Write failing baseline tests**

Create `tests/test_baselines.py`:

```python
import numpy as np
import pandas as pd

from src.baselines import naive_forecast, seasonal_naive_forecast, select_ridge_alpha


def test_naive_forecast_repeats_current_load():
    prediction = naive_forecast(np.array([10.0, 20.0]), horizon=4)
    np.testing.assert_allclose(
        prediction,
        np.array([[10.0, 10.0, 10.0, 10.0], [20.0, 20.0, 20.0, 20.0]]),
    )


def test_seasonal_naive_uses_previous_day_matching_steps():
    index = pd.date_range("2024-01-01", periods=300, freq="15min")
    series = pd.Series(np.arange(300, dtype=float), index=index)
    origins = index[[150, 200]]

    prediction = seasonal_naive_forecast(series, origins, horizon=4)

    np.testing.assert_allclose(prediction[0], [55.0, 56.0, 57.0, 58.0])
    np.testing.assert_allclose(prediction[1], [105.0, 106.0, 107.0, 108.0])


def test_seasonal_naive_96_step_forecast_never_uses_future_data():
    index = pd.date_range("2024-01-01", periods=300, freq="15min")
    series = pd.Series(np.arange(300, dtype=float), index=index)
    origin = index[[150]]

    prediction = seasonal_naive_forecast(series, origin, horizon=96)

    assert prediction.shape == (1, 96)
    assert prediction[0, -1] == series.loc[origin[0]]


def test_ridge_selection_returns_multioutput_model_and_smallest_tied_alpha():
    X_train = pd.DataFrame({"constant": np.zeros(20)})
    y_train = pd.DataFrame(np.ones((20, 4)))
    X_val = pd.DataFrame({"constant": np.zeros(5)})
    y_val = pd.DataFrame(np.ones((5, 4)))

    model, results = select_ridge_alpha(
        X_train,
        y_train,
        X_val,
        y_val,
        alphas=(0.1, 1.0, 10.0),
    )

    assert model.named_steps["ridge"].alpha == 0.1
    assert list(results.columns) == ["alpha", "validation_MAE"]
    assert model.predict(X_val).shape == (5, 4)
```

- [ ] **Step 2: Run tests and verify `src.baselines` is missing**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_baselines.py -v
```

Expected: collection fails with `ModuleNotFoundError: src.baselines`.

- [ ] **Step 3: Implement `src/baselines.py`**

```python
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
```

- [ ] **Step 4: Run baseline tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_baselines.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit Task 3**

```powershell
git add src\baselines.py tests\test_baselines.py
git commit -m "feat: add multi-step forecasting baselines"
```

---

### Task 4: Multi-Step Evaluation Metrics

**Files:**
- Modify: `src/evaluate.py`
- Create: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: matching two-dimensional target and prediction arrays.
- Produces: `evaluate_multistep(y_true, y_pred) -> dict[str, float]`.
- Produces: `per_step_metrics(y_true, y_pred) -> pd.DataFrame`.

- [ ] **Step 1: Write failing metric tests**

Create `tests/test_evaluate.py`:

```python
import numpy as np
import pytest

from src.evaluate import evaluate_multistep, per_step_metrics


def test_multistep_summary_matches_hand_calculation():
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = np.array([[2.0, 2.0], [1.0, 5.0]])

    metrics = evaluate_multistep(y_true, y_pred)

    assert metrics["MAE"] == pytest.approx(1.0)
    assert metrics["RMSE"] == pytest.approx(np.sqrt(1.5))
    assert metrics["Endpoint_MAE"] == pytest.approx(0.5)


def test_per_step_metrics_returns_one_row_per_step():
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = np.array([[2.0, 2.0], [1.0, 5.0]])

    result = per_step_metrics(y_true, y_pred)

    assert result["forecast_step"].tolist() == [1, 2]
    assert result["lead_minutes"].tolist() == [15, 30]
    assert result["MAE"].tolist() == pytest.approx([1.5, 0.5])


def test_multistep_metrics_reject_shape_mismatch():
    with pytest.raises(ValueError, match="identical shapes"):
        evaluate_multistep(np.ones((2, 4)), np.ones((2, 3)))
```

- [ ] **Step 2: Run tests and verify the new metric functions are missing**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_evaluate.py -v
```

Expected: collection fails because the two functions are not defined.

- [ ] **Step 3: Implement multi-step metrics**

Add `import pandas as pd` to `src/evaluate.py`, then append:

```python
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
```

- [ ] **Step 4: Run the focused and complete test suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_evaluate.py -v
.\.venv\Scripts\python.exe -m pytest -v
```

Expected: 3 metric tests pass, followed by all 16 tests passing.

- [ ] **Step 5: Commit Task 4**

```powershell
git add src\evaluate.py tests\test_evaluate.py
git commit -m "feat: add multi-step forecast evaluation"
```

---

### Task 5: Real-Data Baseline Notebook And Artifacts

**Files:**
- Create: `notebooks/02_baseline_models.ipynb`
- Modify: `notebooks/README.md`
- Create: `reports/tables/baseline_metrics.csv` by notebook execution.
- Create: `reports/tables/baseline_metrics_by_step.csv` by notebook execution.
- Create: `reports/tables/ridge_alpha_selection.csv` by notebook execution.
- Create: `reports/figures/baseline_forecast_examples.png` by notebook execution.
- Create: `reports/figures/baseline_mae_by_step.png` by notebook execution.

**Interfaces:**
- Consumes: all Task 1-4 public functions and `data/raw/LD2011_2014.txt`.
- Produces: an executed notebook, three CSV tables, and two PNG figures.

- [ ] **Step 1: Create the notebook with these code cells in order**

Create a notebook with markdown describing the two trajectory horizons, then
use the following complete Python cells.

Setup cell:

```python
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.append(str(PROJECT_ROOT))

from src.baselines import naive_forecast, seasonal_naive_forecast, select_ridge_alpha
from src.config import RAW_DATA_DIR, RAW_DATA_FILENAME, FIGURES_DIR, TABLES_DIR
from src.data_loader import load_electricity_load_data
from src.evaluate import evaluate_multistep, per_step_metrics
from src.features import build_baseline_features
from src.forecasting import make_multistep_targets, split_supervised_by_time

sns.set_theme(style="whitegrid")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
```

Data cell:

```python
raw_path = RAW_DATA_DIR / RAW_DATA_FILENAME
if not raw_path.exists():
    raise FileNotFoundError(f"Dataset not found: {raw_path}")

raw_df = load_electricity_load_data(raw_path)
series = raw_df["MT_252"].rename("load")
features = build_baseline_features(series)
print(series.shape, features.shape, series.index.min(), series.index.max())
```

Experiment cell:

```python
HORIZONS = {"1h": 4, "24h": 96}
ALPHAS = (0.1, 1.0, 10.0, 100.0)
metric_rows = []
step_frames = []
alpha_frames = []
artifacts = {}

for horizon_label, horizon in HORIZONS.items():
    targets = make_multistep_targets(series, horizon).dropna()
    common_index = features.index.intersection(targets.index)
    X = features.loc[common_index]
    y = targets.loc[common_index]
    splits = split_supervised_by_time(X, y, series.index, horizon)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["validation"]
    X_test, y_test = splits["test"]

    ridge_model, alpha_results = select_ridge_alpha(
        X_train, y_train, X_val, y_val, alphas=ALPHAS
    )
    selected_alpha = ridge_model.named_steps["ridge"].alpha
    alpha_results.insert(0, "horizon", horizon_label)
    alpha_results["selected"] = alpha_results["alpha"].eq(selected_alpha)
    alpha_frames.append(alpha_results)

    predictions = {
        "Naive": naive_forecast(X_test["current_load"], horizon),
        "Seasonal Naive": seasonal_naive_forecast(series, X_test.index, horizon),
        "Ridge": ridge_model.predict(X_test),
    }

    for model_name, prediction in predictions.items():
        metrics = evaluate_multistep(y_test, prediction)
        metric_rows.append(
            {
                "horizon": horizon_label,
                "horizon_steps": horizon,
                "model": model_name,
                "selected_alpha": selected_alpha if model_name == "Ridge" else np.nan,
                **metrics,
            }
        )
        step_metrics = per_step_metrics(y_test, prediction)
        step_metrics.insert(0, "model", model_name)
        step_metrics.insert(0, "horizon", horizon_label)
        step_frames.append(step_metrics)

    artifacts[horizon_label] = {
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions,
    }

metrics_df = pd.DataFrame(metric_rows)
step_metrics_df = pd.concat(step_frames, ignore_index=True)
alpha_selection_df = pd.concat(alpha_frames, ignore_index=True)

metrics_df.to_csv(TABLES_DIR / "baseline_metrics.csv", index=False)
step_metrics_df.to_csv(TABLES_DIR / "baseline_metrics_by_step.csv", index=False)
alpha_selection_df.to_csv(TABLES_DIR / "ridge_alpha_selection.csv", index=False)
metrics_df
```

Forecast example figure cell:

```python
fig, axes = plt.subplots(2, 1, figsize=(14, 9))
for ax, (horizon_label, horizon) in zip(axes, HORIZONS.items()):
    artifact = artifacts[horizon_label]
    example_position = min(96, len(artifact["y_test"]) - 1)
    origin = artifact["y_test"].index[example_position]
    future_index = pd.date_range(
        origin + pd.Timedelta(minutes=15), periods=horizon, freq="15min"
    )
    ax.plot(
        future_index,
        artifact["y_test"].iloc[example_position],
        label="Actual",
        color="black",
        linewidth=2.2,
    )
    for model_name, prediction in artifact["predictions"].items():
        ax.plot(future_index, prediction[example_position], label=model_name, alpha=0.85)
    ax.set_title(f"{horizon_label} trajectory forecast from {origin}")
    ax.set_ylabel("Load (kW)")
    ax.legend(ncol=4)

axes[-1].set_xlabel("Forecast timestamp")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "baseline_forecast_examples.png", dpi=160)
plt.show()
```

Per-step MAE figure cell:

```python
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ax, horizon_label in zip(axes, HORIZONS):
    subset = step_metrics_df[step_metrics_df["horizon"] == horizon_label]
    sns.lineplot(
        data=subset,
        x="lead_minutes",
        y="MAE",
        hue="model",
        ax=ax,
    )
    ax.set_title(f"{horizon_label} MAE by forecast lead")
    ax.set_xlabel("Lead time (minutes)")
    ax.set_ylabel("MAE (kW)")

fig.tight_layout()
fig.savefig(FIGURES_DIR / "baseline_mae_by_step.png", dpi=160)
plt.show()
```

Result cell:

```python
display(metrics_df.sort_values(["horizon_steps", "MAE"]))
display(alpha_selection_df)
```

- [ ] **Step 2: Update notebook execution order**

Replace `notebooks/README.md` with:

```markdown
# Notebooks

Run notebooks in this order:

1. `01_eda.ipynb`: data quality, representative meter, and load patterns.
2. `02_baseline_models.ipynb`: 4-step and 96-step trajectory baselines.

Select `.venv\Scripts\python.exe` as the VS Code notebook kernel. Raw data must
exist at `data/raw/LD2011_2014.txt`.
```

- [ ] **Step 3: Execute the notebook in place**

Run:

```powershell
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace notebooks\02_baseline_models.ipynb --ExecutePreprocessor.timeout=1800
```

Expected: exit code 0 and an executed notebook containing metric tables and
both rendered figures.

- [ ] **Step 4: Verify generated artifacts**

Run:

```powershell
Get-Item reports\tables\baseline_metrics.csv
Get-Item reports\tables\baseline_metrics_by_step.csv
Get-Item reports\tables\ridge_alpha_selection.csv
Get-Item reports\figures\baseline_forecast_examples.png
Get-Item reports\figures\baseline_mae_by_step.png
Get-Content reports\tables\baseline_metrics.csv
```

Expected: every file is non-empty; metrics contain 6 rows covering both
horizons and all three models.

- [ ] **Step 5: Visually inspect both figures**

Open both PNG files and verify that axes, legends, titles, and lines are
readable; all four trajectory lines appear; no labels overlap; and the plots
are nonblank.

- [ ] **Step 6: Commit Task 5**

```powershell
git add notebooks\02_baseline_models.ipynb notebooks\README.md reports\figures reports\tables
git commit -m "feat: run multi-step baseline forecasting experiment"
```

---

### Task 6: Portfolio Documentation And Final Verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: measured CSV values and generated figures from Task 5.
- Produces: a GitHub-ready Week 2 project summary with no unsupported claims.

- [ ] **Step 1: Read measured results before writing documentation**

Run:

```powershell
Get-Content reports\tables\baseline_metrics.csv
Get-Content reports\tables\ridge_alpha_selection.csv
```

Use the following command to produce ready-to-paste Markdown rows from the
actual CSV instead of manually transcribing values:

```powershell
& '.\.venv\Scripts\python.exe' -c "import pandas as pd; df=pd.read_csv('reports/tables/baseline_metrics.csv'); print(df[['horizon','model','MAE','RMSE','MAPE','Endpoint_MAE']].to_markdown(index=False)); print('selected alpha:'); print(pd.read_csv('reports/tables/ridge_alpha_selection.csv').query('selected')[['horizon','alpha']].to_markdown(index=False))"
```

State which model has the lowest MAE for each horizon based on the generated
rows; do not claim Ridge wins when its measured MAE is higher than a baseline.

- [ ] **Step 2: Update README using the measured values**

Add these sections with the exact table printed in Step 1 before committing:

```markdown
## Multi-Step Baseline Results

This stage predicts complete future trajectories rather than one endpoint:
4 values for the next hour and 96 values for the next 24 hours. Models are
evaluated on the final chronological 15% of the series.

Paste the generated metric table and selected-alpha table here. Add a short
interpretation naming the lowest-MAE model for 1h and 24h, the selected Ridge
alpha for each horizon, and whether the per-step MAE increases with lead time.

![Baseline trajectory forecasts](reports/figures/baseline_forecast_examples.png)

![MAE by forecast lead](reports/figures/baseline_mae_by_step.png)
```

Replace `### Week 2: Baseline Models` with
`### Week 2: Baseline Models (Completed)` and mark all four existing Week 2
items complete in prose.

- [ ] **Step 3: Run all automated verification**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m compileall src
git diff --check
git status --short
```

Expected: all 16 tests pass, source compilation succeeds, `git diff --check`
prints nothing, and only the intended README change remains uncommitted.

- [ ] **Step 4: Commit documentation**

```powershell
git add README.md
git commit -m "docs: report multi-step baseline results"
```

- [ ] **Step 5: Push and verify GitHub**

```powershell
git push
git status --short --branch
git log -5 --oneline
```

Expected: push succeeds, status reports `main...origin/main` with no changed
files, and the latest commits correspond to Tasks 1-6.
