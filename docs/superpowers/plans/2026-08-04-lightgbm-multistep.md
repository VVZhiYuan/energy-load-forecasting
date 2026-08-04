# LightGBM Multi-Step Forecasting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a leakage-safe direct LightGBM trajectory forecaster, compare it with the existing baselines on `MT_252`, and publish reproducible SHAP-based portfolio evidence.

**Architecture:** `src/ml_models.py` owns validated per-step LightGBM fitting, validation-only candidate selection, prediction, and gain aggregation. `notebooks/03_ml_models.ipynb` reuses the existing feature/split/evaluation APIs, orchestrates the real-data experiment, computes SHAP only after model selection, and exports bounded CSV/PNG artifacts. The root README reports only executed measurements.

**Tech Stack:** Python 3.12, pandas, NumPy, scikit-learn, LightGBM, SHAP, joblib, matplotlib, seaborn, pytest, Jupyter/nbconvert.

## Global Constraints

- Dataset and meter remain UCI ElectricityLoadDiagrams20112014 and `MT_252` at 15-minute resolution.
- Horizons remain 4 steps (1 hour) and 96 steps (24 hours).
- Reuse the existing 20 baseline features and chronological 70/15/15 target-boundary-safe split.
- Primary selection metric is validation trajectory MAE; the test split is touched only after selection.
- Use one `LGBMRegressor` per target step and no recursive forecasts.
- Use `objective="regression_l1"`, `random_state=42`, `subsample=0.9`, `subsample_freq=1`, `colsample_bytree=0.9`, `n_jobs=1`, and 30-round validation early stopping.
- Candidate order and values are exactly Small `(15, 0.05, 300, 40, 1.0)`, Medium `(31, 0.05, 400, 20, 0.1)`, Large `(63, 0.03, 500, 20, 0.1)` for `(num_leaves, learning_rate, n_estimators, min_child_samples, reg_lambda)`.
- A validation-MAE tie selects the first declared candidate.
- SHAP uses at most 500 deterministic test origins and only steps 1/4 for 1h and 1/24/48/96 for 24h.
- Do not track raw data, serialized model binaries, or full prediction dumps.
- Do not claim causality, deployment, savings, or LightGBM superiority unless measured results support it.

---

### Task 1: Direct Per-Step LightGBM Forecaster

**Files:**
- Modify: `requirements.txt`
- Create: `src/ml_models.py`
- Create: `tests/test_ml_models.py`

**Interfaces:**
- Consumes: aligned numeric `pd.DataFrame` train/validation features and multi-step targets.
- Produces: `LightGBMCandidate`, `DirectLightGBMForecaster`, and `fit_direct_lightgbm(X_train, y_train, X_val, y_val, candidate, parallel_jobs=-1)`.

- [ ] **Step 1: Add the model dependencies**

Append exactly:

```text
lightgbm>=4.3.0
shap>=0.46.0
```

Install with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Expected: `lightgbm` and `shap` install successfully in `.venv`.

- [ ] **Step 2: Write failing fit, shape, validation, and reproducibility tests**

Create `tests/test_ml_models.py`:

```python
import numpy as np
import pandas as pd
import pytest

from src.ml_models import (
    LightGBMCandidate,
    fit_direct_lightgbm,
)


SMOKE_CANDIDATE = LightGBMCandidate(
    name="smoke",
    num_leaves=3,
    learning_rate=0.1,
    n_estimators=3,
    min_child_samples=2,
    reg_lambda=0.0,
)


def make_frames(horizon: int):
    train_index = pd.date_range("2024-01-01", periods=48, freq="15min")
    val_index = pd.date_range("2024-01-02", periods=16, freq="15min")
    X_train = pd.DataFrame(
        {
            "load": np.linspace(1.0, 5.0, len(train_index)),
            "hour": train_index.hour.astype(float),
        },
        index=train_index,
    )
    X_val = pd.DataFrame(
        {
            "load": np.linspace(5.1, 6.0, len(val_index)),
            "hour": val_index.hour.astype(float),
        },
        index=val_index,
    )
    offsets = np.arange(1, horizon + 1, dtype=float)
    y_train = pd.DataFrame(
        X_train["load"].to_numpy()[:, None] + offsets[None, :],
        index=train_index,
        columns=[f"target_step_{step}" for step in range(1, horizon + 1)],
    )
    y_val = pd.DataFrame(
        X_val["load"].to_numpy()[:, None] + offsets[None, :],
        index=val_index,
        columns=y_train.columns,
    )
    return X_train, y_train, X_val, y_val


@pytest.mark.parametrize("horizon", [4, 96])
def test_direct_lightgbm_returns_complete_trajectory(horizon):
    X_train, y_train, X_val, y_val = make_frames(horizon)
    forecaster = fit_direct_lightgbm(
        X_train,
        y_train,
        X_val,
        y_val,
        SMOKE_CANDIDATE,
        parallel_jobs=1,
    )

    assert forecaster.predict(X_val).shape == (len(X_val), horizon)
    assert len(forecaster.models) == horizon


def test_direct_lightgbm_rejects_misaligned_targets():
    X_train, y_train, X_val, y_val = make_frames(4)
    y_train.index = y_train.index.shift(1, freq="15min")

    with pytest.raises(ValueError, match="indexes"):
        fit_direct_lightgbm(
            X_train, y_train, X_val, y_val, SMOKE_CANDIDATE, parallel_jobs=1
        )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_direct_lightgbm_rejects_non_finite_features(bad_value):
    X_train, y_train, X_val, y_val = make_frames(4)
    X_train.iloc[0, 0] = bad_value

    with pytest.raises(ValueError, match="finite"):
        fit_direct_lightgbm(
            X_train, y_train, X_val, y_val, SMOKE_CANDIDATE, parallel_jobs=1
        )


def test_direct_lightgbm_rejects_changed_prediction_columns():
    X_train, y_train, X_val, y_val = make_frames(4)
    forecaster = fit_direct_lightgbm(
        X_train, y_train, X_val, y_val, SMOKE_CANDIDATE, parallel_jobs=1
    )

    with pytest.raises(ValueError, match="columns"):
        forecaster.predict(X_val.rename(columns={"hour": "weekday"}))


def test_direct_lightgbm_is_reproducible():
    X_train, y_train, X_val, y_val = make_frames(4)
    first = fit_direct_lightgbm(
        X_train, y_train, X_val, y_val, SMOKE_CANDIDATE, parallel_jobs=1
    )
    second = fit_direct_lightgbm(
        X_train, y_train, X_val, y_val, SMOKE_CANDIDATE, parallel_jobs=1
    )

    np.testing.assert_allclose(first.predict(X_val), second.predict(X_val))
```

- [ ] **Step 3: Run the tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_ml_models.py -v
```

Expected: collection fails because `src.ml_models` does not exist.

- [ ] **Step 4: Implement the direct forecaster**

Create `src/ml_models.py`:

```python
"""Direct per-step LightGBM models for multi-step load forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor, early_stopping, log_evaluation


@dataclass(frozen=True)
class LightGBMCandidate:
    """One bounded LightGBM configuration used during validation search."""

    name: str
    num_leaves: int
    learning_rate: float
    n_estimators: int
    min_child_samples: int
    reg_lambda: float


@dataclass(frozen=True)
class DirectLightGBMForecaster:
    """Ordered collection of one fitted LightGBM model per forecast step."""

    models: tuple[LGBMRegressor, ...]
    feature_names: tuple[str, ...]
    candidate: LightGBMCandidate

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        _validate_frame("features", features)
        if tuple(features.columns) != self.feature_names:
            raise ValueError("prediction feature columns must match training columns.")
        return np.column_stack([model.predict(features) for model in self.models])


def _validate_frame(name: str, frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{name} must be a non-empty DataFrame.")
    if not frame.index.is_unique:
        raise ValueError(f"{name} index must be unique.")
    try:
        values = frame.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values.")


def _validate_supervised_frames(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
) -> None:
    for name, frame in (
        ("X_train", X_train),
        ("y_train", y_train),
        ("X_val", X_val),
        ("y_val", y_val),
    ):
        _validate_frame(name, frame)
    if not X_train.index.equals(y_train.index) or not X_val.index.equals(y_val.index):
        raise ValueError("feature and target indexes must match within each partition.")
    if tuple(X_train.columns) != tuple(X_val.columns):
        raise ValueError("training and validation feature columns must match.")
    if tuple(y_train.columns) != tuple(y_val.columns):
        raise ValueError("training and validation target columns must match.")


def _fit_one_step(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    candidate: LightGBMCandidate,
) -> LGBMRegressor:
    model = LGBMRegressor(
        objective="regression_l1",
        random_state=42,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        n_jobs=1,
        verbosity=-1,
        num_leaves=candidate.num_leaves,
        learning_rate=candidate.learning_rate,
        n_estimators=candidate.n_estimators,
        min_child_samples=candidate.min_child_samples,
        reg_lambda=candidate.reg_lambda,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="l1",
        callbacks=[early_stopping(30, verbose=False), log_evaluation(0)],
    )
    return model


def fit_direct_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    candidate: LightGBMCandidate,
    parallel_jobs: int = -1,
) -> DirectLightGBMForecaster:
    """Fit one deterministic LightGBM estimator for every target step."""

    _validate_supervised_frames(X_train, y_train, X_val, y_val)
    if not isinstance(candidate, LightGBMCandidate):
        raise ValueError("candidate must be a LightGBMCandidate.")
    models = Parallel(n_jobs=parallel_jobs, prefer="threads")(
        delayed(_fit_one_step)(
            X_train,
            y_train.iloc[:, step],
            X_val,
            y_val.iloc[:, step],
            candidate,
        )
        for step in range(y_train.shape[1])
    )
    return DirectLightGBMForecaster(
        models=tuple(models),
        feature_names=tuple(X_train.columns),
        candidate=candidate,
    )
```

- [ ] **Step 5: Run focused and regression tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_ml_models.py -v
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -v
```

Expected: 7 new test cases pass, followed by all existing and new tests passing.

- [ ] **Step 6: Commit Task 1**

```powershell
git add requirements.txt src\ml_models.py tests\test_ml_models.py
git commit -m "feat: add direct LightGBM trajectory forecaster"
```

---

### Task 2: Candidate Selection And Gain Importance

**Files:**
- Modify: `src/ml_models.py`
- Modify: `tests/test_ml_models.py`

**Interfaces:**
- Consumes: `fit_direct_lightgbm`, `DirectLightGBMForecaster`, and validation frames from Task 1.
- Produces: `DEFAULT_LIGHTGBM_CANDIDATES`, `select_lightgbm_candidate(...) -> tuple[DirectLightGBMForecaster, pd.DataFrame]`, and `aggregate_gain_importance(forecaster) -> pd.DataFrame`.

- [ ] **Step 1: Write failing selection and importance tests**

Append to `tests/test_ml_models.py`:

```python
from src import ml_models
from src.ml_models import (
    DirectLightGBMForecaster,
    aggregate_gain_importance,
    select_lightgbm_candidate,
)


class FakeForecaster:
    def __init__(self, candidate, prediction):
        self.candidate = candidate
        self._prediction = prediction

    def predict(self, features):
        return np.full((len(features), 2), self._prediction, dtype=float)


def test_candidate_selection_uses_validation_mae(monkeypatch):
    index = pd.date_range("2024-01-01", periods=8, freq="15min")
    X = pd.DataFrame({"load": np.arange(8, dtype=float)}, index=index)
    y = pd.DataFrame(np.ones((8, 2)), index=index)
    candidates = (
        LightGBMCandidate("small", 3, 0.1, 3, 2, 0.0),
        LightGBMCandidate("medium", 5, 0.1, 3, 2, 0.0),
    )

    def fake_fit(X_train, y_train, X_val, y_val, candidate, parallel_jobs):
        prediction = 3.0 if candidate.name == "small" else 1.0
        return FakeForecaster(candidate, prediction)

    monkeypatch.setattr(ml_models, "fit_direct_lightgbm", fake_fit)
    selected, results = select_lightgbm_candidate(
        X, y, X, y, candidates=candidates, parallel_jobs=1
    )

    assert selected.candidate.name == "medium"
    assert results.loc[results["selected"], "candidate"].item() == "medium"


def test_candidate_selection_breaks_ties_by_declared_order(monkeypatch):
    index = pd.date_range("2024-01-01", periods=8, freq="15min")
    X = pd.DataFrame({"load": np.arange(8, dtype=float)}, index=index)
    y = pd.DataFrame(np.ones((8, 2)), index=index)
    candidates = (
        LightGBMCandidate("first", 3, 0.1, 3, 2, 0.0),
        LightGBMCandidate("second", 5, 0.1, 3, 2, 0.0),
    )

    def fake_fit(X_train, y_train, X_val, y_val, candidate, parallel_jobs):
        return FakeForecaster(candidate, 1.0)

    monkeypatch.setattr(ml_models, "fit_direct_lightgbm", fake_fit)
    selected, _ = select_lightgbm_candidate(
        X, y, X, y, candidates=candidates, parallel_jobs=1
    )

    assert selected.candidate.name == "first"


class FakeBooster:
    def __init__(self, gains):
        self._gains = np.asarray(gains, dtype=float)

    def feature_importance(self, importance_type):
        assert importance_type == "gain"
        return self._gains


class FakeModel:
    def __init__(self, gains):
        self.booster_ = FakeBooster(gains)


def test_gain_importance_contains_and_normalizes_every_feature():
    candidate = LightGBMCandidate("smoke", 3, 0.1, 3, 2, 0.0)
    forecaster = DirectLightGBMForecaster(
        models=(FakeModel([3.0, 1.0]), FakeModel([1.0, 1.0])),
        feature_names=("load", "hour"),
        candidate=candidate,
    )

    importance = aggregate_gain_importance(forecaster)

    assert set(importance["feature"]) == {"load", "hour"}
    assert importance["normalized_gain"].sum() == pytest.approx(1.0)
    assert importance.iloc[0]["rank"] == 1
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_ml_models.py -v
```

Expected: import errors for the three new names.

- [ ] **Step 3: Add fixed candidates, selection, and importance aggregation**

Append to `src/ml_models.py` after `LightGBMCandidate`:

```python
DEFAULT_LIGHTGBM_CANDIDATES = (
    LightGBMCandidate("small", 15, 0.05, 300, 40, 1.0),
    LightGBMCandidate("medium", 31, 0.05, 400, 20, 0.1),
    LightGBMCandidate("large", 63, 0.03, 500, 20, 0.1),
)
```

Add this import with the existing imports:

```python
from src.evaluate import mae
```

Append these functions to `src/ml_models.py`:

```python
def select_lightgbm_candidate(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    candidates=DEFAULT_LIGHTGBM_CANDIDATES,
    parallel_jobs: int = -1,
) -> tuple[DirectLightGBMForecaster, pd.DataFrame]:
    """Select the declared LightGBM candidate using validation MAE only."""

    candidates = tuple(candidates)
    if not candidates:
        raise ValueError("candidates must not be empty.")
    best_forecaster = None
    best_score = np.inf
    rows = []
    for candidate in candidates:
        forecaster = fit_direct_lightgbm(
            X_train,
            y_train,
            X_val,
            y_val,
            candidate,
            parallel_jobs=parallel_jobs,
        )
        score = mae(y_val, forecaster.predict(X_val))
        rows.append(
            {
                "candidate": candidate.name,
                "num_leaves": candidate.num_leaves,
                "learning_rate": candidate.learning_rate,
                "n_estimators": candidate.n_estimators,
                "min_child_samples": candidate.min_child_samples,
                "reg_lambda": candidate.reg_lambda,
                "validation_MAE": score,
            }
        )
        if score < best_score:
            best_score = score
            best_forecaster = forecaster
    results = pd.DataFrame(rows)
    results["selected"] = results["candidate"].eq(best_forecaster.candidate.name)
    return best_forecaster, results


def aggregate_gain_importance(
    forecaster: DirectLightGBMForecaster,
) -> pd.DataFrame:
    """Aggregate and normalize gain importance across forecast-step models."""

    gains = np.vstack(
        [
            model.booster_.feature_importance(importance_type="gain")
            for model in forecaster.models
        ]
    )
    raw_gain = gains.sum(axis=0)
    total = raw_gain.sum()
    normalized = raw_gain / total if total > 0 else np.zeros_like(raw_gain)
    result = pd.DataFrame(
        {
            "feature": forecaster.feature_names,
            "raw_gain": raw_gain,
            "normalized_gain": normalized,
        }
    ).sort_values(["normalized_gain", "feature"], ascending=[False, True])
    result = result.reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    return result
```

- [ ] **Step 4: Run focused and full tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_ml_models.py -v
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -v
```

Expected: all model tests and the complete suite pass.

- [ ] **Step 5: Commit Task 2**

```powershell
git add src\ml_models.py tests\test_ml_models.py
git commit -m "feat: select and explain LightGBM candidates"
```

---

### Task 3: Real-Data LightGBM Notebook And Artifacts

**Files:**
- Create: `notebooks/03_ml_models.ipynb`
- Modify: `notebooks/README.md`
- Create by notebook execution: `reports/tables/ml_model_metrics.csv`
- Create by notebook execution: `reports/tables/ml_metrics_by_step.csv`
- Create by notebook execution: `reports/tables/lgbm_parameter_search.csv`
- Create by notebook execution: `reports/tables/lgbm_feature_importance.csv`
- Create by notebook execution: `reports/figures/lgbm_forecast_examples.png`
- Create by notebook execution: `reports/figures/lgbm_mae_by_step.png`
- Create by notebook execution: `reports/figures/lgbm_feature_importance.png`
- Create by notebook execution: `reports/figures/lgbm_shap_summary.png`

**Interfaces:**
- Consumes: all existing data, feature, split, baseline, evaluation APIs plus Task 2's LightGBM APIs.
- Produces: an executed experiment notebook and the exact eight bounded portfolio artifacts listed above.

- [ ] **Step 1: Create the notebook narrative and setup cells**

Create a valid nbformat 4 notebook with markdown explaining the untouched test split, direct per-step strategy, primary trajectory MAE, and non-causal SHAP interpretation. Its setup code must be:

```python
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.append(str(PROJECT_ROOT))

from src.baselines import naive_forecast, seasonal_naive_forecast, select_ridge_alpha
from src.config import RAW_DATA_DIR, RAW_DATA_FILENAME, FIGURES_DIR, TABLES_DIR
from src.data_loader import load_electricity_load_data
from src.evaluate import evaluate_multistep, per_step_metrics
from src.features import build_baseline_features
from src.forecasting import make_multistep_targets, split_supervised_by_time
from src.ml_models import aggregate_gain_importance, select_lightgbm_candidate

sns.set_theme(style="whitegrid")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
```

Load data with:

```python
raw_path = RAW_DATA_DIR / RAW_DATA_FILENAME
if not raw_path.exists():
    raise FileNotFoundError(f"Dataset not found: {raw_path}")
raw_df = load_electricity_load_data(raw_path)
series = raw_df["MT_252"].rename("load")
features = build_baseline_features(series)
print(f"Series rows: {len(series):,}; feature rows: {len(features):,}")
```

- [ ] **Step 2: Add the complete training and export cell**

Use this experiment code:

```python
HORIZONS = {"1h": 4, "24h": 96}
metric_rows = []
step_frames = []
search_frames = []
importance_frames = []
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

    ridge_model, _ = select_ridge_alpha(X_train, y_train, X_val, y_val)
    lgbm_model, search = select_lightgbm_candidate(
        X_train, y_train, X_val, y_val, parallel_jobs=-1
    )
    search.insert(0, "horizon", horizon_label)
    search_frames.append(search)

    predictions = {
        "Naive": naive_forecast(X_test["current_load"], horizon),
        "Seasonal Naive": seasonal_naive_forecast(series, X_test.index, horizon),
        "Ridge": ridge_model.predict(X_test),
        "LightGBM": lgbm_model.predict(X_test),
    }
    horizon_metrics = {}
    for model_name, prediction in predictions.items():
        metrics = evaluate_multistep(y_test, prediction)
        horizon_metrics[model_name] = metrics
        metric_rows.append(
            {
                "horizon": horizon_label,
                "horizon_steps": horizon,
                "model": model_name,
                **metrics,
            }
        )
        by_step = per_step_metrics(y_test, prediction)
        by_step.insert(0, "model", model_name)
        by_step.insert(0, "horizon", horizon_label)
        step_frames.append(by_step)

    importance = aggregate_gain_importance(lgbm_model)
    importance.insert(0, "horizon", horizon_label)
    importance_frames.append(importance)
    artifacts[horizon_label] = {
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions,
        "lgbm_model": lgbm_model,
    }

metrics_df = pd.DataFrame(metric_rows)
for horizon_label in HORIZONS:
    mask = metrics_df["horizon"].eq(horizon_label)
    seasonal_mae = metrics_df.loc[
        mask & metrics_df["model"].eq("Seasonal Naive"), "MAE"
    ].item()
    lgbm_mask = mask & metrics_df["model"].eq("LightGBM")
    metrics_df.loc[lgbm_mask, "improvement_vs_seasonal_pct"] = (
        (seasonal_mae - metrics_df.loc[lgbm_mask, "MAE"]) / seasonal_mae * 100.0
    )

step_metrics_df = pd.concat(step_frames, ignore_index=True)
search_df = pd.concat(search_frames, ignore_index=True)
importance_df = pd.concat(importance_frames, ignore_index=True)
metrics_df.to_csv(TABLES_DIR / "ml_model_metrics.csv", index=False)
step_metrics_df.to_csv(TABLES_DIR / "ml_metrics_by_step.csv", index=False)
search_df.to_csv(TABLES_DIR / "lgbm_parameter_search.csv", index=False)
importance_df.to_csv(TABLES_DIR / "lgbm_feature_importance.csv", index=False)
metrics_df
```

- [ ] **Step 3: Add and export the comparison figures**

Add deterministic trajectory and lead-error plots using the same origin within each horizon:

```python
COLORS = {
    "Actual": "black",
    "Naive": "#4C72B0",
    "Seasonal Naive": "#DD8452",
    "Ridge": "#55A868",
    "LightGBM": "#C44E52",
}
fig, axes = plt.subplots(2, 1, figsize=(14, 9))
for ax, (horizon_label, horizon) in zip(axes, HORIZONS.items()):
    artifact = artifacts[horizon_label]
    position = min(96, len(artifact["y_test"]) - 1)
    origin = artifact["y_test"].index[position]
    future_index = pd.date_range(
        origin + pd.Timedelta(minutes=15), periods=horizon, freq="15min"
    )
    ax.plot(
        future_index,
        artifact["y_test"].iloc[position],
        color=COLORS["Actual"],
        linewidth=2.2,
        label="Actual",
    )
    for model_name, prediction in artifact["predictions"].items():
        ax.plot(
            future_index,
            prediction[position],
            color=COLORS[model_name],
            alpha=0.85,
            label=model_name,
        )
    ax.set_title(f"{horizon_label} trajectory forecast from {origin}")
    ax.set_ylabel("Load (kW)")
    ax.legend(ncol=5)
axes[-1].set_xlabel("Forecast timestamp")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "lgbm_forecast_examples.png", dpi=160)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ax, horizon_label in zip(axes, HORIZONS):
    subset = step_metrics_df[step_metrics_df["horizon"].eq(horizon_label)]
    sns.lineplot(
        data=subset,
        x="lead_minutes",
        y="MAE",
        hue="model",
        palette=COLORS,
        ax=ax,
    )
    ax.set_title(f"{horizon_label} MAE by forecast lead")
    ax.set_xlabel("Lead time (minutes)")
    ax.set_ylabel("MAE (kW)")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "lgbm_mae_by_step.png", dpi=160)
plt.show()
```

- [ ] **Step 4: Add gain and SHAP explanation figures**

Use aggregate gain:

```python
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, horizon_label in zip(axes, HORIZONS):
    top = (
        importance_df[importance_df["horizon"].eq(horizon_label)]
        .nsmallest(12, "rank")
        .sort_values("normalized_gain")
    )
    sns.barplot(data=top, x="normalized_gain", y="feature", ax=ax, color="#4C72B0")
    ax.set_title(f"{horizon_label} aggregate LightGBM gain")
    ax.set_xlabel("Normalized gain")
    ax.set_ylabel("")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "lgbm_feature_importance.png", dpi=160)
plt.show()
```

Use deterministic SHAP samples and exactly six representative estimators:

```python
SHAP_STEPS = {"1h": [1, 4], "24h": [1, 24, 48, 96]}
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
flat_axes = axes.ravel()
panel = 0
for horizon_label, steps in SHAP_STEPS.items():
    artifact = artifacts[horizon_label]
    X_sample = artifact["X_test"].sample(
        n=min(500, len(artifact["X_test"])), random_state=42
    )
    for step in steps:
        model = artifact["lgbm_model"].models[step - 1]
        explainer = shap.TreeExplainer(model.booster_)
        shap_values = explainer.shap_values(X_sample)
        plt.sca(flat_axes[panel])
        shap.summary_plot(
            shap_values,
            X_sample,
            max_display=10,
            show=False,
            plot_size=None,
        )
        flat_axes[panel].set_title(f"{horizon_label}, step {step}")
        panel += 1
fig.suptitle("LightGBM SHAP summaries by forecast lead", y=1.01)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "lgbm_shap_summary.png", dpi=160, bbox_inches="tight")
plt.show()
```

Add markdown stating that gain and SHAP describe model associations, not causal effects.

- [ ] **Step 5: Update the notebook index and execute the notebook**

Change `notebooks/README.md` to list `03_ml_models.ipynb` as implemented after notebook 02. Then run:

```powershell
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace notebooks\03_ml_models.ipynb --ExecutePreprocessor.timeout=3600
```

Expected: exit code 0 with all code cells executed and no error output.

- [ ] **Step 6: Verify artifact schemas and sizes**

Run:

```powershell
@'
import pandas as pd
from pathlib import Path

tables = Path("reports/tables")
figures = Path("reports/figures")
metrics = pd.read_csv(tables / "ml_model_metrics.csv")
steps = pd.read_csv(tables / "ml_metrics_by_step.csv")
search = pd.read_csv(tables / "lgbm_parameter_search.csv")
importance = pd.read_csv(tables / "lgbm_feature_importance.csv")
assert len(metrics) == 8
assert len(steps) == 400
assert len(search) == 6 and search.groupby("horizon")["selected"].sum().eq(1).all()
assert len(importance) == 40
for name in (
    "lgbm_forecast_examples.png",
    "lgbm_mae_by_step.png",
    "lgbm_feature_importance.png",
    "lgbm_shap_summary.png",
):
    assert (figures / name).stat().st_size > 10_000
print(metrics.to_string(index=False))
'@ | .\.venv\Scripts\python.exe -
```

Expected: all assertions pass and eight metric rows print.

- [ ] **Step 7: Visually inspect all four PNG files**

Open each generated PNG. Confirm nonblank plots, readable legends and labels, six visible SHAP panels, and no clipping or incoherent overlap. Fix notebook plotting code and re-execute if any check fails.

- [ ] **Step 8: Commit Task 3**

```powershell
git add notebooks\03_ml_models.ipynb notebooks\README.md reports\tables\ml_model_metrics.csv reports\tables\ml_metrics_by_step.csv reports\tables\lgbm_parameter_search.csv reports\tables\lgbm_feature_importance.csv reports\figures\lgbm_forecast_examples.png reports\figures\lgbm_mae_by_step.png reports\figures\lgbm_feature_importance.png reports\figures\lgbm_shap_summary.png
git commit -m "feat: run interpretable LightGBM forecast experiment"
```

---

### Task 4: Portfolio Documentation And Final Verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: executed Task 3 CSVs and PNGs.
- Produces: a measured Week 3 portfolio narrative and final verified GitHub state.

- [ ] **Step 1: Read measured results before writing**

Run:

```powershell
@'
import pandas as pd
m = pd.read_csv("reports/tables/ml_model_metrics.csv")
s = pd.read_csv("reports/tables/lgbm_parameter_search.csv")
print(m.to_markdown(index=False))
print(s[s["selected"]].to_markdown(index=False))
'@ | .\.venv\Scripts\python.exe -
```

Expected: eight model rows and one selected LightGBM candidate per horizon. Use these values directly; do not assume LightGBM wins.

- [ ] **Step 2: Update README with the measured Week 3 result**

Add a concise `## Interpretable Machine Learning Results` section that includes:

- The direct per-step LightGBM architecture and validation-only selection.
- A Markdown table rounded to two decimals for horizon, model, MAE, RMSE, MAPE, endpoint MAE, and LightGBM improvement versus Seasonal Naive.
- The selected candidate and validation MAE for each horizon.
- An honest statement naming the measured overall-MAE winner for each horizon.
- A non-causal interpretation of the top gain and SHAP features.
- The four Task 3 figure links.

Mark `### Week 3: Machine Learning Models (Completed)` and replace its planned bullets with the completed LightGBM, baseline comparison, gain importance, and SHAP work. Update Quick Start and Repository Structure for notebook 03, `src/ml_models.py`, `tests/test_ml_models.py`, and the eight generated artifacts. Do not add Random Forest or XGBoost claims because they were alternatives, not implemented experiments.

- [ ] **Step 3: Run complete verification**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -v
.\.venv\Scripts\python.exe -m compileall -q src
git diff --check
git status --short
```

Expected: every test passes, compilation exits 0, diff check has no errors, and only the intended README change is uncommitted.

- [ ] **Step 4: Commit documentation**

```powershell
git add README.md
git commit -m "docs: report interpretable LightGBM results"
```

- [ ] **Step 5: Push and verify GitHub**

```powershell
git push origin main
git status --short --branch
git log -5 --oneline
git ls-remote origin refs/heads/main
```

Expected: local `main` matches `origin/main`, the worktree is clean, and the remote hash equals local `HEAD`.
