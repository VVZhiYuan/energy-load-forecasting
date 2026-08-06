# Latest Forecast CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a leakage-safe CLI that auto-loads UCI-wide or `timestamp,load` data, selects the best forecasting family, predicts the next 1 hour or 24 hours with uncertainty, and writes portfolio-ready reports.

**Architecture:** Extend the existing data and LightGBM modules at their current ownership boundaries, add one orchestration module with structured run results, and keep the root CLI thin. The workflow first produces an honest 70/15/15 backtest, then refits the validation-selected winner on all labeled origins for the operational forecast; reporting consumes the structured result and stages all five artifacts before publication.

**Tech Stack:** Python 3, pandas, NumPy, scikit-learn Ridge, LightGBM 4.7+, python-holidays, Matplotlib, Plotly, pytest.

## Global Constraints

- Preserve continuous 15-minute timestamps and horizons `1h=4` and `24h=96`.
- Select point models by validation MAE only; test metrics never alter the winner.
- Standard mode uses LightGBM `medium` for 1h and `small` for 24h; `--search` evaluates `small`, `medium`, and `large`.
- Quantile output is P10/P50/P90 when LightGBM wins; non-LightGBM winners use validation-residual quantiles.
- Publish `forecast.csv`, `model_comparison.csv`, `summary.json`, `forecast.png`, and self-contained `forecast.html`.
- Custom input has no implicit holiday country; bundled UCI input defaults to `PT`.
- Reject missing intervals and missing/non-finite loads; do not silently impute.
- Allow negative net load but record its count.
- Use fixed random seed `42`; do not persist model binaries.
- Keep raw UCI data untracked and preserve unrelated notebook/image worktree changes.
- Run on Windows PowerShell with `lightgbm>=4.7.0` and existing requirements.

## File Map

- Modify `src/data_loader.py`: detect delimiter/schema, normalize one load series, and validate input.
- Modify `src/features.py`: support explicit disabled holidays and reject unsupported requested countries.
- Modify `src/ml_models.py`: expose fixed-iteration refit and direct quantile LightGBM.
- Create `src/inference.py`: compare model families, choose by validation MAE, refit, and assemble forecasts.
- Create `src/reporting.py`: serialize tables/metadata and render PNG/HTML through staged publication.
- Create `predict_latest.py`: parse arguments, resolve defaults, report progress, and return clean exit codes.
- Create `tests/test_data_loader.py`, `tests/test_inference.py`, `tests/test_reporting.py`, and `tests/test_predict_latest.py`.
- Modify `tests/test_features.py` and `tests/test_ml_models.py` for new behavior.
- Modify `README.md` only after real measured outputs exist.
- Create verified files below `reports/predictions/MT_252/{1h,24h}/` during final integration.

---

### Task 1: Forecast Input And Holiday Contract

**Files:**
- Modify: `src/data_loader.py`
- Modify: `src/features.py`
- Create: `tests/test_data_loader.py`
- Modify: `tests/test_features.py`

**Interfaces:**
- Produces: `LoadedLoadSeries(series, input_format, source_label, meter, negative_load_count)`.
- Produces: `load_forecast_series(path: str | Path, meter: str | None = None) -> LoadedLoadSeries`.
- Changes: `add_holiday_feature(df: pd.DataFrame, timestamp_col: str = "timestamp", country: str | None = "PT") -> pd.DataFrame`.
- Changes: `build_baseline_features(series: pd.Series, country: str | None = "PT") -> pd.DataFrame`.
- Consumed by: Task 3 orchestration and Task 5 CLI.

- [ ] **Step 1: Write failing loader tests**

```python
# tests/test_data_loader.py
import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_forecast_series


def test_loads_two_column_forecast_csv(tmp_path):
    path = tmp_path / "building.csv"
    pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=8, freq="15min"),
        "load": np.arange(8, dtype=float),
    }).to_csv(path, index=False)

    loaded = load_forecast_series(path)

    assert loaded.input_format == "long"
    assert loaded.meter is None
    assert loaded.source_label == "building"
    assert loaded.series.name == "load"


def test_loads_uci_style_wide_file(tmp_path):
    path = tmp_path / "meters.txt"
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=8, freq="15min"),
        "MT_001": np.arange(8, dtype=float) + 0.5,
        "MT_002": np.arange(8, dtype=float) + 2.5,
    })
    frame.to_csv(path, sep=";", decimal=",", index=False)

    loaded = load_forecast_series(path, meter="MT_002")

    assert loaded.input_format == "wide"
    assert loaded.meter == "MT_002"
    np.testing.assert_allclose(loaded.series, frame["MT_002"])


def test_rejects_missing_interval(tmp_path):
    path = tmp_path / "bad.csv"
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=8, freq="15min"),
        "load": np.arange(8, dtype=float),
    }).drop(index=3)
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="continuous 15-minute"):
        load_forecast_series(path)


def test_rejects_unknown_wide_meter(tmp_path):
    path = tmp_path / "meters.csv"
    pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=8, freq="15min"),
        "MT_001": np.arange(8, dtype=float),
    }).to_csv(path, index=False)

    with pytest.raises(ValueError, match="MT_999"):
        load_forecast_series(path, meter="MT_999")
```

- [ ] **Step 2: Run the loader tests and verify the missing interface failure**

Run: `python -m pytest tests/test_data_loader.py -v`

Expected: collection fails because `load_forecast_series` does not exist.

- [ ] **Step 3: Add structured input loading and validation**

Add these public definitions to `src/data_loader.py`, retaining the existing UCI loader and long-format helper:

```python
import csv
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LoadedLoadSeries:
    series: pd.Series
    input_format: str
    source_label: str
    meter: str | None
    negative_load_count: int


def _read_forecast_table(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
        sample = stream.read(8192)
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=";,\t").delimiter
    except csv.Error as exc:
        raise ValueError(f"Could not detect a supported delimiter in {path}.") from exc
    decimal = "," if delimiter == ";" else "."
    return pd.read_csv(path, sep=delimiter, decimal=decimal)


def _validate_normalized_series(series: pd.Series) -> pd.Series:
    if not isinstance(series.index, pd.DatetimeIndex) or series.index.hasnans:
        raise ValueError("timestamps must be parseable and present.")
    series = series.sort_index()
    if not series.index.is_unique:
        raise ValueError("timestamps must be unique.")
    expected = pd.date_range(series.index[0], series.index[-1], freq="15min")
    if not series.index.equals(expected):
        raise ValueError("timestamps must have a continuous 15-minute frequency.")
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("load must contain only finite numeric values.")
    numeric.name = "load"
    return numeric


def load_forecast_series(path: str | Path, meter: str | None = None) -> LoadedLoadSeries:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Forecast input file not found: {path}")
    frame = _read_forecast_table(path)
    serialized_index_columns = []
    for column in frame.columns:
        if not str(column).startswith("Unnamed:"):
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().all() and np.array_equal(numeric.to_numpy(), np.arange(len(frame))):
            serialized_index_columns.append(column)
    frame = frame.drop(columns=serialized_index_columns)
    names = {str(column).strip().lower(): column for column in frame.columns}

    if set(names) == {"timestamp", "load"}:
        timestamp_column = names["timestamp"]
        value_column = names["load"]
        input_format = "long"
        selected_meter = None
    else:
        if frame.shape[1] < 2:
            raise ValueError("Input must be timestamp,load or a timestamp plus meter columns.")
        timestamp_column = frame.columns[0]
        if meter is None:
            raise ValueError("--meter is required for wide input.")
        if meter not in frame.columns:
            raise ValueError(f"Meter {meter!r} was not found in the input columns.")
        value_column = meter
        input_format = "wide"
        selected_meter = meter

    timestamps = pd.to_datetime(frame[timestamp_column], errors="coerce")
    series = pd.Series(frame[value_column].to_numpy(), index=timestamps, name="load")
    series = _validate_normalized_series(series)
    return LoadedLoadSeries(
        series=series,
        input_format=input_format,
        source_label=selected_meter or path.stem,
        meter=selected_meter,
        negative_load_count=int((series < 0).sum()),
    )
```

- [ ] **Step 4: Add failing holiday-mode tests**

```python
# append to tests/test_features.py
import pytest
from src.features import add_holiday_feature


def test_disabled_holiday_country_produces_zero_flag():
    frame = make_series(8).to_frame()
    result = add_holiday_feature(frame, country=None)
    assert result["is_holiday"].eq(0).all()


def test_hong_kong_holiday_is_detected():
    index = pd.DatetimeIndex(["2025-01-01 00:00:00"])
    result = add_holiday_feature(pd.DataFrame({"load": [1.0]}, index=index), country="HK")
    assert result["is_holiday"].item() == 1


def test_unsupported_requested_country_fails():
    frame = make_series(8).to_frame()
    with pytest.raises(ValueError, match="holiday country"):
        add_holiday_feature(frame, country="NOT_A_COUNTRY")
```

- [ ] **Step 5: Make holiday behavior explicit**

Replace silent fallback in `add_holiday_feature` with this contract and change `build_baseline_features` to accept `country: str | None = "PT"`:

```python
def add_holiday_feature(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    country: str | None = "PT",
) -> pd.DataFrame:
    out = df.copy()
    ts = pd.Series(
        pd.to_datetime(out[timestamp_col] if timestamp_col in out.columns else out.index),
        index=out.index,
    )
    if country is None:
        out["is_holiday"] = 0
        return out
    if holidays is None:
        raise RuntimeError("python-holidays is required when a holiday country is requested.")
    supported = holidays.list_supported_countries()
    if country.upper() not in supported:
        raise ValueError(f"Unsupported holiday country: {country}")
    calendar = holidays.country_holidays(country.upper())
    out["is_holiday"] = ts.dt.date.map(lambda date: int(date in calendar)).to_numpy()
    return out
```

- [ ] **Step 6: Run focused and existing feature tests**

Run: `python -m pytest tests/test_data_loader.py tests/test_features.py tests/test_forecasting.py -v`

Expected: all tests pass.

- [ ] **Step 7: Commit the input contract**

```powershell
git add src/data_loader.py src/features.py tests/test_data_loader.py tests/test_features.py
git commit -m "feat: normalize forecast input data"
```

---

### Task 2: LightGBM Operational Refit And Quantiles

**Files:**
- Modify: `src/ml_models.py`
- Modify: `tests/test_ml_models.py`

**Interfaces:**
- Changes: `DirectLightGBMForecaster` gains `objective: str` and `alpha: float | None`.
- Produces: `fit_direct_quantile_lightgbm(X_train, y_train, X_val, y_val, candidate, quantiles=(0.1, 0.5, 0.9), parallel_jobs=-1) -> dict[float, DirectLightGBMForecaster]`.
- Produces: `refit_direct_lightgbm(X_all, y_all, fitted, parallel_jobs=-1) -> DirectLightGBMForecaster`.
- Consumed by: Task 3 after point-family selection.

- [ ] **Step 1: Write failing quantile and refit tests**

```python
# append to tests/test_ml_models.py imports
from src.ml_models import fit_direct_quantile_lightgbm, refit_direct_lightgbm


def test_quantile_lightgbm_returns_requested_forecasters():
    X_train, y_train, X_val, y_val = make_frames(4)
    fitted = fit_direct_quantile_lightgbm(
        X_train, y_train, X_val, y_val,
        candidate=SMOKE_CANDIDATE,
        quantiles=(0.1, 0.5, 0.9),
        parallel_jobs=1,
    )
    assert tuple(fitted) == (0.1, 0.5, 0.9)
    assert all(model.predict(X_val).shape == (len(X_val), 4) for model in fitted.values())
    assert fitted[0.1].objective == "quantile"
    assert fitted[0.1].alpha == 0.1


def test_refit_uses_every_supplied_origin():
    X_train, y_train, X_val, y_val = make_frames(4)
    fitted = fit_direct_lightgbm(
        X_train, y_train, X_val, y_val, SMOKE_CANDIDATE, parallel_jobs=1
    )
    X_all = pd.concat([X_train, X_val])
    y_all = pd.concat([y_train, y_val])
    refitted = refit_direct_lightgbm(X_all, y_all, fitted, parallel_jobs=1)
    assert refitted.predict(X_all.iloc[[-1]]).shape == (1, 4)
    assert all(model.n_features_in_ == X_all.shape[1] for model in refitted.models)


@pytest.mark.parametrize("quantiles", [(0.5, 0.1), (0.0, 0.5, 0.9), (0.1, 1.0)])
def test_quantile_lightgbm_rejects_invalid_quantiles(quantiles):
    X_train, y_train, X_val, y_val = make_frames(4)
    with pytest.raises(ValueError, match="quantiles"):
        fit_direct_quantile_lightgbm(
            X_train, y_train, X_val, y_val,
            SMOKE_CANDIDATE, quantiles=quantiles, parallel_jobs=1,
        )
```

- [ ] **Step 2: Run tests and verify missing-function failures**

Run: `python -m pytest tests/test_ml_models.py -v`

Expected: collection fails for the two new imports.

- [ ] **Step 3: Refactor model creation without changing point behavior**

Add `objective` and `alpha` defaults to the forecaster and centralize estimator creation:

```python
@dataclass(frozen=True)
class DirectLightGBMForecaster:
    models: tuple[LGBMRegressor, ...]
    feature_names: tuple[str, ...]
    candidate: LightGBMCandidate
    objective: str = "regression_l1"
    alpha: float | None = None


def _make_estimator(
    candidate: LightGBMCandidate,
    *,
    objective: str,
    alpha: float | None,
    n_estimators: int | None = None,
) -> LGBMRegressor:
    kwargs = {
        "objective": objective,
        "random_state": 42,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "n_jobs": 1,
        "verbosity": -1,
        "num_leaves": candidate.num_leaves,
        "learning_rate": candidate.learning_rate,
        "n_estimators": n_estimators or candidate.n_estimators,
        "min_child_samples": candidate.min_child_samples,
        "reg_lambda": candidate.reg_lambda,
    }
    if alpha is not None:
        kwargs["alpha"] = alpha
    return LGBMRegressor(**kwargs)
```

Change `_fit_one_step` and `fit_direct_lightgbm` to pass `objective="regression_l1"` and `alpha=None` through this factory while preserving current validation callbacks and return values.

- [ ] **Step 4: Add quantile fitting and fixed-iteration all-history refit**

```python
def fit_direct_quantile_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    candidate: LightGBMCandidate,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    parallel_jobs: int = -1,
) -> dict[float, DirectLightGBMForecaster]:
    quantiles = tuple(float(value) for value in quantiles)
    if not quantiles or tuple(sorted(set(quantiles))) != quantiles or not all(0 < value < 1 for value in quantiles):
        raise ValueError("quantiles must be unique, increasing values between 0 and 1.")
    _validate_supervised_frames(X_train, y_train, X_val, y_val)
    result = {}
    for quantile in quantiles:
        models = Parallel(n_jobs=parallel_jobs, prefer="threads")(
            delayed(_fit_one_step)(
                X_train, y_train.iloc[:, step], X_val, y_val.iloc[:, step],
                candidate, objective="quantile", alpha=quantile,
            )
            for step in range(y_train.shape[1])
        )
        result[quantile] = DirectLightGBMForecaster(
            models=tuple(models), feature_names=tuple(X_train.columns),
            candidate=candidate, objective="quantile", alpha=quantile,
        )
    return result


def refit_direct_lightgbm(
    X_all: pd.DataFrame,
    y_all: pd.DataFrame,
    fitted: DirectLightGBMForecaster,
    parallel_jobs: int = -1,
) -> DirectLightGBMForecaster:
    _validate_frame("X_all", X_all)
    _validate_frame("y_all", y_all)
    if not X_all.index.equals(y_all.index):
        raise ValueError("X_all and y_all indexes must match.")
    if tuple(X_all.columns) != fitted.feature_names or y_all.shape[1] != len(fitted.models):
        raise ValueError("all-history frames must match the fitted forecaster schema.")
    iterations = tuple(
        max(1, int(getattr(model, "best_iteration_", 0) or fitted.candidate.n_estimators))
        for model in fitted.models
    )

    def fit_step(step: int) -> LGBMRegressor:
        model = _make_estimator(
            fitted.candidate,
            objective=fitted.objective,
            alpha=fitted.alpha,
            n_estimators=iterations[step],
        )
        model.fit(X_all, y_all.iloc[:, step])
        return model

    models = Parallel(n_jobs=parallel_jobs, prefer="threads")(
        delayed(fit_step)(step) for step in range(y_all.shape[1])
    )
    return DirectLightGBMForecaster(
        models=tuple(models), feature_names=fitted.feature_names,
        candidate=fitted.candidate, objective=fitted.objective, alpha=fitted.alpha,
    )
```

- [ ] **Step 5: Run LightGBM and full unit tests**

Run: `python -m pytest tests/test_ml_models.py -v`

Expected: all LightGBM tests pass, including existing reproducibility and tie tests.

Run: `python -m pytest -q`

Expected: all project tests pass.

- [ ] **Step 6: Commit probabilistic model support**

```powershell
git add src/ml_models.py tests/test_ml_models.py
git commit -m "feat: add probabilistic LightGBM refit"
```

---

### Task 3: Leakage-Safe Model Selection And Latest Forecast

**Files:**
- Create: `src/inference.py`
- Create: `tests/test_inference.py`

**Interfaces:**
- Produces: `ForecastRun(observed, forecast, model_comparison, summary)`.
- Produces: `run_latest_forecast(loaded, horizon_label, holiday_country, search=False, parallel_jobs=-1, progress=None) -> ForecastRun`.
- Produces: `ordered_quantiles(values) -> tuple[np.ndarray, int]` for report-safe intervals.
- Consumes: `LoadedLoadSeries`, existing baseline/metric/split functions, and Task 2 LightGBM functions.
- Consumed by: Task 4 reporting and Task 5 CLI.

- [ ] **Step 1: Write failing orchestration tests with patched model runners**

```python
# tests/test_inference.py
import numpy as np
import pandas as pd
import pytest

from src.data_loader import LoadedLoadSeries
from src.inference import _select_row, ordered_quantiles, run_latest_forecast
from src.ml_models import LightGBMCandidate


def make_loaded(length=3000):
    index = pd.date_range("2024-01-01", periods=length, freq="15min")
    values = 100 + 10 * np.sin(np.arange(length) * 2 * np.pi / 96)
    return LoadedLoadSeries(
        series=pd.Series(values, index=index, name="load"),
        input_format="long", source_label="fixture", meter=None,
        negative_load_count=0,
    )


class FakeDirectForecaster:
    def __init__(self, horizon, candidate=None):
        self.candidate = candidate or LightGBMCandidate("small", 3, 0.1, 3, 2, 0.0)
        self.horizon = horizon

    def predict(self, features):
        return np.zeros((len(features), self.horizon), dtype=float)


@pytest.fixture
def patch_fast_lightgbm(monkeypatch):
    def fake_fit(X_train, y_train, X_val, y_val, candidate, parallel_jobs=-1):
        return FakeDirectForecaster(y_train.shape[1], candidate)

    monkeypatch.setattr("src.inference.fit_direct_lightgbm", fake_fit)


def test_ordered_quantiles_corrects_crossing():
    values = np.array([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
    corrected, count = ordered_quantiles(values)
    np.testing.assert_allclose(corrected, [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    assert count == 1


def test_latest_forecast_has_four_future_rows(patch_fast_lightgbm):
    loaded = make_loaded()
    run = run_latest_forecast(
        loaded, horizon_label="1h", holiday_country=None,
        search=False, parallel_jobs=1,
    )
    assert len(run.forecast) == 4
    assert run.forecast.index[0] == loaded.series.index[-1] + pd.Timedelta(minutes=15)
    assert run.forecast.index[-1] == loaded.series.index[-1] + pd.Timedelta(hours=1)
    assert set(["prediction", "p10", "p50", "p90"]).issubset(run.forecast.columns)


def test_winner_is_lowest_validation_mae_not_lowest_test_mae():
    table = pd.DataFrame([
        {"model": "Naive", "validation_mae": 1.0, "test_mae": 10.0},
        {"model": "LightGBM", "validation_mae": 2.0, "test_mae": 0.1},
    ])
    selected = _select_row(table)
    assert selected.loc[selected["selected"], "model"].item() == "Naive"


def test_search_passes_all_declared_lightgbm_candidates(monkeypatch):
    captured = {}

    def fake_select(X_train, y_train, X_val, y_val, candidates, parallel_jobs=-1):
        candidates = tuple(candidates)
        captured["names"] = [candidate.name for candidate in candidates]
        search = pd.DataFrame({
            "candidate": captured["names"],
            "validation_MAE": [3.0, 2.0, 1.0],
            "selected": [False, False, True],
        })
        return FakeDirectForecaster(y_train.shape[1], candidates[-1]), search

    monkeypatch.setattr("src.inference.select_lightgbm_candidate", fake_select)
    run = run_latest_forecast(
        make_loaded(), horizon_label="1h", holiday_country=None,
        search=True, parallel_jobs=1,
    )
    assert captured["names"] == ["small", "medium", "large"]
    assert len(run.summary["lightgbm_search"]) == 3
```

- [ ] **Step 2: Run tests and verify the missing module failure**

Run: `python -m pytest tests/test_inference.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'src.inference'`.

- [ ] **Step 3: Add run-result types and deterministic quantile correction**

```python
# src/inference.py
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
```

- [ ] **Step 4: Implement feature alignment, family scoring, and validation-only selection**

Use this fixed model order and comparison schema in `src/inference.py`:

```python
MODEL_ORDER = {"Naive": 0, "Seasonal Naive": 1, "Ridge": 2, "LightGBM": 3}
STANDARD_CANDIDATE = {
    "1h": next(candidate for candidate in DEFAULT_LIGHTGBM_CANDIDATES if candidate.name == "medium"),
    "24h": next(candidate for candidate in DEFAULT_LIGHTGBM_CANDIDATES if candidate.name == "small"),
}


def _score_row(model, configuration, y_val, val_prediction, y_test, test_prediction, seconds):
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
```

`run_latest_forecast` must build features and targets once, align complete target rows, call `split_supervised_by_time`, create validation/test predictions for all four families on identical indexes, and retain the fitted Ridge and LightGBM objects needed for refit. In search mode call `select_lightgbm_candidate` with all candidates; otherwise call `fit_direct_lightgbm` with only `STANDARD_CANDIDATE[horizon_label]`. Store candidate-search rows as `search_results.to_dict(orient="records")` in `summary["lightgbm_search"]` while `model_comparison` has one family row per model.

- [ ] **Step 5: Implement all-history refit and interval strategy**

After selecting the comparison row:

```python
complete_targets = targets.dropna()
all_index = features.index.intersection(complete_targets.index)
X_all = features.loc[all_index]
y_all = complete_targets.loc[all_index]
latest_X = features.iloc[[-1]]

if winner == "Naive":
    prediction = naive_forecast([loaded.series.iloc[-1]], horizon)[0]
elif winner == "Seasonal Naive":
    prediction = seasonal_naive_forecast(
        loaded.series, pd.DatetimeIndex([loaded.series.index[-1]]), horizon
    )[0]
elif winner == "Ridge":
    ridge_model.fit(X_all, y_all)
    prediction = ridge_model.predict(latest_X)[0]
else:
    point_model = refit_direct_lightgbm(X_all, y_all, lightgbm_model, parallel_jobs)
    prediction = point_model.predict(latest_X)[0]
```

If LightGBM wins, fit quantiles on train/validation with the selected candidate, refit each quantile forecaster on `X_all/y_all`, and stack the latest `[0.1, 0.5, 0.9]` predictions. Otherwise calculate `residuals = y_validation.to_numpy() - selected_validation_prediction`, take `np.quantile(residuals, [0.1, 0.5, 0.9], axis=0).T`, and add those values to the operational point forecast. Pass the stacked values through `ordered_quantiles`.

Build the output with:

```python
future_index = pd.date_range(
    loaded.series.index[-1] + pd.Timedelta(minutes=15),
    periods=horizon,
    freq="15min",
    name="forecast_timestamp",
)
forecast = pd.DataFrame({
    "step": np.arange(1, horizon + 1),
    "prediction": prediction,
    "p10": quantiles[:, 0],
    "p50": quantiles[:, 1],
    "p90": quantiles[:, 2],
    "point_model": winner,
    "interval_method": interval_method,
}, index=future_index)
```

The summary must include source metadata, observed start/end, forecast origin, horizon label/steps, holiday country, search mode, selected model/configuration, split row counts, quantile correction count, interval method, negative-load count, random seed, and runtime. Store every timestamp with `.isoformat()` so the mapping is directly JSON serializable.

- [ ] **Step 6: Run orchestration tests and add 24-hour timestamp coverage**

Add this parametrized test:

```python
@pytest.mark.parametrize("label,rows", [("1h", 4), ("24h", 96)])
def test_latest_forecast_row_count_matches_horizon(label, rows, patch_fast_lightgbm):
    run = run_latest_forecast(
        make_loaded(), horizon_label=label, holiday_country=None,
        search=False, parallel_jobs=1,
    )
    assert len(run.forecast) == rows
    assert run.forecast["step"].tolist() == list(range(1, rows + 1))
```

Then run:

Run: `python -m pytest tests/test_inference.py -v`

Expected: all tests pass without training production-size LightGBM collections.

- [ ] **Step 7: Commit orchestration**

```powershell
git add src/inference.py tests/test_inference.py
git commit -m "feat: select and refit latest forecast model"
```

---

### Task 4: Staged CSV, JSON, PNG, And Interactive HTML Reports

**Files:**
- Create: `src/reporting.py`
- Create: `tests/test_reporting.py`

**Interfaces:**
- Produces: `write_forecast_artifacts(run: ForecastRun, output_dir: str | Path) -> dict[str, Path]`.
- Consumes: Task 3 `ForecastRun` only; it does not call training code.
- Consumed by: Task 5 CLI.

- [ ] **Step 1: Write a failing artifact-contract test**

```python
# tests/test_reporting.py
import json

import numpy as np
import pandas as pd

from src.inference import ForecastRun
from src.reporting import write_forecast_artifacts


def make_run():
    observed_index = pd.date_range("2025-01-01", periods=700, freq="15min")
    forecast_index = pd.date_range(observed_index[-1] + pd.Timedelta(minutes=15), periods=4, freq="15min", name="forecast_timestamp")
    forecast = pd.DataFrame({
        "step": [1, 2, 3, 4], "prediction": [10.0, 11.0, 12.0, 13.0],
        "p10": [9.0, 10.0, 11.0, 12.0], "p50": [10.0, 11.0, 12.0, 13.0],
        "p90": [11.0, 12.0, 13.0, 14.0], "point_model": "Ridge",
        "interval_method": "residual_calibration",
    }, index=forecast_index)
    comparison = pd.DataFrame([{
        "model": "Ridge", "configuration": "alpha=0.1",
        "validation_mae": 1.0, "validation_rmse": 1.2,
        "test_mae": 1.1, "test_rmse": 1.3,
        "selected": True, "training_seconds": 0.1,
    }])
    return ForecastRun(
        observed=pd.Series(np.arange(700), index=observed_index, name="load"),
        forecast=forecast, model_comparison=comparison,
        summary={"source_label": "fixture", "horizon": "1h", "selected_model": "Ridge"},
    )


def test_writes_complete_report_set(tmp_path):
    output = tmp_path / "report"
    paths = write_forecast_artifacts(make_run(), output)
    assert set(paths) == {"forecast_csv", "comparison_csv", "summary_json", "png", "html"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())
    assert len(pd.read_csv(paths["forecast_csv"])) == 4
    assert json.loads(paths["summary_json"].read_text(encoding="utf-8"))["selected_model"] == "Ridge"
    html = paths["html"].read_text(encoding="utf-8")
    assert "plotly" in html.lower()
    assert "Ridge" in html
```

- [ ] **Step 2: Run the test and verify the missing module failure**

Run: `python -m pytest tests/test_reporting.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'src.reporting'`.

- [ ] **Step 3: Implement staged table and metadata serialization**

Create `src/reporting.py` with `Path`, `json`, `shutil`, `tempfile`, Matplotlib, Plotly, pandas, and package-version imports. `write_forecast_artifacts` must create a staging directory beside the final directory, write `forecast.csv` with the named index, write `model_comparison.csv` without an index, enrich `summary.json` with package versions and artifact names, and validate all tabular files by reading them back before publication.

Use this publication order:

```python
names = {
    "forecast_csv": "forecast.csv",
    "comparison_csv": "model_comparison.csv",
    "png": "forecast.png",
    "html": "forecast.html",
    "summary_json": "summary.json",
}
output_dir.mkdir(parents=True, exist_ok=True)
for key in ("forecast_csv", "comparison_csv", "png", "html", "summary_json"):
    source = staging_dir / names[key]
    destination = output_dir / names[key]
    source.replace(destination)
```

Wrap staging in `try/finally` and remove any remaining staging directory with `shutil.rmtree(staging_dir, ignore_errors=True)`. Publishing `summary.json` last is the completion marker.

- [ ] **Step 4: Render static and interactive trajectories**

For both plots, show the last `min(len(observed), 7 * 96)` history points, the operational point forecast, and a P10-P90 filled band. Use a constrained Matplotlib layout, explicit axis labels `Timestamp` and `Load`, and title `<source> <horizon> Latest Load Forecast`.

Build the Plotly report with one trajectory figure plus a model-comparison HTML table:

```python
trajectory_html = figure.to_html(
    full_html=False,
    include_plotlyjs=True,
    config={"displaylogo": False, "responsive": True},
)
leaderboard_html = run.model_comparison.sort_values("validation_mae").to_html(
    index=False, float_format=lambda value: f"{value:.4f}", border=0,
)
html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Latest Load Forecast</title></head><body>
<main><h1>Latest Load Forecast</h1>{trajectory_html}<h2>Model Comparison</h2>{leaderboard_html}</main>
</body></html>"""
```

Write UTF-8 HTML and assert the PNG dimensions are nonzero, forecast columns are finite, quantiles are ordered, and HTML contains both `plotly` and the selected model before publishing.

- [ ] **Step 5: Run report and full tests**

Run: `python -m pytest tests/test_reporting.py -v`

Expected: complete report set test passes.

Run: `python -m pytest -q`

Expected: all project tests pass.

- [ ] **Step 6: Commit reporting**

```powershell
git add src/reporting.py tests/test_reporting.py
git commit -m "feat: generate latest forecast reports"
```

---

### Task 5: Root Prediction CLI And Progress Output

**Files:**
- Create: `predict_latest.py`
- Create: `tests/test_predict_latest.py`
- Modify: `src/config.py`

**Interfaces:**
- Produces: `build_parser() -> argparse.ArgumentParser`.
- Produces: `main(argv: list[str] | None = None) -> int`.
- Consumes: Tasks 1, 3, and 4 public interfaces.

- [ ] **Step 1: Write failing parser/default/error tests**

```python
# tests/test_predict_latest.py
from pathlib import Path

import predict_latest


def test_parser_requires_supported_horizon():
    args = predict_latest.build_parser().parse_args(["--horizon", "24h"])
    assert args.horizon == "24h"
    assert args.search is False


def test_default_input_uses_uci_meter_and_portugal_holidays(monkeypatch):
    captured = {}

    def fake_load(path, meter):
        captured["meter"] = meter
        return type("Loaded", (), {"source_label": "MT_252"})()

    def fake_run(loaded, horizon_label, holiday_country, **kwargs):
        captured["country"] = holiday_country
        return object()

    monkeypatch.setattr(predict_latest, "load_forecast_series", fake_load)
    monkeypatch.setattr(predict_latest, "run_latest_forecast", fake_run)
    monkeypatch.setattr(predict_latest, "write_forecast_artifacts", lambda run, output: {"summary_json": Path("summary.json")})
    assert predict_latest.main(["--horizon", "1h"]) == 0
    assert captured == {"meter": "MT_252", "country": "PT"}


def test_custom_input_has_no_implicit_holiday_country(monkeypatch, tmp_path):
    path = tmp_path / "custom.csv"
    path.write_text("timestamp,load\n2025-01-01,1\n", encoding="utf-8")
    captured = {}

    def fake_run(loaded, horizon_label, holiday_country, **kwargs):
        captured["country"] = holiday_country
        return object()

    monkeypatch.setattr(predict_latest, "load_forecast_series", lambda path, meter: object())
    monkeypatch.setattr(predict_latest, "run_latest_forecast", fake_run)
    monkeypatch.setattr(
        predict_latest,
        "write_forecast_artifacts",
        lambda run, output: {"summary_json": Path("summary.json")},
    )
    assert predict_latest.main(["--input", str(path), "--horizon", "1h"]) == 0
    assert captured["country"] is None


def test_user_input_error_returns_two(monkeypatch, capsys):
    monkeypatch.setattr(
        predict_latest, "load_forecast_series",
        lambda path, meter: (_ for _ in ()).throw(ValueError("bad timestamps")),
    )
    code = predict_latest.main(["--horizon", "1h"])
    assert code == 2
    assert "bad timestamps" in capsys.readouterr().err
```

- [ ] **Step 2: Run tests and verify the missing module failure**

Run: `python -m pytest tests/test_predict_latest.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'predict_latest'`.

- [ ] **Step 3: Add prediction output configuration**

Add to `src/config.py`:

```python
PREDICTIONS_DIR = REPORTS_DIR / "predictions"
```

- [ ] **Step 4: Implement the thin CLI**

Create `predict_latest.py` with this command contract:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import PREDICTIONS_DIR, PROJECT_ROOT
from src.data_loader import get_default_raw_data_path, load_forecast_series
from src.inference import run_latest_forecast
from src.reporting import write_forecast_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Forecast the latest electricity load trajectory.")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--meter")
    parser.add_argument("--horizon", choices=("1h", "24h"), required=True)
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--holiday-country")
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    using_default = args.input is None
    input_path = args.input or get_default_raw_data_path(PROJECT_ROOT)
    meter = args.meter or ("MT_252" if using_default else None)
    country = args.holiday_country or ("PT" if using_default else None)
    try:
        print(f"[1/4] Loading {input_path}")
        loaded = load_forecast_series(input_path, meter=meter)
        if country is None:
            print("Warning: holiday features are disabled for custom input.", file=sys.stderr)
        print("[2/4] Building leakage-safe backtest and selecting model")
        run = run_latest_forecast(
            loaded,
            horizon_label=args.horizon,
            holiday_country=country,
            search=args.search,
            progress=print,
        )
        output_dir = args.output_dir or PREDICTIONS_DIR / loaded.source_label / args.horizon
        print("[3/4] Rendering forecast reports")
        paths = write_forecast_artifacts(run, output_dir)
        print(f"[4/4] Complete: {paths['summary_json']}")
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
```

Unexpected exceptions are not caught so programming failures retain tracebacks. `run_latest_forecast` must call the supplied `progress` callback before each model family, before quantile fitting, and before all-history refit.

- [ ] **Step 5: Run CLI tests and help smoke test**

Run: `python -m pytest tests/test_predict_latest.py -v`

Expected: all CLI tests pass.

Run: `python predict_latest.py --help`

Expected: usage contains all six arguments and exits zero.

- [ ] **Step 6: Run the full suite and commit the CLI**

Run: `python -m pytest -q`

Expected: all tests pass.

```powershell
git add predict_latest.py src/config.py tests/test_predict_latest.py
git commit -m "feat: add latest forecast command"
```

---

### Task 6: Real UCI Forecasts And Portfolio Documentation

**Files:**
- Modify: `README.md`
- Create: `reports/predictions/MT_252/1h/forecast.csv`
- Create: `reports/predictions/MT_252/1h/model_comparison.csv`
- Create: `reports/predictions/MT_252/1h/summary.json`
- Create: `reports/predictions/MT_252/1h/forecast.png`
- Create: `reports/predictions/MT_252/1h/forecast.html`
- Create: `reports/predictions/MT_252/24h/forecast.csv`
- Create: `reports/predictions/MT_252/24h/model_comparison.csv`
- Create: `reports/predictions/MT_252/24h/summary.json`
- Create: `reports/predictions/MT_252/24h/forecast.png`
- Create: `reports/predictions/MT_252/24h/forecast.html`

**Interfaces:**
- Consumes: completed CLI and bundled raw UCI data.
- Produces: measured portfolio evidence and exact README instructions.

- [ ] **Step 1: Run all automated checks before expensive training**

Run: `python -m pytest -v`

Expected: all tests pass.

Run: `python -m compileall -q src predict_latest.py`

Expected: exit code 0 with no output.

- [ ] **Step 2: Run standard 1-hour forecast**

Run: `python predict_latest.py --horizon 1h`

Expected: four progress stages complete and five files appear in `reports/predictions/MT_252/1h/`.

- [ ] **Step 3: Run standard 24-hour forecast**

Run: `python predict_latest.py --horizon 24h`

Expected: five files appear in `reports/predictions/MT_252/24h/`; based on current evidence Seasonal Naive is likely but not forced to win.

- [ ] **Step 4: Exercise full search mode without overwriting the standard example**

Run: `python predict_latest.py --horizon 1h --search --output-dir reports/predictions/MT_252/1h-search`

Expected: summary records `search=true` and all three LightGBM candidates.

- [ ] **Step 5: Verify structured artifact contracts**

Run this PowerShell command:

```powershell
@'
import json
from pathlib import Path
import numpy as np
import pandas as pd

for horizon, rows in (("1h", 4), ("24h", 96)):
    root = Path("reports/predictions/MT_252") / horizon
    forecast = pd.read_csv(root / "forecast.csv", parse_dates=["forecast_timestamp"])
    comparison = pd.read_csv(root / "model_comparison.csv")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert len(forecast) == rows
    assert np.isfinite(forecast[["prediction", "p10", "p50", "p90"]]).all().all()
    assert (forecast["p10"] <= forecast["p50"]).all()
    assert (forecast["p50"] <= forecast["p90"]).all()
    assert comparison["selected"].sum() == 1
    assert summary["horizon"] == horizon
print("ARTIFACTS_OK")
'@ | python -
```

Expected: `ARTIFACTS_OK`.

- [ ] **Step 6: Visually inspect both report formats**

Open both PNG files and both HTML files. Verify nonblank trajectories, visible P10-P90 bands, readable model tables, no clipped labels, and that the prediction begins after the last historical point. Capture desktop screenshots if needed for comparison, but do not commit screenshots outside the designed reports.

- [ ] **Step 7: Update README using measured values only**

Add a `Latest Forecast CLI` section after the machine-learning results with these exact command categories:

````markdown
## Latest Forecast CLI

```powershell
python predict_latest.py --horizon 1h
python predict_latest.py --horizon 24h
python predict_latest.py --horizon 24h --search
python predict_latest.py --input data/custom/hk_building.csv --holiday-country HK --horizon 24h
```

The command selects Naive, Seasonal Naive, Ridge, or LightGBM using validation
MAE, reports untouched chronological test metrics, and then refits the selected
family on all labeled history for the latest forecast. P10, P50, and P90 show
lower, median, and upper empirical load scenarios; they are not guaranteed
coverage bounds.

![Latest 1-hour forecast](reports/predictions/MT_252/1h/forecast.png)

![Latest 24-hour forecast](reports/predictions/MT_252/24h/forecast.png)
````

Immediately after this text, add a Markdown table populated from the two generated `model_comparison.csv` files. State the measured selected model for each horizon, link the corresponding CSV/JSON/HTML files, and retain the existing honest note that LightGBM is not forced to win.

- [ ] **Step 8: Commit verified reports and README**

```powershell
git add README.md reports/predictions/MT_252/1h reports/predictions/MT_252/24h
git commit -m "docs: publish latest load forecast reports"
```

Do not add `reports/predictions/MT_252/1h-search` unless it adds distinct portfolio value after inspection.

---

### Task 7: Final Review, Regression Check, And GitHub Update

**Files:**
- Review all task-owned files and commits.
- Do not stage existing unrelated notebook or `reports/figures/lgbm_shap_summary.png` modifications.

**Interfaces:**
- Produces: verified `main` branch pushed to `origin/main`.

- [ ] **Step 1: Run final automated verification**

Run: `python -m pytest -v`

Expected: all tests pass.

Run: `python -m compileall -q src predict_latest.py`

Expected: exit code 0.

- [ ] **Step 2: Check repository integrity and task-owned diff**

Run: `git diff --check`

Expected: no whitespace errors.

Run: `git status --short`

Expected: only the pre-existing notebook/SHAP modifications may remain; no task-owned file is uncommitted.

Run: `git log --oneline 53f3fef..HEAD`

Expected: design, plan, input, probabilistic model, orchestration, reporting, CLI, and report commits appear in logical order.

- [ ] **Step 3: Review for leakage and output truthfulness**

Inspect `src/inference.py` and confirm model-family selection reads only `validation_mae`, test metrics are report-only, all-history refit happens after the selected row is fixed, output timestamps begin at origin plus 15 minutes, and README claims match generated CSV/JSON values.

- [ ] **Step 4: Push the verified branch**

Run: `git push origin main`

Expected: remote `main` advances to the local final commit without rejection.

- [ ] **Step 5: Confirm remote synchronization**

Run: `git rev-parse HEAD`

Run: `git ls-remote origin refs/heads/main`

Expected: both commands show the same commit hash.
