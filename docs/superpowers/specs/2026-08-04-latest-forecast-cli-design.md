# Latest Forecast CLI And Probabilistic Reporting Design

## Purpose

Add a portfolio-ready command-line workflow that accepts either the original
UCI ElectricityLoadDiagrams20112014 data or a user-supplied load CSV, compares
multiple forecasting models without test leakage, and predicts the next one
hour or 24 hours from the latest observation.

The workflow must connect the existing AI forecasting experiment to realistic
smart-grid dispatch, building energy management, storage scheduling, and
electricity-price optimization use cases. It produces deterministic,
machine-readable forecasts plus static and interactive reports suitable for a
GitHub portfolio.

## Scope

This phase will:

- Add a root `predict_latest.py` command for 1-hour and 24-hour forecasts.
- Auto-detect a two-column `timestamp,load` CSV or a UCI-style wide meter
  table.
- Support both a standard model-comparison mode and an optional full
  LightGBM candidate search.
- Select the point-forecast model using chronological validation MAE only.
- Preserve a final chronological test partition for honest backtest metrics.
- Refit the selected model using all available labeled forecast origins before
  forecasting beyond the final observation.
- Generate P10, P50, and P90 uncertainty estimates with Quantile LightGBM when
  LightGBM wins, with residual calibration as a fallback for other winners.
- Write forecast CSV, model-comparison CSV, JSON metadata, a static PNG, and a
  self-contained interactive HTML report.
- Document the workflow and measured example results in the README.

This phase will not add model persistence, a web API, a dashboard server,
weather or price APIs, multiple-meter joint models, deep learning, automatic
missing-value imputation, or production deployment. Forecast timestamps mean
the intervals immediately after the supplied data; they must not be described
as current real-world dates when the input is historical.

## Confirmed Product Decisions

The user selected the following design:

- Both long and wide CSV formats are supported through automatic detection.
- Every run compares model families instead of forcing an AI model to win.
- Standard mode uses the already validated LightGBM configuration for each
  horizon; `--search` reruns all three declared LightGBM candidates.
- Every run produces CSV, JSON, PNG, and interactive HTML artifacts.
- LightGBM uncertainty uses direct quantile models for P10, P50, and P90.
- A non-LightGBM winner receives a per-step residual-calibrated interval so
  every successful forecast has a complete uncertainty contract.

## Alternatives Considered

### Fixed LightGBM Only

Always fit the previously selected LightGBM candidate and use it for the
latest forecast. This is fast and clearly AI-oriented, but it would knowingly
ignore the measured 24-hour result where Seasonal Naive beats LightGBM. It was
rejected because operational model choice should follow evidence.

### Full Search On Every Run

Rerun all model and LightGBM candidates for every invocation. This maximizes
search coverage but makes routine 24-hour forecasting unnecessarily slow. It
was rejected as the only mode.

### Standard And Search Modes

The selected approach compares Naive, Seasonal Naive, Ridge, and LightGBM on
every run. Standard mode uses the previously measured horizon-specific
LightGBM candidate, while `--search` evaluates all three existing candidates.
This provides a practical default and a reproducible experiment mode without
maintaining two separate pipelines.

## Command-Line Contract

The root command is:

```powershell
python predict_latest.py --horizon 24h
```

Supported arguments are:

| Argument | Required | Default | Meaning |
|---|---|---|---|
| `--input PATH` | No | Project UCI raw-data path | Input load file |
| `--meter NAME` | Wide input only | `MT_252` for default UCI input | Meter column to forecast |
| `--horizon {1h,24h}` | Yes | None | Four-step or 96-step forecast |
| `--search` | No | False | Evaluate all three LightGBM candidates |
| `--holiday-country CODE` | No | `PT` for default UCI; disabled for custom input | `python-holidays` country code |
| `--output-dir PATH` | No | Derived prediction directory | Artifact destination |

Examples:

```powershell
python predict_latest.py --horizon 1h
python predict_latest.py --horizon 24h --search
python predict_latest.py --input data/custom/building_a.csv --holiday-country HK --horizon 24h
python predict_latest.py --input data/custom/meter_table.csv --meter MT_002 --horizon 1h
```

The command exits nonzero on invalid input or training failure. Errors name the
argument, column, timestamp, dependency, or minimum-history condition that
caused the failure. It does not emit partial artifacts after a failed run.

## Input Detection And Normalization

Both formats normalize to a finite `pandas.Series` named `load` with a unique,
strictly increasing `DatetimeIndex`.

### Two-Column Input

A case-insensitive `timestamp` column and `load` column identify the generic
format. Additional unnamed index columns may be ignored only when they are
recognized as CSV serialization artifacts. Other ambiguous extra columns
cause a validation error rather than silent selection.

### Wide Meter Input

A parseable first timestamp column plus multiple numeric value columns
identify the wide format. The selected `--meter` must exist. The original UCI
semicolon delimiter and decimal comma remain supported through the existing
loader; custom comma-separated wide tables are also accepted.

### Validation Rules

The normalized series must have:

- Parseable timestamps with no missing or duplicate values.
- A continuous 15-minute frequency.
- Numeric, finite load values.
- Enough observations to build the 672-step weekly lag, chronological
  partitions, and the requested complete target horizon.
- At least one supervised origin in train, validation, and test after target
  boundary protection.

Rows may be sorted chronologically, but duplicate timestamps are rejected.
Missing intervals and missing loads are rejected with counts and representative
timestamps; this phase does not silently interpolate. Negative values are
allowed because custom net-load data can represent electricity export, but the
run summary records their count.

For the bundled UCI input, Portugal holidays are enabled by default. Custom
input has no implicit country assumption. If `--holiday-country` is omitted,
the holiday flag is zero and the command prints a warning. Unsupported country
codes fail clearly instead of silently degrading to all-zero flags.

## Feature And Horizon Contract

The workflow reuses `build_baseline_features`, `make_multistep_targets`, and
`split_supervised_by_time`. Features remain known at or before each forecast
origin:

- Current load.
- Lags 1, 4, 96, 192, and 672.
- Shifted rolling mean and standard deviation over 4, 96, and 672 steps.
- Cyclical hour, day-of-week, and month encodings.
- Weekend and holiday flags.

`1h` maps to four 15-minute target columns. `24h` maps to 96 target columns.
The latest feature row is the origin for the operational forecast, and output
timestamps begin exactly 15 minutes after the last observed timestamp.

## Backtest And Latest-Forecast Separation

The workflow has two deliberately separate stages.

### Stage 1: Leakage-Safe Backtest

Use the existing chronological 70/15/15 split with target-boundary protection:

1. Fit each point-forecast candidate on the training partition.
2. Compute validation trajectory MAE and RMSE.
3. Select the model with the lowest validation MAE. Deterministic ties use a
   fixed order: Naive, Seasonal Naive, Ridge, then LightGBM. LightGBM candidate
   ties retain the existing small, medium, then large declaration order.
4. Evaluate each trained candidate on the held-out test partition for the
   report, without allowing test metrics to change the winner.

The winner is therefore selected only by validation evidence. The report can
show that another model happened to score better on test, but it must retain
the validation-selected winner and explain the distinction.

### Stage 2: Operational Refit

After the backtest report is fixed, refit the selected family using all
available labeled origins and predict from the latest feature row:

- Naive and Seasonal Naive use their deterministic latest-history rules.
- Ridge reuses the validation-selected alpha and fits on all supervised rows.
- LightGBM reuses the selected candidate. Per-step best iteration counts are
  learned without test influence during candidate fitting, then bounded fixed
  iteration counts are used for the all-history refit without early stopping.

The held-out test metrics remain honest historical estimates even though the
separate operational model subsequently uses all available data.

## Model Comparison Modes

Every run compares these point predictors on identical origins:

- Naive persistence.
- Seasonal Naive using the previous day's corresponding load.
- Direct multi-output Ridge with the existing alpha candidates.
- Direct per-step LightGBM.

Standard mode uses the measured UCI winner among LightGBM configurations for
each horizon: `medium` for 1 hour and `small` for 24 hours. This is a starting
configuration for custom data, not a claim that it is universally optimal.

Search mode evaluates the existing `small`, `medium`, and `large` LightGBM
candidates on validation data, chooses the lowest validation MAE, and then
places that selected LightGBM result into the family-level comparison.

## Probabilistic Forecast Contract

The output always includes `prediction`, `p10`, `p50`, and `p90` for each
forecast step.

### LightGBM Winner

After the point-model winner is known, train direct per-step LightGBM models
with quantile objectives at alpha 0.1, 0.5, and 0.9. Quantile fitting uses only
the selected LightGBM candidate, even in search mode; it does not multiply the
candidate search. Early-stopping choices are derived from train/validation and
then fixed for the all-history operational refit.

`prediction` is the selected point-regression forecast. P10, P50, and P90 come
from the quantile estimators and can differ from the point forecast.

### Non-LightGBM Winner

Compute per-forecast-step residual quantiles from the winner's validation
predictions. Add the 10th, 50th, and 90th residual quantiles to the operational
point forecast. This preserves the winner while still giving downstream
storage or dispatch code an uncertainty range.

### Quantile Crossing

Sort P10, P50, and P90 at each forecast step after prediction so the published
contract always satisfies `p10 <= p50 <= p90`. The JSON summary records the
number of corrected steps. This post-processing is disclosed in the report.

The intervals are empirical forecast ranges, not guaranteed coverage bounds
or causal risk estimates.

## Architecture And Ownership

Use existing modules where they already own the behavior:

- Extend `src/data_loader.py` with format detection and normalized-series
  loading.
- Extend `src/ml_models.py` with quantile direct forecasting and controlled
  all-history refitting.
- Add `src/inference.py` for orchestration, model-family comparison, winner
  selection, and typed run results.
- Add `src/reporting.py` for output tables, JSON serialization, Matplotlib PNG,
  and Plotly HTML generation.
- Add root `predict_latest.py` as a thin argument-parsing and error-reporting
  entry point.

The CLI must not contain forecasting logic, and reporting must consume
structured result objects rather than parsing console text.

## Output Contract

The default directory is:

```text
reports/predictions/<source-or-meter>/<horizon>/
```

Each successful run publishes these files:

```text
forecast.csv
model_comparison.csv
summary.json
forecast.png
forecast.html
```

`forecast.csv` contains:

```text
forecast_timestamp
step
prediction
p10
p50
p90
point_model
interval_method
```

`model_comparison.csv` contains:

```text
model
configuration
validation_mae
validation_rmse
test_mae
test_rmse
selected
training_seconds
```

`summary.json` records input path and format, selected meter, observed date
range, forecast origin, horizon, holiday country, search mode, selected model
and parameters, split sizes, metrics, interval method, quantile-crossing
corrections, negative-load count, runtime, package versions, random seed, and
artifact paths.

The PNG shows a bounded recent-history window plus the future point forecast
and P10-P90 band at GitHub README width. The self-contained Plotly HTML shows
the same trajectory with hover values, the model leaderboard, selected model,
interval method, date range, and runtime. It must open without a running server
or internet connection.

Artifacts are first generated and validated in a staging directory. Only after
the complete artifact set passes validation are files moved into the final
directory, with `summary.json` published last as the completion marker. This
prevents ordinary generation failures from exposing partial new reports; the
publisher does not claim a filesystem-wide atomic replacement of all files.

## Reproducibility And Runtime

All trainable models use the existing fixed random seed and stable feature
order. Search candidates retain their declared deterministic tie order.
Estimator and library parallelism must avoid uncontrolled nested CPU usage.

Quantile models train only after LightGBM wins. This matters for the measured
24-hour UCI case, where Seasonal Naive currently wins and residual calibration
avoids fitting 288 unnecessary quantile estimators. The CLI prints stage names,
candidate progress, elapsed time, and output paths so long runs remain
observable.

Model binaries are not persisted in this phase. Reproducibility comes from the
input, fixed code, JSON metadata, and deterministic training settings.

## Error Handling

The command must fail with actionable messages for:

- Missing input file or unsupported delimiter/schema.
- Missing or unknown `--meter` for wide input.
- Invalid horizon or holiday country.
- Duplicate, missing, non-monotonic, or non-15-minute timestamps.
- Non-numeric, NaN, or infinite loads.
- Insufficient history or empty chronological partitions.
- Missing LightGBM, Plotly, holidays, or other required dependencies.
- Model fitting, prediction, or artifact-write failures.

User input errors should not include a Python traceback by default. Unexpected
programming errors may retain tracebacks for debugging. Warnings and failures
are also recorded in the JSON summary only when a successful run can still
produce a complete, truthful artifact set.

## Testing Strategy

Add focused tests using deterministic synthetic 15-minute series and monkeypatch
heavy estimators where appropriate. Cover:

- Long and wide input detection, including the original UCI delimiter and
  decimal convention.
- Unknown meters, ambiguous schemas, invalid values, duplicate timestamps,
  missing intervals, and insufficient history.
- Default Portugal holidays and explicit Hong Kong holidays.
- Exact four-row and 96-row future timestamp sequences.
- Validation-only winner selection and deterministic tie handling.
- Standard mode evaluating one LightGBM configuration and search mode
  evaluating all three.
- Quantile training only after a LightGBM family win.
- Residual fallback after a non-LightGBM win.
- Quantile crossing correction and correction counts.
- Required CSV columns, JSON fields, nonblank PNG, and valid self-contained
  HTML content.
- CLI exit codes and readable error messages.

The full existing test suite must remain green. A small CLI fixture provides an
integration smoke test without training the full UCI 24-hour estimator set.

## Verification And Acceptance Criteria

Before completion:

1. Run the complete pytest suite.
2. Run source compilation or import checks.
3. Run 1-hour and 24-hour standard forecasts on the bundled UCI `MT_252` data.
4. Run at least the 1-hour `--search` path.
5. Verify forecast row counts, 15-minute timestamps, finite values, and ordered
   quantiles with a script.
6. Parse both CSV files and `summary.json` back into structured objects.
7. Open the HTML report and visually inspect the PNG and HTML at desktop width.
8. Run `git diff --check` and inspect only task-owned changes before commits.

Acceptance requires no NaN or infinite forecast outputs, no timestamp overlap
with observed data, no quantile crossing after correction, no test leakage,
all required artifacts, and a passing test suite.

## README And Portfolio Narrative

Add a `Latest Forecast CLI` section with:

- Standard, search, UCI-wide, and Hong Kong two-column command examples.
- The generated PNG and a model-comparison table from the verified example.
- A plain-language explanation of P10, P50, and P90.
- Links to the CSV, JSON, and HTML artifacts.
- A statement that the validation winner is selected automatically.
- The measured fact that LightGBM wins the current 1-hour experiment while
  Seasonal Naive wins the current 24-hour experiment.

The narrative emphasizes disciplined AI model selection for green-energy
operations. It must not claim production deployment, guaranteed interval
coverage, live Hong Kong grid results, operational savings, or LightGBM
superiority where the measured evidence does not support those claims.

Raw UCI data and model binaries remain untracked. Verified example reports may
be committed for portfolio review. Hosting the interactive report with GitHub
Pages is a possible later deployment phase, not part of this implementation.
