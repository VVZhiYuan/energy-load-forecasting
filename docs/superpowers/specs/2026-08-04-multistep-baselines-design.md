# Multi-Step Forecasting Baselines Design

## Objective

Build the first evaluated forecasting stage for the portfolio project using
the representative meter `MT_252`. The stage predicts complete future load
trajectories at two horizons:

- 1-hour forecast: 4 future 15-minute load values.
- 24-hour forecast: 96 future 15-minute load values.

The implementation compares two transparent statistical baselines with a
direct multi-output Ridge Regression model. It produces reproducible metrics,
plots, an executed notebook, tests, and a concise README summary.

## Scope

### Included

- One meter: `MT_252`.
- Naive, seasonal-naive, and multi-output Ridge models.
- Chronological train, validation, and test partitions.
- Ridge hyperparameter selection on validation data.
- Overall, endpoint, and per-step forecast metrics.
- Reproducible notebook and machine-readable result tables.
- Unit tests for windows, leakage boundaries, predictions, and metrics.

### Excluded

- Multiple-meter training.
- Recursive forecasting.
- Probabilistic prediction intervals.
- Random Forest, XGBoost, LightGBM, LSTM, GRU, or Transformer models.
- Robustness perturbations and anomaly correction.
- A dashboard or deployment API.

## Forecast Semantics

For a forecast origin at timestamp `t`, the observed history includes the load
at `t` and all earlier timestamps. No value after `t` may appear in the model
features.

For horizon `H`, the target is the ordered vector:

```text
[load(t+1), load(t+2), ..., load(t+H)]
```

where one step equals 15 minutes. The two configured horizons are `H=4` and
`H=96`.

## Data Selection

The raw UCI file is loaded through `src.data_loader`. Only column `MT_252` is
used after loading. The source frequency remains 15 minutes and values remain
in kW.

The supervised sample builder receives a single ordered load series and a
horizon. It returns:

- A feature DataFrame indexed by forecast origin timestamp.
- A target DataFrame with columns `target_step_1` through `target_step_H`.

Rows without enough historical context or complete future targets are removed.

## Feature Set

Every feature must be known at forecast origin `t`.

### Load Features

- Current load at `t`.
- Lags at 1, 4, 96, 192, and 672 steps.
- Shifted rolling mean and standard deviation over 4, 96, and 672 steps.

Rolling features use observations ending at `t-1`. This retains the existing
anti-leakage behavior in `src.features.add_rolling_features`. Current load is
included separately.

### Calendar Features

- Cyclical hour: sine and cosine.
- Cyclical day of week: sine and cosine.
- Cyclical month: sine and cosine.
- Weekend indicator.
- Portugal holiday indicator (`PT`) because the UCI timestamps are Portuguese
  local time.

The same forecast-origin features are supplied to every Ridge output. Each
output step has its own Ridge coefficient vector, allowing the model to learn
a different relationship for each future offset.

## Chronological Splitting

The raw timestamp range is divided at 70% and 85% of ordered timestamps:

- Train interval: first 70%.
- Validation interval: next 15%.
- Test interval: final 15%.

Samples are assigned by both origin and target end:

- Train: origin is in the train interval and `t+H` is before the train end.
- Validation: origin is at or after the train end and `t+H` is before the
  validation end.
- Test: origin is at or after the validation end and `t+H` is within the data.

Historical features for validation and test may reference earlier intervals,
because those observations would be available at prediction time. Targets may
never cross a partition boundary. This removes target leakage while preserving
realistic forecast context.

## Models

### Naive

Repeat the current load at forecast origin across all `H` future steps.

```text
prediction(t+h) = load(t), h=1..H
```

### Seasonal Naive

Use the observed load from the corresponding time on the previous day.

```text
prediction(t+h) = load(t+h-96), h=1..H
```

All required seasonal values are at or before `t` when `H <= 96`.

### Multi-Output Ridge

Use a scikit-learn pipeline containing `StandardScaler` and `Ridge`. Scikit-
learn Ridge natively accepts a two-dimensional target matrix and learns one
coefficient vector per future step.

Candidate `alpha` values are:

```text
[0.1, 1.0, 10.0, 100.0]
```

Each candidate is fitted on train data and ranked by validation overall MAE.
The lowest-MAE candidate is selected. Ties select the smaller alpha. The
selected pipeline is evaluated once on test data without refitting on validation
data, keeping the reported test result independent of model selection.

## Evaluation

For each horizon and model, calculate:

- Overall MAE across every sample and forecast step.
- Overall RMSE across every sample and forecast step.
- Overall MAPE across every sample and forecast step.
- Endpoint MAE at step 4 or step 96.
- Per-step MAE for steps 1 through `H`.

MAPE remains for comparability but is interpreted cautiously because near-zero
loads can inflate percentage error. MAE is the primary selection and comparison
metric; RMSE communicates sensitivity to large misses.

## Outputs

### Code

- Extend `src/forecasting.py` with multi-step target construction and
  leakage-safe chronological masks.
- Extend `src/features.py` with cyclical calendar features and the complete
  baseline feature frame.
- Add `src/baselines.py` for naive, seasonal-naive, Ridge fitting, prediction,
  and alpha selection.
- Extend `src/evaluate.py` with multi-step summary and per-step metrics.

### Notebook

Create and execute `notebooks/02_baseline_models.ipynb`. It loads `MT_252`,
builds both horizons, trains the models, displays metrics, and saves all result
artifacts.

### Tables

- `reports/tables/baseline_metrics.csv`: one row per horizon and model.
- `reports/tables/baseline_metrics_by_step.csv`: one row per horizon, model,
  and forecast step.
- `reports/tables/ridge_alpha_selection.csv`: validation MAE for every horizon
  and alpha candidate.

### Figures

- `reports/figures/baseline_forecast_examples.png`: 1-hour and 24-hour example
  trajectories with actual, naive, seasonal-naive, and Ridge lines.
- `reports/figures/baseline_mae_by_step.png`: per-step MAE curves for each
  horizon and model.

### Documentation

Update README Week 2 status, commands, selected alpha values, test metrics, and
the two result figures. Explain why seasonal naive is the critical benchmark
for a strongly daily-periodic series.

## Error Handling

Raise `ValueError` for:

- A horizon below 1 or above the supported 96-step range.
- A series shorter than the maximum required lag plus the horizon.
- Missing or non-monotonic timestamps.
- Empty train, validation, or test samples after applying boundaries.
- Prediction arrays whose shapes differ from their target arrays.

The notebook should fail with a clear `FileNotFoundError` when the raw dataset
is missing, matching the EDA notebook behavior.

## Testing

Use pytest with small deterministic synthetic 15-minute series.

- Verify target step 1 and target step H contain the expected future values.
- Verify each split's target end stays inside its own partition.
- Verify validation and test features can use earlier historical context.
- Verify naive predictions repeat the current value with shape `(n, H)`.
- Verify seasonal-naive predictions select the prior-day aligned values.
- Verify Ridge predictions have shape `(n, H)` for both horizons.
- Verify alpha selection minimizes validation MAE and resolves ties toward the
  smaller alpha.
- Verify overall and per-step metrics against hand-calculated arrays.
- Run the complete notebook and check that all CSV and PNG outputs exist and
  are non-empty.

## Success Criteria

- Both horizons produce full 4-step and 96-step trajectories.
- No target crosses train, validation, or test boundaries.
- All tests pass.
- The notebook executes from start to finish with the project `.venv`.
- Metrics and plots are generated and committed.
- README reports actual test results without claiming Ridge wins unless the
  measured MAE supports that statement.
