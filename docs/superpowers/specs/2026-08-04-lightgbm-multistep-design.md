# LightGBM Multi-Step Forecasting Design

## Purpose

Extend the portfolio project from linear and persistence baselines to an
interpretable machine-learning model for short-term electricity load
forecasting. The implementation must preserve the existing leakage-safe
evaluation protocol and produce evidence suitable for AI and green-energy job
applications.

The primary optimization target is overall trajectory MAE. The model predicts
all four 15-minute values in the next hour and all 96 values in the next 24
hours. RMSE, MAPE, and endpoint MAE remain secondary diagnostics.

## Scope

This phase will:

- Train LightGBM models for the existing `MT_252` meter.
- Reuse the existing 20 load, lag, rolling, calendar, weekend, and Portugal
  holiday features.
- Compare LightGBM against Naive, Seasonal Naive, and direct multi-output
  Ridge on identical chronological test windows.
- Select LightGBM hyperparameters using validation trajectory MAE only.
- Explain predictions with aggregate gain importance and representative SHAP
  summaries.
- Generate an executed notebook, machine-readable result tables, portfolio
  figures, tests, and README documentation.

This phase will not introduce LSTM, GRU, Transformer, probabilistic forecasts,
multiple meters, weather data, a dashboard, deployment code, or tracked model
binaries.

## Alternatives Considered

### Direct LightGBM Per Forecast Step

Train one `LGBMRegressor` for every lead time: four models for the 1-hour
horizon and 96 models for the 24-hour horizon. This is the selected approach.
It supports complete trajectories without recursive error accumulation, works
well with structured features, and provides mature gain and SHAP explanations.
Its cost is managing 100 small estimators.

### Native Multi-Output XGBoost

Train one native multi-output estimator per horizon. This reduces estimator
count and uses a recognizable library, but multi-output behavior and SHAP
support are less mature than the selected single-output tree workflow. That
adds avoidable reproducibility risk to a portfolio project.

### Multi-Output Random Forest

Use scikit-learn's native multi-output trees. This is simple but is likely to
consume more memory for the 96-step task, offers coarser interpretability, and
does not provide a stronger portfolio trade-off than gradient boosting.

## Data And Evaluation Contract

The experiment uses UCI ElectricityLoadDiagrams20112014 and meter `MT_252` at
15-minute resolution. It reuses `build_baseline_features`,
`make_multistep_targets`, and `split_supervised_by_time` rather than defining a
second feature or splitting pipeline.

The split remains chronological 70/15/15. A forecast origin belongs to a
partition only when its entire target vector remains inside that partition.
Feature history may cross backward over a boundary, because it is observable
at forecast time, but targets may never cross forward into another partition.

For each horizon:

1. Train candidate LightGBM configurations on the training partition.
2. Predict the validation partition and compute overall trajectory MAE across
   every origin and forecast step.
3. Select the configuration with the lowest validation MAE. Deterministic ties
   use the least complex candidate in declared order.
4. Evaluate the selected training-only model once on the untouched test
   partition.

The validation partition may support LightGBM early stopping and configuration
selection. The test partition must not influence early stopping, parameter
selection, feature selection, or model selection. Models will not be refitted
on train plus validation before the test comparison, preserving the same
selection contract used by the Ridge baseline.

## Model Architecture

Add `src/ml_models.py` with a focused direct-forecasting interface. It will:

- Validate that feature and target frames are finite, two-dimensional,
  non-empty, index-aligned, and column-compatible across partitions.
- Train one `LGBMRegressor` per target column using a shared candidate
  configuration and fixed random seed.
- Store estimators in forecast-step order and return predictions with shape
  `(n_origins, horizon_steps)`.
- Search a small, explicitly ordered set of three configurations suitable for
  a roughly 10-20 minute CPU experiment.
- Return the selected estimator collection, validation search results, and
  aggregate gain importance.

Every candidate uses `objective="regression_l1"`, `random_state=42`,
`subsample=0.9`, `subsample_freq=1`, `colsample_bytree=0.9`, and quiet logging.
Validation MAE is monitored with 30-round early stopping independently for
each forecast step. The three candidates, in deterministic tie-break order,
are:

| Candidate | `num_leaves` | `learning_rate` | `n_estimators` | `min_child_samples` | `reg_lambda` |
|---|---:|---:|---:|---:|---:|
| Small | 15 | 0.05 | 300 | 40 | 1.0 |
| Medium | 31 | 0.05 | 400 | 20 | 0.1 |
| Large | 63 | 0.03 | 500 | 20 | 0.1 |

No additional parameter candidates or open-ended search are part of this
phase. If the installed LightGBM version requires a mechanical callback syntax
change for early stopping, the semantics and 30-round limit remain unchanged.

Model binaries will not be committed. The executed notebook and fixed
configuration provide the reproducibility path while keeping the repository
small.

## Explainability

Two complementary explanation layers will be generated:

1. **Aggregate gain importance:** combine normalized gain importance across all
   estimators for each horizon, then rank the existing input features. This
   describes which inputs the fitted tree collection used most often and most
   effectively, without claiming causality.
2. **SHAP summaries:** calculate SHAP values on a deterministic, bounded sample
   of test origins for representative forecast steps. The 1-hour task uses
   steps 1 and 4. The 24-hour task uses steps 1, 24, 48, and 96.

SHAP computation is explanatory only and happens after model selection. It
must not change the selected features, parameters, or reported test metrics.
Charts and README text will describe associations with model output, not causal
effects on electricity demand.

## Notebook And Artifacts

Create `notebooks/03_ml_models.ipynb` with these sections:

1. Experiment objective and leakage controls.
2. Data loading and reuse of baseline features.
3. LightGBM candidate training and validation selection.
4. Test comparison with all existing baselines.
5. Per-step error analysis and example trajectories.
6. Gain importance and SHAP explanations.
7. Measured conclusions and next research questions.

The executed notebook will write:

```text
reports/tables/
  ml_model_metrics.csv
  ml_metrics_by_step.csv
  lgbm_parameter_search.csv
  lgbm_feature_importance.csv

reports/figures/
  lgbm_forecast_examples.png
  lgbm_mae_by_step.png
  lgbm_feature_importance.png
  lgbm_shap_summary.png
```

`ml_model_metrics.csv` will contain horizon, model, MAE, RMSE, MAPE, endpoint
MAE, and LightGBM improvement relative to Seasonal Naive where applicable.
`ml_metrics_by_step.csv` will retain one row per horizon, model, and forecast
step. Parameter-search rows will record validation results and the selected
configuration. Feature-importance rows will record horizon, feature, raw gain,
normalized gain, and rank.

Figures must be readable at GitHub README width, nonblank, and free of clipped
labels. Example trajectories will use deterministic test origins and identical
origins for compared models within each horizon.

## Dependencies And Reproducibility

Add bounded minimum versions of `lightgbm` and `shap` to `requirements.txt`.
The notebook must run with the project `.venv` on Windows PowerShell. All model
training uses a fixed random seed and stable feature order. CPU parallelism may
be used, but estimator-level and LightGBM-level parallelism must be configured
to avoid uncontrolled nested parallelism.

If LightGBM, SHAP, or the raw dataset is missing, the failure message must name
the missing dependency or expected path. NaN and infinite model inputs are
rejected before fitting rather than silently imputed in this phase.

## Testing Strategy

Add `tests/test_ml_models.py` using small deterministic synthetic datasets. The
tests will verify:

- Four-step and 96-step fitted collections return the required prediction
  shape.
- Misaligned indexes, mismatched feature columns, NaN, and infinite values are
  rejected with useful errors.
- Candidate selection uses validation MAE and chooses the declared simpler
  candidate for a deterministic tie.
- Feature importance includes every input feature and normalized importance is
  well formed.
- Repeated fitting with the same seed produces consistent predictions.

The full existing test suite must continue to pass. Final verification also
includes source compilation, notebook execution, CSV row/schema checks,
visual inspection of all generated figures, `git diff --check`, and a clean
tracked worktree after commits.

## Documentation And Portfolio Narrative

Update the root README and notebook index after measured results exist. The
README will explain the direct per-step strategy, validation-only model
selection, leakage controls, result tables, and SHAP limitations. It will name
the measured winner for each horizon and will not claim LightGBM superiority
unless the generated test metrics support it.

The portfolio narrative connects computer science and AI methods to smart-grid
dispatch, building energy management, storage planning, and price-optimization
pre-forecasting. It does not claim production deployment, operational savings,
causal drivers, or business impact that the experiment did not measure.

## Success Criteria

Engineering success requires:

- A deterministic LightGBM direct multi-step implementation with focused tests.
- Leakage-safe validation selection and untouched test evaluation.
- A successfully executed notebook with all declared CSV and PNG artifacts.
- Existing and new tests passing.
- README documentation based only on generated measurements.
- Commits pushed to the existing GitHub repository without raw data or model
  binaries.

Model success is evaluated separately: LightGBM should attempt to reduce
overall trajectory MAE below Seasonal Naive for each horizon. Failure to beat
that baseline does not invalidate the engineering result; it must be reported
honestly and used to motivate later deep-learning or robustness experiments.
