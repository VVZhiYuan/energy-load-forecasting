# Task 1 Report: Direct Per-Step LightGBM Forecaster

## Implementation

Implemented the direct per-step LightGBM model interface from the task brief.

- Added `lightgbm>=4.3.0` and `shap>=0.46.0` to `requirements.txt`.
- Installed dependencies with the linked project `.venv`; `lightgbm-4.7.0` and `shap-0.52.0` installed successfully.
- Added `LightGBMCandidate`, `DirectLightGBMForecaster`, and `fit_direct_lightgbm`.
- Added numeric, finite, unique-index, index-alignment, feature-column, target-column, and candidate-type validation.
- Fit one deterministic `LGBMRegressor` per target step using `joblib.Parallel`.
- Added the exact focused tests from the brief.

## Files

- Modified: `requirements.txt`
- Created: `src/ml_models.py`
- Created: `tests/test_ml_models.py`
- Created: `.superpowers/sdd/2026-08-04-lightgbm-multistep/task-1-report.md`

## RED Command/Output

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_ml_models.py -v
```

Output:

```text
============================= test session starts =============================
platform win32 -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\clt\Documents\Codex\2026-08-03\lie\work\energy-load-forecasting-lightgbm\.venv\Scripts\python.exe
rootdir: C:\Users\clt\Documents\Codex\2026-08-03\lie\work\energy-load-forecasting-lightgbm
plugins: anyio-4.14.2
collecting ... collected 0 items / 1 error

=================================== ERRORS ====================================
__________________ ERROR collecting tests/test_ml_models.py ___________________
ImportError while importing test module 'C:\Users\clt\Documents\Codex\2026-08-03\lie\work\energy-load-forecasting-lightgbm\tests\test_ml_models.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\Users\clt\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\importlib\__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests\test_ml_models.py:5: in <module>
    from src.ml_models import (
E   ModuleNotFoundError: No module named 'src.ml_models'
=========================== short test summary info ===========================
ERROR tests/test_ml_models.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
============================== 1 error in 0.71s ===============================
```

## GREEN Command/Output

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_ml_models.py -v
```

Output:

```text
============================= test session starts =============================
platform win32 -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\clt\Documents\Codex\2026-08-03\lie\work\energy-load-forecasting-lightgbm\.venv\Scripts\python.exe
rootdir: C:\Users\clt\Documents\Codex\2026-08-03\lie\work\energy-load-forecasting-lightgbm
plugins: anyio-4.14.2
collecting ... collected 7 items

tests/test_ml_models.py::test_direct_lightgbm_returns_complete_trajectory[4] PASSED [ 14%]
tests/test_ml_models.py::test_direct_lightgbm_returns_complete_trajectory[96] PASSED [ 28%]
tests/test_ml_models.py::test_direct_lightgbm_rejects_misaligned_targets PASSED [ 42%]
tests/test_ml_models.py::test_direct_lightgbm_rejects_non_finite_features[nan] PASSED [ 57%]
tests/test_ml_models.py::test_direct_lightgbm_rejects_non_finite_features[inf] PASSED [ 71%]
tests/test_ml_models.py::test_direct_lightgbm_rejects_changed_prediction_columns PASSED [ 85%]
tests/test_ml_models.py::test_direct_lightgbm_is_reproducible PASSED     [100%]

============================== warnings summary ===============================
tests/test_ml_models.py: 112 warnings
  C:\Users\clt\Documents\Codex\2026-08-03\lie\work\energy-load-forecasting-lightgbm\.venv\Lib\site-packages\lightgbm\sklearn.py:1106: LGBMDeprecationWarning: The argument 'eval_set' is deprecated, use 'eval_X' and 'eval_y' instead.
    eval_set = _validate_eval_set_Xy(eval_set=eval_set, eval_X=eval_X, eval_y=eval_y)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================= 7 passed, 112 warnings in 1.70s =======================
```

## Full-Suite Output

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -v
```

Output:

```text
============================= test session starts =============================
platform win32 -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\clt\Documents\Codex\2026-08-03\lie\work\energy-load-forecasting-lightgbm\.venv\Scripts\python.exe
rootdir: C:\Users\clt\Documents\Codex\2026-08-03\lie\work\energy-load-forecasting-lightgbm
plugins: anyio-4.14.2
collecting ... collected 23 items

tests/test_baselines.py::test_naive_forecast_repeats_current_load PASSED [  4%]
tests/test_baselines.py::test_seasonal_naive_uses_previous_day_matching_steps PASSED [  8%]
tests/test_baselines.py::test_seasonal_naive_96_step_forecast_never_uses_future_data PASSED [ 13%]
tests/test_baselines.py::test_ridge_selection_returns_multioutput_model_and_smallest_tied_alpha PASSED [ 17%]
tests/test_evaluate.py::test_multistep_summary_matches_hand_calculation PASSED [ 21%]
tests/test_evaluate.py::test_per_step_metrics_returns_one_row_per_step PASSED [ 26%]
tests/test_evaluate.py::test_multistep_metrics_reject_shape_mismatch PASSED [ 30%]
tests/test_features.py::test_cyclical_features_wrap_at_daily_boundary PASSED [ 34%]
tests/test_features.py::test_baseline_features_start_after_full_historical_context PASSED [ 39%]
tests/test_features.py::test_baseline_features_contain_only_expected_columns PASSED [ 43%]
tests/test_forecasting.py::test_make_multistep_targets_contains_ordered_future_values PASSED [ 47%]
tests/test_forecasting.py::test_make_multistep_targets_rejects_unsupported_horizon[0] PASSED [ 52%]
tests/test_forecasting.py::test_make_multistep_targets_rejects_unsupported_horizon[97] PASSED [ 56%]
tests/test_forecasting.py::test_make_multistep_targets_rejects_irregular_index PASSED [ 60%]
tests/test_forecasting.py::test_split_keeps_every_target_inside_its_partition PASSED [ 65%]
tests/test_forecasting.py::test_validation_features_can_use_pre_boundary_history PASSED [ 69%]
tests/test_ml_models.py::test_direct_lightgbm_returns_complete_trajectory[4] PASSED [ 73%]
tests/test_ml_models.py::test_direct_lightgbm_returns_complete_trajectory[96] PASSED [ 78%]
tests/test_ml_models.py::test_direct_lightgbm_rejects_misaligned_targets PASSED [ 82%]
tests/test_ml_models.py::test_direct_lightgbm_rejects_non_finite_features[nan] PASSED [ 86%]
tests/test_ml_models.py::test_direct_lightgbm_rejects_non_finite_features[inf] PASSED [ 91%]
tests/test_ml_models.py::test_direct_lightgbm_rejects_changed_prediction_columns PASSED [ 95%]
tests/test_ml_models.py::test_direct_lightgbm_is_reproducible PASSED     [100%]

============================== warnings summary ===============================
tests/test_ml_models.py: 112 warnings
  C:\Users\clt\Documents\Codex\2026-08-03\lie\work\energy-load-forecasting-lightgbm\.venv\Lib\site-packages\lightgbm\sklearn.py:1106: LGBMDeprecationWarning: The argument 'eval_set' is deprecated, use 'eval_X' and 'eval_y' instead.
    eval_set = _validate_eval_set_Xy(eval_set=eval_set, eval_X=eval_X, eval_y=eval_y)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
====================== 23 passed, 112 warnings in 1.97s =======================
```

## Self-Review

- Scope check: changes are limited to the dependency file, the new model module, the new focused tests, and this requested report. The SDD ledger was not modified.
- Interface check: exported dataclasses and `fit_direct_lightgbm` match the brief.
- Validation check: tests cover output shape, model count, index mismatch rejection, non-finite feature rejection, prediction column mismatch rejection, and reproducibility.
- Determinism check: each LightGBM model uses `random_state=42`, `n_jobs=1`, and thread-based per-step parallelism.
- Patch hygiene: `git diff --check` passed with only Git's CRLF warning for `requirements.txt`.

## Concerns

- LightGBM 4.7.0 emits deprecation warnings for `eval_set`; the implementation keeps the exact API style from the task brief.
- The report file is intentionally outside the code commit scope described by the task brief's `git add` command unless the maintainer wants SDD reports committed separately.

## Fix Round 1

### What Changed

- Replaced deprecated LightGBM validation argument `eval_set=[(X_val, y_val)]` with LightGBM 4.7's keyword-only `eval_X=X_val` and `eval_y=y_val`.
- Preserved the same validation-only data, `eval_metric="l1"`, and `early_stopping(30, verbose=False)` callback semantics.
- Did not suppress warnings; the warning source was removed.

### Covering Test Files

- `tests/test_ml_models.py`
- Full regression suite

### Focused Command/Output

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_ml_models.py -v
```

Output:

```text
============================= test session starts =============================
platform win32 -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\clt\Documents\Codex\2026-08-03\lie\work\energy-load-forecasting-lightgbm\.venv\Scripts\python.exe
rootdir: C:\Users\clt\Documents\Codex\2026-08-03\lie\work\energy-load-forecasting-lightgbm
plugins: anyio-4.14.2
collecting ... collected 7 items

tests/test_ml_models.py::test_direct_lightgbm_returns_complete_trajectory[4] PASSED [ 14%]
tests/test_ml_models.py::test_direct_lightgbm_returns_complete_trajectory[96] PASSED [ 28%]
tests/test_ml_models.py::test_direct_lightgbm_rejects_misaligned_targets PASSED [ 42%]
tests/test_ml_models.py::test_direct_lightgbm_rejects_non_finite_features[nan] PASSED [ 57%]
tests/test_ml_models.py::test_direct_lightgbm_rejects_non_finite_features[inf] PASSED [ 71%]
tests/test_ml_models.py::test_direct_lightgbm_rejects_changed_prediction_columns PASSED [ 85%]
tests/test_ml_models.py::test_direct_lightgbm_is_reproducible PASSED     [100%]

============================== 7 passed in 1.59s ==============================
```

### Full-Suite Command/Output

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -v
```

Output:

```text
============================= test session starts =============================
platform win32 -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\clt\Documents\Codex\2026-08-03\lie\work\energy-load-forecasting-lightgbm\.venv\Scripts\python.exe
rootdir: C:\Users\clt\Documents\Codex\2026-08-03\lie\work\energy-load-forecasting-lightgbm
plugins: anyio-4.14.2
collecting ... collected 23 items

tests/test_baselines.py::test_naive_forecast_repeats_current_load PASSED [  4%]
tests/test_baselines.py::test_seasonal_naive_uses_previous_day_matching_steps PASSED [  8%]
tests/test_baselines.py::test_seasonal_naive_96_step_forecast_never_uses_future_data PASSED [ 13%]
tests/test_baselines.py::test_ridge_selection_returns_multioutput_model_and_smallest_tied_alpha PASSED [ 17%]
tests/test_evaluate.py::test_multistep_summary_matches_hand_calculation PASSED [ 21%]
tests/test_evaluate.py::test_per_step_metrics_returns_one_row_per_step PASSED [ 26%]
tests/test_evaluate.py::test_multistep_metrics_reject_shape_mismatch PASSED [ 30%]
tests/test_features.py::test_cyclical_features_wrap_at_daily_boundary PASSED [ 34%]
tests/test_features.py::test_baseline_features_start_after_full_historical_context PASSED [ 39%]
tests/test_features.py::test_baseline_features_contain_only_expected_columns PASSED [ 43%]
tests/test_forecasting.py::test_make_multistep_targets_contains_ordered_future_values PASSED [ 47%]
tests/test_forecasting.py::test_make_multistep_targets_rejects_unsupported_horizon[0] PASSED [ 52%]
tests/test_forecasting.py::test_make_multistep_targets_rejects_unsupported_horizon[97] PASSED [ 56%]
tests/test_forecasting.py::test_make_multistep_targets_rejects_irregular_index PASSED [ 60%]
tests/test_forecasting.py::test_split_keeps_every_target_inside_its_partition PASSED [ 65%]
tests/test_forecasting.py::test_validation_features_can_use_pre_boundary_history PASSED [ 69%]
tests/test_ml_models.py::test_direct_lightgbm_returns_complete_trajectory[4] PASSED [ 73%]
tests/test_ml_models.py::test_direct_lightgbm_returns_complete_trajectory[96] PASSED [ 78%]
tests/test_ml_models.py::test_direct_lightgbm_rejects_misaligned_targets PASSED [ 82%]
tests/test_ml_models.py::test_direct_lightgbm_rejects_non_finite_features[nan] PASSED [ 86%]
tests/test_ml_models.py::test_direct_lightgbm_rejects_non_finite_features[inf] PASSED [ 91%]
tests/test_ml_models.py::test_direct_lightgbm_rejects_changed_prediction_columns PASSED [ 95%]
tests/test_ml_models.py::test_direct_lightgbm_is_reproducible PASSED     [100%]

============================= 23 passed in 1.94s ==============================
```

### Fix Round 1 Concerns

- None. Focused and full test outputs are warning-free.
