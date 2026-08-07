# GRU Deep Learning Benchmark Design

## Goal

Add a reproducible PyTorch GRU benchmark for 1-hour and 24-hour electricity
load forecasting, evaluated on the same chronological partitions as the
existing Naive, Seasonal Naive, Ridge, and LightGBM models.

## Scope

This phase adds an independent deep-learning benchmark and report. It does not
replace the current LightGBM selection path, change the saved forecast contract,
or alter storage optimization inputs. The GRU uses the selected meter's recent
raw load sequence as its input; this makes the comparison explicit against the
feature-rich LightGBM pipeline rather than hiding different feature budgets.

The benchmark supports both existing horizons:

- `1h`: 4 future 15-minute values.
- `24h`: 96 future 15-minute values.

The model predicts a complete trajectory in one forward pass. Validation
selects the best checkpoint by multi-step validation MAE. The untouched test
partition is used only for final evaluation.

## Architecture

Add `src/deep_learning.py` with an optional PyTorch import guard, validated
configuration, sequence-window creation, deterministic training, evaluation,
and forecast-interval calibration. Add `deep_learning_benchmark.py` as a CLI
that loads a selected meter, runs one or both horizons, and writes reports.

PyTorch is kept in a separate `requirements-deep-learning.txt` file so the
existing tree-model and storage workflows do not require a large framework
installation. The CLI reports a clear installation error when PyTorch is not
available. It selects `cuda` when available and otherwise uses `cpu`.

## Data Windows And Leakage Control

For each forecast origin, the input window contains the previous 96 observed
15-minute load values, shaped as `(context_steps, 1)`. The target contains the
next `horizon` values. Windows are created only from the historical series and
are aligned to the target-origin timestamp.

The existing `make_multistep_targets` and `split_supervised_by_time` contracts
define the chronological 70/15/15 partitions. A sample belongs to a partition
only when its complete future target remains inside that partition. The load
normalization mean and standard deviation are fitted on the training partition
only, then reused for validation, test, and latest inference.

## Model And Training

The default model is a single-layer GRU followed by a linear multi-output head:

    input: 96 x 1 load sequence
    GRU: hidden_size=64, num_layers=1
    head: hidden_size -> horizon

Defaults are deterministic and CPU-friendly: seed 42, Adam learning rate
0.001, batch size 256, maximum 15 epochs, and early stopping patience 3 based
on validation MAE. The model uses gradient clipping at norm 1.0. The best
validation checkpoint is restored before test scoring. Training progress is
reported by the CLI and never uses the test partition for decisions.

## Output And Interval Calibration

The primary output is the point forecast and test metrics MAE, RMSE, MAPE, and
endpoint MAE. To keep the benchmark compatible with downstream uncertainty and
storage experiments, P10/P50/P90 values are also produced for the latest
forecast:

- `p50` is the GRU point prediction.
- P10 and P90 are calibrated from validation residual quantiles for each lead.
- Each row is ordered and clipped to non-negative load.

Reports include training configuration, device, train/validation/test row
counts, normalization statistics, runtime, metrics, and artifact filenames.
They do not claim that residual intervals have guaranteed coverage.

## CLI And Artifacts

The CLI interface is:

    python deep_learning_benchmark.py --horizon both

It supports `1h`, `24h`, and `both`, plus `--input`, `--meter`,
`--holiday-country` only for source metadata consistency, `--output-dir`,
`--epochs`, `--batch-size`, `--hidden-size`, `--learning-rate`, `--patience`,
`--context-steps`, and `--seed`.

The default output is `reports/deep_learning/<source_label>/<horizon>/` with:

- `metrics.json`: configuration, model metadata, and final test metrics.
- `comparison.csv`: GRU test metrics alongside saved classical comparison
  metrics when available.
- `forecast.csv`: latest timestamp, prediction, P10/P50/P90, and step.
- `forecast.png`: actual test examples and latest forecast trajectory.
- `training_history.csv`: train and validation losses by epoch.

## Error Handling

The CLI returns exit code 2 for missing input data, invalid window or training
configuration, missing PyTorch, infeasible short series, or malformed output
paths. It does not publish a partial report after a failed horizon. A failed
`both` run reports the horizon that failed.

## Testing

Unit tests cover configuration validation, window shapes and timestamp
alignment, train-only normalization, deterministic seeded training, output
shapes, non-negative ordered intervals, and test metric calculation. CLI tests
use a small synthetic series and skip only when optional PyTorch is absent.
The full existing test suite must continue to pass without PyTorch installed.

## Acceptance Criteria

1. Both 1h and 24h GRU horizons use complete future trajectories.
2. No target values or normalization statistics from validation/test leak into
   training.
3. Repeated runs with the same seed produce the same test metrics and forecast
   values within documented floating-point tolerance.
4. The best checkpoint is selected by validation MAE, never test MAE.
5. Latest P10/P50/P90 values are finite, non-negative, and ordered.
6. Missing PyTorch produces a concise installation error rather than an import
   traceback in the CLI.
7. Existing tests and the existing LightGBM, Agent, robustness, and MILP CLI
   contracts remain unchanged.

## Portfolio Positioning

This phase adds a transparent deep-learning comparison:

    raw load sequence -> GRU multi-step forecast -> calibrated uncertainty
    -> comparison with feature-rich LightGBM -> optional storage optimization

The README will clearly report whether GRU improves over the existing models;
it will not claim that a neural network is better merely because it is deeper.

PyTorch installation should follow the official platform selector rather than
hard-coding a CUDA wheel in the repository:
https://docs.pytorch.org/get-started/locally/
