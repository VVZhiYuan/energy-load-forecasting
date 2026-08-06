# Robustness Analysis Design

## Goal

Add a reproducible robustness experiment for short-term load forecasting under
sensor noise, missing observations, abnormal peaks, and recent distribution
shift.

## Design

The experiment uses a historical rolling-origin holdout that is still present
in the UCI dataset. For each scenario, only the observed prefix is perturbed;
the clean future horizon remains the evaluation target. The existing
`run_latest_forecast` orchestration is reused on the perturbed prefix, so model
selection, refitting, quantile handling, and reporting stay consistent with the
normal forecasting path.

The scenario layer is pure and deterministic under a seed. Missing values are
temporarily created and then linearly interpolated before model fitting, while
the report records how many points were affected and imputed. Noise, spikes,
and distribution shift are applied only before the forecast origin, never to
the target future values.

## Scenarios

- `clean`: unmodified reference case.
- `sensor_noise_5pct`: additive Gaussian noise with standard deviation equal to
  5% of observed-load standard deviation.
- `missing_blocks_1pct`: contiguous one-hour missing blocks covering about 1%
  of observed points, repaired by time interpolation.
- `spikes_1pct`: about 1% of observed points receive a positive 3-standard-
  deviation sensor spike.
- `distribution_shift_10pct`: the final 20% of observed history is increased
  by 10% before forecasting.

## Outputs

The CLI writes `robustness_metrics.csv`, `robustness_summary.json`, and a
`robustness_mae.png` comparison chart. The table includes baseline MAE,
scenario MAE, MAE delta, MAE degradation percentage, selected model, affected
points, imputed points, and the historical forecast origin.

## Constraints

- The original input file is never modified.
- The clean future target is never perturbed.
- The same random seed produces identical scenario series and metrics.
- Existing forecasting and Agent interfaces remain unchanged.

