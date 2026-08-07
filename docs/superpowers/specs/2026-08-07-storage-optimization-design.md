# Forecast-Driven Storage Optimization Design

## Goal

Add a portfolio-ready battery dispatch stage that converts the existing
24-hour P10/P50/P90 load forecast into an auditable charging and discharging
schedule. The first version uses linear programming. Its public interfaces and
report format must remain suitable for a later mixed-integer upgrade.

## Scope

The feature uses the saved 24-hour forecast report as input. It does not retrain
the forecasting models and does not claim that the battery or tariff parameters
come from the UCI dataset. The default configuration represents a synthetic
commercial-building demonstration.

The optimization compares three operating strategies:

- no_storage: the forecast load is supplied entirely by the grid.
- rule_based: charge during off-peak periods and discharge during peak
  periods while respecting battery constraints.
- optimized: solve the linear dispatch problem.

The P50 trajectory is the primary planning case. P10 and P90 are also evaluated
as low-load and high-load sensitivity cases.

## Architecture

The existing forecast report remains the numerical source of truth.
src/storage_optimization.py owns battery and tariff validation, strategy
simulation, objective calculation, and the linear solver. optimize_storage.py
loads a saved forecast, applies the three strategies to P10/P50/P90, and writes
portfolio artifacts.

The solver is isolated behind the optimization function rather than exposed to
the CLI or reporting code. A future mixed-integer implementation can add binary
charge/discharge states without changing forecast loading, command-line
arguments, metric names, or report files.

## Configuration

The default battery represents a synthetic building-scale system:

- Usable nameplate capacity: 500 kWh.
- Maximum charging power: 100 kW.
- Maximum discharging power: 100 kW.
- Initial state of charge: 50%.
- Minimum state of charge: 10%.
- Maximum state of charge: 90%.
- Required terminal state of charge: 50%.
- Round-trip efficiency: 90%, represented by symmetric charge and discharge
  efficiencies equal to the square root of 0.90.
- Time step: 0.25 hours.

The tariff is a configurable positive three-period synthetic time-of-use
schedule. Each 15-minute interval receives an off-peak, shoulder, or peak
energy price from its timestamp. The configuration also includes a peak-demand
penalty and a small battery throughput cost.

Every report records the complete battery and tariff configuration and labels
the source as synthetic_demo.

## Optimization Model

For interval t, the decision variables are charging power, discharging power,
grid import, battery energy, and one horizon-wide peak-import variable.

Grid balance:

    grid_import[t] = forecast_load[t] + charge[t] - discharge[t]

Battery dynamics:

    energy[t] = energy[t-1] + charge_efficiency * charge[t] * 0.25
                - discharge[t] * 0.25 / discharge_efficiency

The objective minimizes:

1. Time-of-use energy cost.
2. Peak-grid-import penalty.
3. Battery charge/discharge throughput cost.

The constraints enforce charge and discharge power limits, energy bounds,
non-negative grid import, peak import greater than every interval's grid
import, the initial energy level, and the required terminal energy level.

The linear version does not use a binary variable to prohibit simultaneous
charging and discharging. Positive tariffs, efficiency losses, and throughput
cost make that behavior economically dominated under the default
configuration. The report still measures simultaneous activity and documents
this as the main limitation before the mixed-integer upgrade.

## Data Flow

1. Load forecast.csv from a completed 24-hour report.
2. Validate that it contains 96 continuous 15-minute rows and the columns p10,
   p50, and p90.
3. Build the timestamp-aligned tariff schedule.
4. Run no_storage, rule_based, and optimized for each load scenario.
5. Validate dispatch feasibility and calculate comparable metrics.
6. Write the combined interval-level schedule and scenario summaries.
7. Render a P50 figure showing forecast load, optimized grid import, charge,
   discharge, SOC, and tariff periods.

## Outputs

The default output directory is reports/optimization/<source_label>/24h/.

- dispatch.csv: timestamp-level scenario, strategy, forecast load, grid import,
  charge, discharge, battery energy, SOC, tariff period, energy price, and
  interval cost.
- optimization_summary.json: configuration, solver metadata, scenario-level
  costs, savings, peaks, peak reduction, battery throughput, SOC extrema,
  terminal SOC, and simultaneous charge/discharge count.
- storage_dispatch.png: the P50 optimized schedule and operational context.

## Error Handling

The CLI returns a user-input error for a missing report, a non-24-hour
forecast, missing or non-finite forecast columns, non-continuous timestamps,
invalid battery bounds, non-positive efficiencies, negative tariff values, or
an infeasible optimization problem.

Artifacts are written only after the complete result passes feasibility and
round-trip validation. The original forecast reports are never modified.

## Testing

Unit tests cover configuration validation, tariff assignment, battery dynamics,
SOC and power bounds, terminal SOC, grid balance, deterministic optimization,
cost calculation, rule-based feasibility, and infeasible inputs.

CLI tests use a synthetic 96-step forecast and verify argument parsing,
scenario execution, CSV/JSON/PNG creation, metadata contents, and user-input
error exit codes.

The full existing test suite must continue to pass.

## Acceptance Criteria

1. SOC remains within configured limits for every interval.
2. Charge and discharge remain within configured power limits.
3. Terminal SOC matches the configured target within solver tolerance.
4. Grid import is non-negative and satisfies the interval balance.
5. The optimized objective is no worse than no_storage.
6. All three strategies and all three forecast scenarios are reported.
7. CSV, JSON, and PNG artifacts are non-empty and internally consistent.
8. Repeated runs with identical inputs produce identical numeric results.
9. The report clearly identifies all tariff and battery values as synthetic
   demonstration assumptions.
10. Existing and new automated tests pass.

## Portfolio Positioning

This stage creates the project narrative:

    AI load forecast -> uncertainty scenarios -> battery dispatch optimization
    -> peak and cost reduction

The linear implementation demonstrates optimization fundamentals and
explainability. The documented next upgrade is a mixed-integer model that
strictly enforces mutually exclusive charging and discharging and can add more
industrial operating states.
