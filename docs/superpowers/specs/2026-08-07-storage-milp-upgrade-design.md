# Storage MILP Upgrade Design

## Goal

Upgrade the portfolio battery optimizer from a linear program to a mixed
integer linear program (MILP) that strictly prevents simultaneous charging and
discharging in every 15-minute interval. Preserve the existing forecast input,
CLI arguments, strategy labels, report artifact paths, and metric schema.

## Scope

The feature changes only the `optimized` storage strategy. `no_storage` and
`rule_based` remain transparent comparison baselines. The optimizer continues
to evaluate P10, P50, and P90 load trajectories from a saved 24-hour forecast.

The default 500 kWh battery and three-period `synthetic_demo` tariff remain
demonstration assumptions. The upgrade does not add grid export, degradation
curves, renewable generation, real tariffs, new external APIs, or a dashboard.

## Architecture

`src/storage_optimization.py` remains the single owner of dispatch
optimization. Its public `optimize_dispatch(load, tariff_schedule, battery,
tariff)` function will use `scipy.optimize.milp` and retain its current
`DispatchResult` return type. `run_storage_scenarios` will continue to expose
the optimized schedule under strategy name `optimized`; its top-level solver
metadata changes to `scipy_highs_milp`.

`optimize_storage.py` remains unchanged in its invocation and artifact names.
It will automatically publish the updated MILP schedule and solver metadata.

## Optimization Model

For each interval t, the continuous decisions remain charge power,
discharge power, grid import, and battery energy. The model retains one
horizon-wide continuous peak-import variable and adds one binary activity mode
per interval:

    mode[t] in {0, 1}

The existing grid balance, battery dynamics, terminal energy, non-negative
grid import, SOC bounds, charge/discharge power limits, peak-import bound, and
objective remain unchanged.

The two new Big-M constraints are:

    charge[t] <= max_charge_kw * mode[t]
    discharge[t] <= max_discharge_kw * (1 - mode[t])

`mode[t] = 1` permits charging and sets discharge to zero. `mode[t] = 0`
permits discharging and sets charge to zero. Both powers may be zero. The
existing maximum charge and discharge power values are the exact Big-M values,
so no looser constant is introduced.

SciPy HiGHS MILP uses the same numeric objective:

1. Time-of-use grid energy cost.
2. Peak-import penalty.
3. Charge and discharge throughput cost.

## Result Metadata And Compatibility

The optimized `DispatchResult.solver` object will report:

- `method`: `scipy_highs_milp`
- `success`, `status`, `message`, and `objective_value` from SciPy.
- `mip_node_count` and `mip_dual_bound` when SciPy returns finite values.

The top-level `optimization_summary.json` field `solver_method` will be
`scipy_highs_milp`. The strategy remains named `optimized`, so existing charts,
CSV consumers, and command lines do not require migration.

## Error Handling

An infeasible MILP or a solver result without a finite solution raises the
existing user-facing `ValueError` prefix: `storage optimization failed:`.
No artifact is published when solving fails. The CLI continues to return exit
code 2 for that error.

## Testing

Tests will retain all existing feasibility checks and add coverage that the
optimized schedule has zero simultaneous charge/discharge intervals, including
a zero-throughput-cost configuration where the previous economic argument is
not sufficient. Tests will verify the binary-mode model returns the MILP solver
metadata, remains deterministic for fixed input, obeys the terminal SOC, and
has an objective no worse than no storage.

CLI coverage will assert that the published summary identifies the MILP solver.
The saved UCI `MT_252` 24-hour report will be regenerated, visually inspected,
and documented in the README. The full test suite must pass.

## Acceptance Criteria

1. `optimize_dispatch` solves with SciPy HiGHS MILP and reports
   `scipy_highs_milp`.
2. Every optimized interval has either zero charge power or zero discharge
   power, within a 1e-8 kW numerical tolerance.
3. Grid balance, non-negative import, SOC limits, power limits, and terminal
   SOC all remain satisfied.
4. The optimized objective is no worse than the no-storage baseline for the
   same load, tariff, and battery configuration.
5. All P10/P50/P90 reports keep their existing strategy names and files.
6. The generated JSON identifies the MILP solver and the README accurately
   explains the mutually exclusive operation.
7. Focused and full automated tests pass.
