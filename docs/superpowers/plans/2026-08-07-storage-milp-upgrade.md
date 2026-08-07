# Storage MILP Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the optimized battery-dispatch LP with a HiGHS MILP that enforces mutually exclusive charging and discharging per interval.

**Architecture:** Keep `optimize_dispatch` and the `optimized` strategy name as stable public interfaces. Add one binary activity variable per 15-minute interval using exact power-limit Big-M values, then solve with `scipy.optimize.milp`. Existing reporting code consumes unchanged schedule and metric schemas with updated solver metadata.

**Tech Stack:** Python 3.12, NumPy, pandas, SciPy 1.18 HiGHS MILP, matplotlib, pytest.

## Global Constraints

- Do not add a dependency: `scipy>=1.9.0` is already required and the checked environment exposes `scipy.optimize.milp`.
- Preserve `optimize_dispatch(load, tariff_schedule, battery, tariff) -> DispatchResult`, the `optimized` strategy label, CLI invocation, CSV columns, and artifact paths.
- Use exact Big-M values `battery.max_charge_kw` and `battery.max_discharge_kw`; do not add arbitrary large constants.
- Keep the default battery and tariff explicitly labelled as `synthetic_demo` assumptions.
- Do not add grid export, degradation models, real tariff claims, external services, or a dashboard.
- No optimized interval may have both charge and discharge above `1e-8` kW.

---

### Task 1: Replace The LP Optimizer With A HiGHS MILP

**Files:**
- Modify: `src/storage_optimization.py:10,458-570`
- Modify: `tests/test_storage_optimization.py:431-505`

**Interfaces:**
- Consumes: `BatteryConfig`, `TariffConfig`, `_validate_dispatch_inputs`, and `_build_dispatch_result`.
- Produces: `optimize_dispatch(...) -> DispatchResult` with `solver["method"] == "scipy_highs_milp"` and mutually exclusive charge/discharge powers.

- [ ] **Step 1: Write failing MILP contract tests**

Add this test to `tests/test_storage_optimization.py`:

```python
def test_optimizer_uses_milp_and_strictly_excludes_simultaneous_activity():
    load, tariff_schedule = make_load_and_tariff()
    result = optimize_dispatch(
        load,
        tariff_schedule,
        BatteryConfig(),
        TariffConfig(throughput_cost=0.0),
    )

    schedule = result.schedule
    assert result.solver["method"] == "scipy_highs_milp"
    assert result.metrics["simultaneous_activity_count"] == 0
    assert not (
        (schedule["charge_kw"] > 1e-8) & (schedule["discharge_kw"] > 1e-8)
    ).any()
```

Change `test_runner_reports_three_scenarios_and_three_strategies` to assert:

```python
assert summary["solver_method"] == "scipy_highs_milp"
```

Add `assert first.solver["method"] == "scipy_highs_milp"` to the
determinism test. Retain existing grid-balance, SOC, power-bound, terminal,
no-storage objective, and infeasibility tests.

- [ ] **Step 2: Run the focused tests to verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_storage_optimization.py -q
```

Expected: the new solver-method assertions fail under the LP implementation.

- [ ] **Step 3: Create MILP variables, exact bounds, and integrality**

Replace the solver import with:

```python
from scipy.optimize import Bounds, LinearConstraint, milp
```

In `optimize_dispatch`, retain continuous offsets and append interval modes:

```python
mode_start = peak_position + 1
variable_count = mode_start + count
```

Use these bounds and integrality values after calculating `min_energy` and
`max_energy`:

```python
lower_bounds = np.concatenate(
    (
        np.zeros(count), np.zeros(count), np.zeros(count),
        np.full(count, min_energy), np.array([0.0]), np.zeros(count),
    )
)
upper_bounds = np.concatenate(
    (
        np.full(count, battery.max_charge_kw),
        np.full(count, battery.max_discharge_kw),
        np.full(count, np.inf), np.full(count, max_energy),
        np.array([np.inf]), np.ones(count),
    )
)
integrality = np.zeros(variable_count, dtype=int)
integrality[mode_start:] = 1
```

- [ ] **Step 4: Add mutual-exclusivity constraint rows**

Keep the existing equality rows for grid balance, battery dynamics, and terminal
energy. Create `peak_rows`, `charge_mode_rows`, and `discharge_mode_rows`, each
with shape `(count, variable_count)`. For every `position`, set:

```python
peak_rows[position, grid_start + position] = 1.0
peak_rows[position, peak_position] = -1.0

charge_mode_rows[position, charge_start + position] = 1.0
charge_mode_rows[position, mode_start + position] = -battery.max_charge_kw

discharge_mode_rows[position, discharge_start + position] = 1.0
discharge_mode_rows[position, mode_start + position] = battery.max_discharge_kw
```

Build the solver constraint as:

```python
constraint_matrix = np.vstack(
    (equality, peak_rows, charge_mode_rows, discharge_mode_rows)
)
constraint_lower = np.concatenate(
    (equality_rhs, np.full(3 * count, -np.inf))
)
constraint_upper = np.concatenate(
    (
        equality_rhs,
        np.zeros(count),
        np.zeros(count),
        np.full(count, battery.max_discharge_kw),
    )
)
```

This enforces `charge <= max_charge * mode` and
`discharge <= max_discharge * (1 - mode)` without a loose Big-M.

- [ ] **Step 5: Solve and emit JSON-safe MILP metadata**

Replace `linprog` with:

```python
result = milp(
    c=objective,
    integrality=integrality,
    bounds=Bounds(lower_bounds, upper_bounds),
    constraints=LinearConstraint(
        constraint_matrix, constraint_lower, constraint_upper
    ),
    options={"disp": False},
)
```

Keep the existing failure and finite-solution checks. Set the base solver
metadata to:

```python
solver = {
    "method": "scipy_highs_milp",
    "success": True,
    "status": int(result.status),
    "message": str(result.message),
    "objective_value": float(result.fun),
}
```

For each of `mip_node_count`, `mip_dual_bound`, and `mip_gap`, add the field
only when `getattr(result, name, None)` is finite, converting to a plain `int`
for node count or a plain `float` otherwise. Set top-level
`summary["solver_method"]` to `scipy_highs_milp`.

- [ ] **Step 6: Run focused tests and commit**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_storage_optimization.py -q
```

Expected: all focused storage tests pass, including zero simultaneous activity
with zero throughput cost.

Commit:

```powershell
git add src\storage_optimization.py tests\test_storage_optimization.py
git commit -m "feat: enforce exclusive battery dispatch with MILP"
```

### Task 2: Refresh CLI Expectations And Portfolio Artifacts

**Files:**
- Modify: `tests/test_storage_cli.py:37-63`
- Modify: `README.md:20,60-70,339-380,540-570`
- Modify: `reports/optimization/MT_252/24h/dispatch.csv`
- Modify: `reports/optimization/MT_252/24h/optimization_summary.json`
- Modify: `reports/optimization/MT_252/24h/storage_dispatch.png`

**Interfaces:**
- Consumes: unchanged `optimize_storage.main(argv)` and MILP-enabled `run_storage_scenarios`.
- Produces: regenerated UCI artifacts with `solver_method` `scipy_highs_milp` and matching documentation.

- [ ] **Step 1: Update and run the CLI metadata assertion**

Change the existing assertion to:

```python
assert summary["solver_method"] == "scipy_highs_milp"
```

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_storage_cli.py -q
```

Expected: all CLI tests pass and the generated JSON names the MILP solver.

- [ ] **Step 2: Regenerate and inspect the saved UCI report**

Run:

```powershell
.\.venv\Scripts\python.exe optimize_storage.py --forecast-dir reports\predictions\MT_252\24h
```

Verify the JSON has nine result rows, `solver_method` is `scipy_highs_milp`,
every optimized row reports `simultaneous_activity_count` zero, and CSV/JSON/PNG
files are non-empty. Inspect `storage_dispatch.png` for visible P50
load/import, charge/discharge, SOC, and tariff panels.

- [ ] **Step 3: Update README terminology and real metrics**

Replace optimized-strategy references to LP/linear programming with HiGHS
MILP/mixed-integer linear programming. State that one binary mode per interval
enforces mutually exclusive charge/discharge. Preserve the `synthetic_demo`
disclaimer and rule-baseline caveat. Update the P50 metrics table only from the
regenerated JSON.

- [ ] **Step 4: Run all regression and publication checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
git diff --check
```

Expected: full suite passes and no whitespace errors are reported.

- [ ] **Step 5: Commit and push the portfolio refresh**

Run:

```powershell
git add README.md tests\test_storage_cli.py reports\optimization\MT_252\24h
git commit -m "docs: publish MILP storage dispatch results"
git push origin main
git status --short --branch
git ls-remote origin refs/heads/main
```

Expected: clean working tree and local `HEAD` equal to `origin/main`.

## Plan Self-Review

- Spec coverage: Task 1 covers binary variables, exact Big-M constraints,
  metadata, compatibility, and numerical acceptance. Task 2 covers CLI,
  artifacts, visual inspection, README, full tests, and publication.
- Placeholder scan: no TODO/TBD markers or ambiguous implementation steps are present.
- Type consistency: `DispatchResult`, `optimized`, and `solver_method` remain stable public contracts.
