# Forecast-Driven Storage Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Convert the saved 24-hour P10/P50/P90 load forecast into feasible battery schedules and compare no-storage, rule-based, and linearly optimized operation.

**Architecture:** Add one numerical module for configuration, tariff generation, dispatch simulation, metrics, and SciPy linear programming. Add a thin CLI that loads an existing 24-hour forecast report and atomically publishes CSV, JSON, and PNG artifacts. Keep solver-specific decisions inside the numerical module so a later MILP implementation can preserve the public interface.

**Tech Stack:** Python 3.12, pandas, NumPy, SciPy optimize.linprog, matplotlib, pytest.

## Global Constraints

- The original forecast report is never modified.
- The input must contain exactly 96 continuous 15-minute rows.
- P10, P50, and P90 are all evaluated; P50 is the primary displayed schedule.
- Battery and tariff defaults are labelled synthetic_demo.
- Default battery: 500 kWh, 100 kW charge, 100 kW discharge, 10%-90% SOC, 50% initial and terminal SOC, 90% round-trip efficiency.
- Default tariff: 0.60 off-peak, 1.00 shoulder, 1.50 peak units/kWh; 5.00 units/kW peak penalty; 0.02 units/kWh throughput cost.
- All results are deterministic for identical inputs.
- The linear implementation records simultaneous charge/discharge activity and leaves strict mutual exclusion to the future MILP upgrade.
- Preserve all unrelated user changes.

---

### Task 1: Configuration, Forecast Validation, And Tariff Schedule

**Files:**
- Modify: requirements.txt
- Modify: src/config.py
- Create: src/storage_optimization.py
- Create: tests/test_storage_optimization.py

**Interfaces:**
- BatteryConfig dataclass with validate() and efficiency properties.
- TariffConfig dataclass with validate().
- validate_forecast_frame(frame: pd.DataFrame) -> pd.DataFrame.
- build_tariff_schedule(index: pd.DatetimeIndex, tariff: TariffConfig) -> pd.DataFrame.
- OPTIMIZATION_DIR constant in src/config.py.

- [ ] **Step 1: Write failing configuration and tariff tests**

~~~python
def test_default_battery_configuration_is_valid():
    battery = BatteryConfig()
    battery.validate()
    assert battery.capacity_kwh == 500.0
    assert battery.initial_energy_kwh == 250.0
    assert battery.terminal_energy_kwh == 250.0
    assert battery.charge_efficiency == pytest.approx(np.sqrt(0.90))


def test_tariff_assigns_reproducible_periods():
    index = pd.date_range("2025-01-01", periods=96, freq="15min")
    schedule = build_tariff_schedule(index, TariffConfig())
    assert schedule.loc["2025-01-01 02:00", "tariff_period"] == "off_peak"
    assert schedule.loc["2025-01-01 12:00", "tariff_period"] == "shoulder"
    assert schedule.loc["2025-01-01 18:00", "tariff_period"] == "peak"
    assert schedule.loc["2025-01-01 23:15", "energy_price"] == 0.60


def test_forecast_requires_complete_24_hour_quantiles():
    frame = make_forecast_frame().iloc[:-1]
    with pytest.raises(ValueError, match="96"):
        validate_forecast_frame(frame)
~~~

- [ ] **Step 2: Run the focused tests and verify import failure**

Run:

~~~powershell
.\.venv\Scripts\python.exe -m pytest tests\test_storage_optimization.py -q
~~~

Expected: FAIL because src.storage_optimization does not exist.

- [ ] **Step 3: Add the direct SciPy dependency and core dataclasses**

Append scipy>=1.9.0 to requirements.txt. Add OPTIMIZATION_DIR = REPORTS_DIR / "optimization" to src/config.py.

Implement frozen dataclasses with these fields:

~~~python
@dataclass(frozen=True)
class BatteryConfig:
    capacity_kwh: float = 500.0
    max_charge_kw: float = 100.0
    max_discharge_kw: float = 100.0
    initial_soc: float = 0.50
    min_soc: float = 0.10
    max_soc: float = 0.90
    terminal_soc: float = 0.50
    round_trip_efficiency: float = 0.90
    interval_hours: float = 0.25


@dataclass(frozen=True)
class TariffConfig:
    off_peak_price: float = 0.60
    shoulder_price: float = 1.00
    peak_price: float = 1.50
    peak_import_penalty: float = 5.00
    throughput_cost: float = 0.02
    source: str = "synthetic_demo"
~~~

Validation must reject non-positive capacity or power, invalid SOC ordering,
terminal or initial SOC outside bounds, efficiencies outside (0, 1], non-
positive interval duration, negative prices, penalties, or throughput cost.

- [ ] **Step 4: Implement forecast and tariff validation**

validate_forecast_frame must:

- Require a unique DatetimeIndex.
- Sort and verify 15-minute continuity.
- Require exactly 96 rows.
- Require p10, p50, and p90.
- Coerce to finite floats.
- Reject negative loads.
- Require p10 <= p50 <= p90 at every timestamp.

build_tariff_schedule must assign:

- off_peak from 00:00 inclusive to 07:00 exclusive and 23:00 inclusive to 24:00.
- shoulder from 07:00 inclusive to 17:00 exclusive and 21:00 inclusive to 23:00.
- peak from 17:00 inclusive to 21:00 exclusive.

- [ ] **Step 5: Run focused tests**

Expected: all Task 1 tests pass.

- [ ] **Step 6: Commit**

~~~powershell
git add requirements.txt src/config.py src/storage_optimization.py tests/test_storage_optimization.py
git commit -m "feat: add storage optimization configuration"
~~~

### Task 2: Baseline Dispatch Strategies And Metrics

**Files:**
- Modify: src/storage_optimization.py
- Modify: tests/test_storage_optimization.py

**Interfaces:**
- DispatchResult dataclass with schedule: pd.DataFrame, metrics: dict[str, object], and solver: dict[str, object].
- no_storage_dispatch(load, tariff_schedule, battery, tariff) -> DispatchResult.
- rule_based_dispatch(load, tariff_schedule, battery, tariff) -> DispatchResult.
- summarize_dispatch(schedule, battery, tariff) -> dict[str, object].

- [ ] **Step 1: Write failing baseline tests**

~~~python
def test_no_storage_preserves_load_and_has_zero_battery_activity():
    result = no_storage_dispatch(load, prices, BatteryConfig(), TariffConfig())
    np.testing.assert_allclose(result.schedule["grid_import_kw"], load)
    assert result.schedule["charge_kw"].eq(0.0).all()
    assert result.schedule["discharge_kw"].eq(0.0).all()
    assert result.metrics["terminal_soc"] == pytest.approx(0.50)


def test_rule_strategy_is_feasible_and_returns_to_terminal_soc():
    result = rule_based_dispatch(load, prices, BatteryConfig(), TariffConfig())
    assert result.schedule["soc"].between(0.10, 0.90).all()
    assert result.schedule["charge_kw"].max() <= 100.0
    assert result.schedule["discharge_kw"].max() <= 100.0
    assert result.metrics["terminal_soc"] == pytest.approx(0.50, abs=1e-8)
    assert result.metrics["simultaneous_activity_count"] == 0
~~~

- [ ] **Step 2: Run the focused tests and verify missing functions**

Expected: FAIL on imports.

- [ ] **Step 3: Implement common schedule assembly and metrics**

Every schedule must contain:

- forecast_load_kw
- grid_import_kw
- charge_kw
- discharge_kw
- battery_energy_kwh
- soc
- tariff_period
- energy_price
- interval_energy_cost

Metrics must include total_energy_cost, peak_import_kw, objective_value,
battery_charge_kwh, battery_discharge_kwh, battery_throughput_kwh, min_soc,
max_soc, terminal_soc, and simultaneous_activity_count.

Objective value equals energy cost plus peak penalty times peak import plus
throughput cost times battery throughput.

- [ ] **Step 4: Implement no-storage and reachable rule-based strategies**

The no-storage schedule keeps battery energy at the initial level.

The rule strategy requests charging in off-peak periods, discharging in peak
periods, and no action in shoulder periods. After each requested action, clamp
the next battery energy to the range from which the terminal target remains
reachable using all remaining charge or discharge power. Convert the clamped
energy transition back into one charge or discharge power value. This
guarantees terminal SOC without simultaneous activity when the configuration
is feasible.

- [ ] **Step 5: Run focused tests**

Expected: baseline strategy tests pass.

- [ ] **Step 6: Commit**

~~~powershell
git add src/storage_optimization.py tests/test_storage_optimization.py
git commit -m "feat: add storage dispatch baselines"
~~~

### Task 3: Linear Programming Optimizer And Scenario Runner

**Files:**
- Modify: src/storage_optimization.py
- Modify: tests/test_storage_optimization.py

**Interfaces:**
- optimize_dispatch(load, tariff_schedule, battery, tariff) -> DispatchResult.
- run_storage_scenarios(forecast, battery, tariff) -> tuple[pd.DataFrame, dict[str, object]].

- [ ] **Step 1: Write failing optimizer tests**

~~~python
def test_optimizer_satisfies_balance_bounds_and_terminal_energy():
    result = optimize_dispatch(load, prices, BatteryConfig(), TariffConfig())
    schedule = result.schedule
    np.testing.assert_allclose(
        schedule["grid_import_kw"],
        schedule["forecast_load_kw"]
        + schedule["charge_kw"]
        - schedule["discharge_kw"],
        atol=1e-7,
    )
    assert schedule["soc"].between(0.10 - 1e-8, 0.90 + 1e-8).all()
    assert result.metrics["terminal_soc"] == pytest.approx(0.50, abs=1e-7)
    assert result.solver["success"] is True


def test_optimized_objective_is_not_worse_than_no_storage():
    baseline = no_storage_dispatch(load, prices, BatteryConfig(), TariffConfig())
    optimized = optimize_dispatch(load, prices, BatteryConfig(), TariffConfig())
    assert optimized.metrics["objective_value"] <= (
        baseline.metrics["objective_value"] + 1e-7
    )


def test_runner_reports_three_scenarios_and_three_strategies():
    dispatch, summary = run_storage_scenarios(
        make_forecast_frame(), BatteryConfig(), TariffConfig()
    )
    assert set(dispatch["scenario"]) == {"p10", "p50", "p90"}
    assert set(dispatch["strategy"]) == {
        "no_storage", "rule_based", "optimized"
    }
    assert len(summary["results"]) == 9
~~~

- [ ] **Step 2: Run tests and verify optimizer imports fail**

- [ ] **Step 3: Implement the SciPy linear program**

Use scipy.optimize.linprog with method="highs".

For N intervals create variable slices for charge[N], discharge[N],
grid_import[N], energy[N], and peak_import[1].

Objective coefficients:

- grid_import[t]: energy_price[t] * interval_hours
- charge[t] and discharge[t]: throughput_cost * interval_hours
- peak_import: peak_import_penalty

Equalities:

- grid_import[t] - charge[t] + discharge[t] = forecast_load[t]
- energy[0] - charge_efficiency * charge[0] * dt
  + discharge[0] * dt / discharge_efficiency = initial_energy
- energy[t] - energy[t-1] - charge_efficiency * charge[t] * dt
  + discharge[t] * dt / discharge_efficiency = 0
- energy[N-1] = terminal_energy

Inequalities:

- grid_import[t] - peak_import <= 0

Bounds:

- charge: 0 to max_charge_kw
- discharge: 0 to max_discharge_kw
- grid_import: 0 to infinity
- energy: min_energy to max_energy
- peak_import: 0 to infinity

Raise ValueError with the solver message when the problem is infeasible.

- [ ] **Step 4: Implement P10/P50/P90 orchestration**

For each scenario, run no_storage, rule_based, and optimized. Concatenate all
interval schedules with scenario and strategy columns. Add savings versus
no_storage and peak reduction versus no_storage to each summary row. Include
battery configuration, tariff configuration, solver method, and primary
scenario p50 in the top-level summary.

- [ ] **Step 5: Run focused tests**

Expected: all storage numerical tests pass.

- [ ] **Step 6: Commit**

~~~powershell
git add src/storage_optimization.py tests/test_storage_optimization.py
git commit -m "feat: optimize forecast-driven battery dispatch"
~~~

### Task 4: CLI And Portfolio Reports

**Files:**
- Create: optimize_storage.py
- Create: tests/test_storage_cli.py

**Interfaces:**
- CLI: python optimize_storage.py --forecast-dir reports/predictions/MT_252/24h
- Output: dispatch.csv, optimization_summary.json, storage_dispatch.png.

- [ ] **Step 1: Write failing CLI tests**

~~~python
def test_parser_defaults_to_portfolio_battery():
    args = optimize_storage.build_parser().parse_args([])
    assert args.capacity_kwh == 500.0
    assert args.max_charge_kw == 100.0
    assert args.max_discharge_kw == 100.0


def test_main_writes_complete_optimization_report(monkeypatch, tmp_path, capsys):
    forecast_dir = tmp_path / "forecast"
    forecast_dir.mkdir()
    make_forecast_frame().to_csv(forecast_dir / "forecast.csv")
    (forecast_dir / "summary.json").write_text(
        json.dumps({"source_label": "fixture", "horizon": "24h"}),
        encoding="utf-8",
    )
    output = tmp_path / "optimization"
    assert optimize_storage.main(
        ["--forecast-dir", str(forecast_dir), "--output-dir", str(output)]
    ) == 0
    assert (output / "dispatch.csv").is_file()
    assert (output / "optimization_summary.json").is_file()
    assert (output / "storage_dispatch.png").is_file()
    assert "Complete" in capsys.readouterr().out
~~~

- [ ] **Step 2: Run CLI tests and verify module import failure**

- [ ] **Step 3: Implement arguments and input loading**

Support:

- --forecast-dir, defaulting to reports/predictions/MT_252/24h
- --output-dir
- --capacity-kwh
- --max-charge-kw
- --max-discharge-kw
- --initial-soc
- --min-soc
- --max-soc
- --terminal-soc
- --round-trip-efficiency
- --off-peak-price
- --shoulder-price
- --peak-price
- --peak-import-penalty
- --throughput-cost

Load forecast.csv with its timestamp index. Load summary.json when present to
derive source_label and verify horizon is 24h.

- [ ] **Step 4: Implement atomic report publication**

Write artifacts into a temporary sibling directory, validate readback, then
replace destination files.

The P50 chart uses three aligned panels:

1. Forecast load and optimized grid import.
2. Charge and discharge power.
3. SOC with tariff period background or an aligned energy-price line.

The JSON report includes all nine result rows, complete assumptions, artifact
names, package versions, and the source forecast directory.

- [ ] **Step 5: Run CLI tests and inspect the PNG**

Expected: CLI tests pass and the generated test image is non-empty.

- [ ] **Step 6: Commit**

~~~powershell
git add optimize_storage.py tests/test_storage_cli.py
git commit -m "feat: add storage optimization CLI"
~~~

### Task 5: Real UCI Report, Documentation, And Regression Verification

**Files:**
- Create: reports/optimization/MT_252/24h/dispatch.csv
- Create: reports/optimization/MT_252/24h/optimization_summary.json
- Create: reports/optimization/MT_252/24h/storage_dispatch.png
- Modify: README.md

- [ ] **Step 1: Run the real saved 24-hour forecast**

~~~powershell
.\.venv\Scripts\python.exe optimize_storage.py
~~~

Expected: all three artifacts are generated under
reports/optimization/MT_252/24h/.

- [ ] **Step 2: Inspect numerical feasibility and visualization**

Check that P50 optimized terminal SOC is 50%, all SOC values are between 10%
and 90%, no power limits are exceeded, and optimized objective is not worse
than no_storage. Open storage_dispatch.png and verify labels and plotted values
do not overlap.

- [ ] **Step 3: Update README**

Document:

- The AI forecast to storage optimization workflow.
- Synthetic battery and tariff assumptions.
- The three strategy and three uncertainty scenario comparison.
- The exact CLI command.
- Real UCI result table and image.
- The linear model limitation and planned MILP upgrade.
- Current project completion rising from about 70% to about 80%.

- [ ] **Step 4: Run complete verification**

~~~powershell
.\.venv\Scripts\python.exe -m pytest -q
git diff --check
~~~

Expected: all existing and new tests pass with no diff whitespace errors.

- [ ] **Step 5: Commit**

~~~powershell
git add README.md reports/optimization/MT_252/24h
git commit -m "docs: publish storage optimization results"
~~~

- [ ] **Step 6: Push and verify**

~~~powershell
git push origin main
git ls-remote origin refs/heads/main
~~~

Expected: remote main equals local HEAD.
