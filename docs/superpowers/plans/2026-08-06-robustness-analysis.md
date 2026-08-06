# Robustness Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate the existing forecasting pipeline under deterministic data-quality and distribution-shift scenarios.

**Architecture:** Add pure scenario perturbation functions in `src/robustness.py`, then add a CLI that truncates a historical series before a known future horizon, perturbs only the observed prefix, reuses `run_latest_forecast`, and scores predictions against the untouched future. Export CSV, JSON, and PNG artifacts.

**Tech Stack:** Python 3.12, pandas, NumPy, matplotlib, existing forecasting modules, pytest.

## Global Constraints

- The original input file is never modified.
- The clean future target is never perturbed.
- The same random seed produces identical scenario series and metrics.
- Existing forecasting and Agent interfaces remain unchanged.
- Preserve unrelated uncommitted notebook, figure, and search-report changes.

---

### Task 1: Deterministic Scenario Perturbations

**Files:**
- Create: `src/robustness.py`
- Create: `tests/test_robustness.py`

**Interfaces:**
- `scenario_catalog() -> tuple[RobustnessScenario, ...]`
- `apply_scenario(series: pd.Series, scenario: RobustnessScenario, seed: int) -> PerturbationResult`

- [ ] **Step 1: Write failing tests** for catalog names, deterministic output, missing-value repair, non-negative loads, and unchanged input.
- [ ] **Step 2: Run `pytest tests/test_robustness.py -q` and verify the import fails.
- [ ] **Step 3: Implement the five scenarios defined in the design.** Use a local NumPy generator seeded per call. Return the perturbed series plus affected and imputed counts.
- [ ] **Step 4: Run the focused tests and verify they pass.
- [ ] **Step 5: Commit with `git add src/robustness.py tests/test_robustness.py && git commit -m "feat: add deterministic robustness scenarios"`.

### Task 2: Robustness Experiment Runner

**Files:**
- Modify: `src/robustness.py`
- Modify: `tests/test_robustness.py`

**Interfaces:**
- `run_robustness_experiment(loaded, horizon_label, holiday_country, scenarios, seed, search, parallel_jobs) -> pd.DataFrame`

- [ ] **Step 1: Write failing tests** using a patched forecast runner for clean-future scoring, scenario ordering, and MAE degradation calculation.
- [ ] **Step 2: Run the focused tests and verify the new runner tests fail.
- [ ] **Step 3: Implement the historical holdout runner.** Use origin position `len(series) - horizon_steps - 1`; score every scenario against the untouched next horizon with `evaluate_forecast`.
- [ ] **Step 4: Run all robustness tests and verify they pass.
- [ ] **Step 5: Commit with `git add src/robustness.py tests/test_robustness.py && git commit -m "feat: evaluate forecast robustness"`.

### Task 3: Robustness CLI And Reports

**Files:**
- Create: `robustness_analysis.py`
- Create: `tests/test_robustness_cli.py`

**Interfaces:**
- CLI: `python robustness_analysis.py --horizon 1h`
- Output: `reports/robustness/<source>/<horizon>/robustness_metrics.csv`

- [ ] **Step 1: Write failing tests** for parser defaults, mock runner output, output files, and user-input errors.
- [ ] **Step 2: Run the focused CLI tests and verify failure.
- [ ] **Step 3: Implement the CLI and chart writer.** Support the same input, meter, holiday, search, and output conventions as `predict_latest.py`; add `--seed` and `--scenarios`.
- [ ] **Step 4: Run a synthetic CLI test and verify CSV, JSON, and PNG artifacts.
- [ ] **Step 5: Commit with `git add robustness_analysis.py tests/test_robustness_cli.py && git commit -m "feat: add robustness analysis CLI"`.

### Task 4: Portfolio Documentation And Regression Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document the scenarios, evaluation protocol, and interpretation of degradation metrics.
- [ ] **Step 2: Add the real UCI 1h and 24h robustness report links when generated.
- [ ] **Step 3: Run `pytest -q` and `git diff --check`.
- [ ] **Step 4: Commit with `git add README.md && git commit -m "docs: explain robustness evaluation"`.

