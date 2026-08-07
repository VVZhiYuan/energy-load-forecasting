# GRU Deep Learning Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Add a reproducible PyTorch GRU benchmark for direct 1h and 24h electricity-load forecasting without changing the existing LightGBM inference contract.

**Architecture:** src/deep_learning.py owns optional PyTorch access, leakage-safe windows, train-only scaling, direct GRU training, checkpoint selection, latest inference, and residual intervals. src/deep_learning_reporting.py publishes reports. deep_learning_benchmark.py is a thin CLI over the existing data loader.

**Tech Stack:** Python 3.12, PyTorch, NumPy, pandas, matplotlib, pytest, and existing forecast/evaluation modules.

## Global Constraints

- Declare PyTorch only in requirements-deep-learning.txt; leave requirements.txt unchanged.
- Preserve all current forecasting, Agent, robustness, storage, CLI, and report contracts.
- Use make_multistep_targets and split_supervised_by_time for complete-target chronological splits.
- Use exactly 96 historical 15-minute load values and one input feature per GRU sample.
- Fit normalization only from training inputs; select checkpoints only by validation MAE.
- Use seed 42, num_workers=0, Adam, gradient clipping 1.0, and validation-MAE early stopping.
- Label P10/P50/P90 as validation-residual calibration, not guaranteed coverage.
- Missing PyTorch must produce a concise CLI exit-code-2 error.

---

### Task 1: Optional Dependency And Leakage-Safe Sequence Data

**Files:**
- Create: requirements-deep-learning.txt
- Create: src/deep_learning.py
- Create: tests/test_deep_learning.py

**Interfaces:**
- Produces: GRUConfig, SequenceWindows, SequencePartition, LoadScaler, require_torch, make_sequence_windows, and split_sequence_windows.

- [ ] **Step 1: Add optional dependency declaration**

Create requirements-deep-learning.txt:

~~~text
# Select a CPU or CUDA wheel for the host at pytorch.org/get-started/locally.
torch>=2.0.0
~~~

- [ ] **Step 2: Write failing sequence and scaler tests**

Add continuous 15-minute Series fixtures and test:
- A 96-value historical context maps to the next complete horizon target.
- The first 4-step window has origin series.index[95], context series.iloc[:96], and target series.iloc[96:100].
- A 96-step horizon gets non-empty train/validation/test partitions and each target stays inside its chronological partition.
- LoadScaler.fit on [[[1.0]], [[3.0]]] reports mean 2.0 and standard deviation 1.0.

~~~python
def test_sequence_windows_keep_historical_context_and_future_targets():
    series = make_series(240)
    windows = make_sequence_windows(series, horizon=4, context_steps=96)
    assert windows.inputs.shape[1:] == (96, 1)
    assert windows.index[0] == series.index[95]
    np.testing.assert_allclose(windows.inputs[0, :, 0], series.iloc[:96])
    np.testing.assert_allclose(windows.targets.iloc[0], series.iloc[96:100])
~~~

- [ ] **Step 3: Implement dependency-safe sequence preparation**

Guard PyTorch at import time and raise only from training entry points:

~~~python
try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

def require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for GRU benchmarking. Install it with: python -m pip install -r requirements-deep-learning.txt")
~~~

Implement frozen GRUConfig defaults: context 96, hidden 64, one layer, batch 256, epochs 15, learning rate 0.001, patience 3, seed 42. Reject non-positive parameters and patience greater than or equal to epochs.

make_sequence_windows builds float32 arrays shaped (samples, context_steps, 1) at origins beginning at context_steps - 1, paired with non-null targets from make_multistep_targets. split_sequence_windows passes an aligned origin frame to split_supervised_by_time and selects array rows by returned timestamps. LoadScaler.fit uses training inputs only, replaces zero standard deviation with 1.0, and preserves shapes.

- [ ] **Step 4: Run and commit Task 1**

Run: .\.venv\Scripts\python.exe -m pytest tests\test_deep_learning.py -q
Expected: dependency-free data tests pass before PyTorch installation.

Commit: git add requirements-deep-learning.txt src\deep_learning.py tests\test_deep_learning.py && git commit -m "feat: add leakage-safe GRU sequence windows"

### Task 2: Deterministic Direct GRU Training And Intervals

**Files:**
- Modify: src/deep_learning.py
- Modify: tests/test_deep_learning.py

**Interfaces:**
- Produces: DirectGRU, GRUTrainingResult, fit_direct_gru, predict_gru, calibrate_residual_intervals, and run_gru_benchmark.

- [ ] **Step 1: Install and verify PyTorch**

Run nvidia-smi. Install the host-appropriate PyTorch wheel with .\.venv\Scripts\python.exe -m pip install -r requirements-deep-learning.txt. Verify: .\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())".

- [ ] **Step 2: Write failing training and interval tests**

Mark training tests with pytest.mark.skipif when PyTorch is absent. Run two seeded three-epoch jobs on small partitions and assert equal validation predictions, correct horizon shape, and positive best epoch. Test that residual intervals clip negative values and sort each P10/P50/P90 row.

~~~python
def test_direct_gru_is_deterministic_on_a_small_partition():
    train, validation = make_small_partitions()
    config = GRUConfig(epochs=3, patience=1, batch_size=16, hidden_size=8)
    first = fit_direct_gru(train, validation, config)
    second = fit_direct_gru(train, validation, config)
    np.testing.assert_allclose(first.validation_prediction, second.validation_prediction, atol=1e-6)
~~~

- [ ] **Step 3: Implement direct multi-step GRU**

Use a GRU encoder and a trajectory head:

~~~python
self.encoder = nn.GRU(1, config.hidden_size, config.num_layers, batch_first=True)
self.head = nn.Linear(config.hidden_size, horizon)
~~~

Set Python, NumPy, and Torch seeds and torch.use_deterministic_algorithms(True, warn_only=True). Train with normalized MSE, Adam, a seeded num_workers=0 DataLoader, and clip_grad_norm_ at 1.0. Evaluate validation MAE in raw units each epoch, deep-copy the best state, stop after patience, and restore it.

run_gru_benchmark builds/splits windows, scores the untouched test target with evaluate_multistep, creates future timestamps after the final observation, and forms p10/p90 from lead-wise 0.1/0.9 validation residual quantiles around point p50.

- [ ] **Step 4: Run and commit Task 2**

Run: .\.venv\Scripts\python.exe -m pytest tests\test_deep_learning.py -q
Expected: all data and PyTorch-enabled tests pass.

Commit: git add src\deep_learning.py tests\test_deep_learning.py && git commit -m "feat: train deterministic direct GRU forecaster"

### Task 3: CLI And Atomic Benchmark Reports

**Files:**
- Create: src/deep_learning_reporting.py
- Create: deep_learning_benchmark.py
- Create: tests/test_deep_learning_cli.py

**Interfaces:**
- Consumes: LoadedLoadSeries, run_gru_benchmark, and saved forecast comparisons.
- Produces per horizon: metrics.json, comparison.csv, forecast.csv, forecast.png, and training_history.csv.

- [ ] **Step 1: Write failing CLI tests**

Test horizon both, missing-PyTorch exit code 2, and a monkeypatched benchmark that creates all five artifacts:

~~~python
def test_main_publishes_complete_gru_report(monkeypatch, tmp_path):
    monkeypatch.setattr(deep_learning_benchmark, "load_forecast_series", fake_loaded)
    monkeypatch.setattr(deep_learning_benchmark, "run_gru_benchmark", fake_run)
    output = tmp_path / "report"
    assert deep_learning_benchmark.main(["--input", "fixture.csv", "--horizon", "1h", "--output-dir", str(output)]) == 0
    assert (output / "metrics.json").is_file()
    assert (output / "forecast.png").stat().st_size > 0
~~~

- [ ] **Step 2: Implement reporting and CLI**

Write reports to a temporary sibling directory, validate CSV/JSON/PNG readback, then publish. The PNG has actual-vs-predicted test and latest P10/P50/P90 panels. comparison.csv contains GRU plus saved classical rows when available. JSON records device, runtime, config, split sizes, scaler, validation/test metrics, validation_residual_quantiles, and artifact names.

Support input, meter, horizon (1h, 24h, both), output path, and all config overrides. Default UCI uses MT_252; both runs 1h then 24h. Catch OSError, ValueError, and RuntimeError, print Error:, and return 2.

- [ ] **Step 3: Run and commit Task 3**

Run: .\.venv\Scripts\python.exe -m pytest tests\test_deep_learning_cli.py tests\test_deep_learning.py -q
Commit: git add src\deep_learning_reporting.py deep_learning_benchmark.py tests\test_deep_learning_cli.py && git commit -m "feat: publish GRU benchmark reports"

### Task 4: UCI Benchmark, Documentation, And Publication

**Files:**
- Modify: README.md
- Create: reports/deep_learning/MT_252/1h/*
- Create: reports/deep_learning/MT_252/24h/*

**Interfaces:**
- Consumes: completed GRU CLI and UCI raw data.
- Produces: committed UCI reports and accurate portfolio documentation.

- [ ] **Step 1: Run and inspect real horizons**

Run: .\.venv\Scripts\python.exe deep_learning_benchmark.py --horizon both --epochs 15 --batch-size 256

Verify each JSON has non-zero partition sizes, 4/96 output lengths, device and runtime, and ordered non-negative intervals. Inspect both PNGs and comparison CSVs.

- [ ] **Step 2: Document results honestly**

Mark deep learning complete. Add optional-install and benchmark commands, artifact links, real MAE/RMSE values, model assumptions, interval limitation, and an evidence-based GRU comparison to LightGBM and Seasonal Naive. Update architecture and roadmap while leaving dashboard and portfolio packaging next.

- [ ] **Step 3: Run full regression and publish**

Run: .\.venv\Scripts\python.exe -m pytest -q
Run: git diff --check
Expected: all tests pass and no whitespace errors.

Commit: git add README.md reports\deep_learning && git commit -m "docs: publish GRU benchmark results"
Publish: git push origin main; git status --short --branch; git ls-remote origin refs/heads/main
Expected: committed reports and local HEAD equal to origin/main.

## Plan Self-Review

- Spec coverage: dependency, leakage prevention, deterministic GRU, intervals, CLI, artifacts, UCI run, documentation, tests, and publishing are assigned.
- Placeholder scan: no unfinished markers or unbounded work items remain.
- Type consistency: windows feed partitions, partitions train the GRU, the benchmark run feeds reporting, and existing LightGBM inference remains intact.
