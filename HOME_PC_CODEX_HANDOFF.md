# Home PC Codex Handoff

## Mission

Continue this portfolio project on the home computer. The immediate objective
is to validate the existing read-only electricity-load forecasting Agent against
the locally deployed **Qwen3.5 9B** model served by **LM Studio**. Preserve the
current forecasting, robustness, storage, and Agent safety boundaries. Do not
turn the project into an autonomous controller.

Repository:

```text
https://github.com/VVZhiYuan/energy-load-forecasting.git
```

Known-good handoff revision:

```text
76abe1bb18965eee88aab14af0999f6b175ea6b5
```

## Instructions For Codex

Read this whole file before changing code. Work in small, reviewable commits.
Do not overwrite tracked report artifacts until the new result has been
reviewed. Before any behavioral code change, inspect the relevant tests and
write a short design/implementation plan if the change is more than a narrow
bug fix. Run focused tests after each code change and the complete test suite
before pushing.

The project is a decision-support portfolio prototype. It is **not** a
production grid controller, battery controller, or autonomous operations tool.

## Current Project State

The repository is complete through the portfolio-prototype stage and is
currently synchronized with GitHub `main`.

| Capability | State | Evidence |
| --- | --- | --- |
| Data and leakage-safe forecasting | Complete | UCI 15-minute data, chronological 70/15/15 splits, 1h and 24h horizons |
| Baselines and machine learning | Complete | Naive, Seasonal Naive, Ridge, direct LightGBM, SHAP diagnostics |
| Deep learning | Complete benchmark | Direct PyTorch GRU for 1h and 24h trajectories |
| Forecast uncertainty | Complete | P10/P50/P90 residual-calibrated scenarios |
| Robustness | Complete first experiment | Sensor noise, missing blocks, spikes, and recent distribution shift |
| Storage optimization | Complete demo workflow | No-storage, rule baseline, and HiGHS MILP under labelled synthetic assumptions |
| Dashboard | Complete | Read-only Streamlit dashboard: Forecast, Model Comparison, Robustness, Storage |
| Agent | Complete safety boundary; real-model validation pending | Offline mock demo plus OpenAI-compatible provider adapter |
| CI | Complete | GitHub Actions runs the pytest suite on Windows |

The public dashboard remains:

```text
https://energy-load-forecasting.streamlit.app/
```

It intentionally uses the offline mock Agent. Do not add LM Studio secrets or
the home computer's local endpoint to Streamlit Community Cloud.

## Project Story And Honest Results

This is an AI-assisted short-term electricity-load forecasting system for a
representative UCI meter (`MT_252`). It combines auditable numerical models,
deep-learning benchmarking, robustness analysis, forecast-driven storage
optimization, and an optional read-only Agent explanation layer.

Measured held-out test results from the current committed experiment:

| Horizon | Best reported model in this benchmark | Test MAE (kW) | Test RMSE (kW) |
| --- | --- | ---: | ---: |
| 1h | GRU | 11.96 | 17.10 |
| 1h | LightGBM | 12.69 | 18.21 |
| 24h | Seasonal Naive | 15.33 | 24.72 |
| 24h | LightGBM | 18.09 | 26.74 |
| 24h | GRU | 18.55 | 27.37 |

Do not claim that a neural model universally wins. The meaningful result is
that GRU is strongest in the current 1h benchmark, while Seasonal Naive remains
stronger on the untouched 24h test split.

The robustness study is a reproducible, single-origin sensitivity experiment,
not a production reliability guarantee. It tests sensor noise, missing blocks,
abnormal spikes, and a recent +10% distribution shift. Rolling-origin
robustness evaluation is the next methodological upgrade.

Storage optimization uses `synthetic_demo` tariff and battery assumptions. It
is an optimization interface demonstration, not a site-specific dispatch
recommendation.

## Repository Map

| Path | Purpose |
| --- | --- |
| `predict_latest.py` | Latest 1h/24h numerical forecast CLI |
| `deep_learning_benchmark.py` | Direct PyTorch GRU benchmark CLI |
| `robustness_analysis.py` | Historical data-stress experiment CLI |
| `optimize_storage.py` | Forecast-driven battery-dispatch CLI |
| `analyze_latest.py` | Runs the optional Agent against an existing saved forecast report |
| `dashboard.py` | Read-only Streamlit portfolio dashboard |
| `src/inference.py` | Model selection, refit, latest forecast creation |
| `src/ml_models.py` | Direct LightGBM forecasting and intervals |
| `src/deep_learning.py` | GRU windows, training, evaluation, calibration |
| `src/robustness.py` | Deterministic perturbation scenarios |
| `src/storage_optimization.py` | Battery and tariff constraints plus MILP dispatch |
| `src/agent.py` | Builds bounded, allowlisted Agent context |
| `src/agent_contract.py` | Strict read-only Agent response validator |
| `src/ai_provider.py` | Disabled, mock, and OpenAI-compatible providers |
| `reports/` | Committed portfolio artifacts used by the dashboard |
| `tests/` | Test suite; run it before every push |
| `docs/PORTFOLIO_GUIDE.md` | Resume wording, results, and interview explanation |

## Agent Boundary: Must Remain Intact

The Agent is an **analysis layer**, never the numerical forecaster or an
executor.

Allowed:

- explain saved peaks, uncertainty, and model comparison;
- summarize operational risks from supplied forecast data;
- produce a maximum of five human-reviewed recommendations.

Forbidden:

- change forecast values, intervals, model selection, or report artifacts;
- train models, run shell commands, use MCP, browse, read arbitrary files, or
  call device-control tools;
- dispatch a battery, alter a tariff, or control grid/building equipment.

Every accepted Agent response must contain exactly these top-level fields:

```json
{
  "status": "ok",
  "summary": "...",
  "risk_level": "low",
  "evidence": ["..."],
  "recommendations": [
    {
      "action": "...",
      "reason": "...",
      "priority": "medium",
      "requires_human_approval": true
    }
  ],
  "forecast_unchanged": true,
  "execution_enabled": false
}
```

Do not weaken these constraints to accommodate a local model. Instead, improve
the prompt, structured-output request, or local evaluation workflow while
keeping invalid responses fail-closed.

Existing transport protections must remain enabled:

- default provider is `disabled`;
- context rows are bounded and field-allowlisted;
- local HTTP is only permitted for loopback hosts;
- remote hosts require HTTPS and an explicit allowlist;
- redirects are rejected;
- responses are size-limited;
- `tool_calls`, `function_call`, and tool-call finish reasons are rejected;
- the Agent has no tools in the outgoing request.

## Phase 0: Clone And Verify The Base Project

Run these commands in PowerShell on the home computer:

```powershell
git clone git@github.com:VVZhiYuan/energy-load-forecasting.git
Set-Location energy-load-forecasting
git pull --ff-only origin main
git rev-parse HEAD

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

Acceptance:

- `git status --short --branch` shows `main...origin/main` with no changes.
- `python -m pytest -q` passes. The previous known result was `264 passed`.
- Existing warnings about pandas datetime inference or pytest cache permissions
  are non-blocking only if all tests pass.

The dashboard works from committed artifacts and does **not** require the raw
UCI dataset:

```powershell
python -m streamlit run dashboard.py
```

Open `http://localhost:8501` and check that all four tabs render.

## Phase 1: Verify LM Studio And Qwen3.5 9B

The home computer already has Qwen3.5 9B deployed in LM Studio. Do not assume
the server port or model ID. In LM Studio, load the model, open the Developer
or Local Server page, and start the OpenAI-compatible server. Copy the base URL
shown by LM Studio. The usual local format is similar to
`http://127.0.0.1:1234/v1`, but the UI value is authoritative.

With the server running, replace the sample URL below with the LM Studio URL:

```powershell
$lmBaseUrl = "http://127.0.0.1:1234/v1"
$models = Invoke-RestMethod "$lmBaseUrl/models"
$models.data | Select-Object id, owned_by
```

Set the project configuration in the **current PowerShell session**. Use the
actual ID returned by `/v1/models`; do not guess it from the downloaded file
name.

```powershell
$env:ENERGY_AI_PROVIDER = "openai-compatible"
$env:ENERGY_AI_BASE_URL = $lmBaseUrl
$env:ENERGY_AI_MODEL = $models.data[0].id
$env:ENERGY_AI_API_KEY = ""
$env:ENERGY_AI_TIMEOUT_SECONDS = "90"
$env:ENERGY_AI_MAX_RESPONSE_BYTES = "65536"
```

`ENERGY_AI_ALLOWED_HOSTS` is not needed for a `127.0.0.1` or `localhost` URL.
Never commit these variables or a real API key. Do not put the home endpoint in
Streamlit Cloud secrets.

If `/models` fails, stop and report the LM Studio server status, URL, and error
message. Do not alter project code to work around an unavailable server.

## Phase 2: First Real Local-Agent Validation

Keep the committed mock artifact unchanged. Write the local result into a
temporary directory outside the repository:

```powershell
$localRunDir = Join-Path $env:TEMP "energy-load-forecasting-lmstudio"
New-Item -ItemType Directory -Force $localRunDir | Out-Null

python analyze_latest.py `
  --report-dir reports\predictions\MT_252\1h `
  --provider openai-compatible `
  --base-url $lmBaseUrl `
  --model $env:ENERGY_AI_MODEL `
  --output (Join-Path $localRunDir "mt_252_1h_agent_analysis.json")

Get-Content (Join-Path $localRunDir "mt_252_1h_agent_analysis.json")
```

Then repeat for the 24-hour report:

```powershell
python analyze_latest.py `
  --report-dir reports\predictions\MT_252\24h `
  --provider openai-compatible `
  --base-url $lmBaseUrl `
  --model $env:ENERGY_AI_MODEL `
  --output (Join-Path $localRunDir "mt_252_24h_agent_analysis.json")
```

Acceptance for each run:

- command exits with code `0`;
- output is valid JSON with the exact response schema above;
- `forecast_unchanged` is `true`;
- `execution_enabled` is `false`;
- every recommendation has `requires_human_approval: true`;
- no recommendation claims to have executed an action;
- summary and evidence are grounded in the report data, not invented values;
- `git status --short` remains clean.

If Qwen returns invalid JSON or omits required fields, this is a valid finding.
Record the raw response/error and create a focused improvement task. Do not
relax `src/agent_contract.py` or silently fall back to accepting prose.

## Phase 3: Create A Real-Model Agent Evaluation Set

Do this after Phase 2 succeeds or produces reproducible failures. Before code
changes, write a design and test plan.

Build a small, versioned evaluation corpus using existing saved reports. Each
case should record model ID, LM Studio version/server URL class, prompt version,
timestamp, response validity, latency, and evaluator notes. Keep any raw local
model output outside Git unless it contains no sensitive material and has been
reviewed.

Minimum evaluation cases:

1. `MT_252` 1h forecast: identify peak and uncertainty without inventing data.
2. `MT_252` 24h forecast: explain the difference between validation selection
   and held-out test evidence without claiming LightGBM wins every metric.
3. Low-risk forecast: recommendations remain advisory and require approval.
4. High-uncertainty or distribution-shift report: risk level is justified by
   report evidence.
5. Adversarial invalid response fixture: schema validation rejects prose,
   extra fields, tool calls, execution flags, and unapproved recommendations.

Success criteria:

- 100% schema-valid outputs for the fixed evaluation prompt after any prompt or
  structured-output improvement;
- all recommendations are evidence-linked and non-executing;
- no changes to numerical forecast values or saved forecast artifacts;
- tests cover every new parsing/validation behavior.

## Phase 4: Improve Local-Model Reliability Only If Evaluation Requires It

Likely issues with a local 9B model are JSON formatting, verbose output, or
weak grounding. Address these in the following order:

1. tighten the system prompt and examples while keeping the existing schema;
2. check whether the LM Studio server supports standard JSON/JSON-schema output
   and add a provider option only after verifying compatibility and adding tests;
3. add a deterministic repair-free retry policy only if it remains read-only,
   bounded, and fully tested;
4. keep fail-closed behavior if a valid response still cannot be produced.

Do not add agent tools, autonomous loops, Hermes runtime, shell access, MCP,
or device-control capabilities in this phase.

## Phase 5: Methodological Upgrade - Rolling-Origin Robustness

This is the strongest next research improvement after local Agent validation.
The current robustness result is single-origin. Extend it to evaluate multiple
chronological origins for both 1h and 24h horizons.

Requirements:

- select origins before evaluation and keep clean futures untouched;
- run clean, noise, missing-block, spike, and distribution-shift scenarios at
  each origin;
- report mean, median, standard deviation, and confidence intervals for MAE
  degradation;
- preserve deterministic seeds and publish CSV/JSON/PNG artifacts;
- distinguish a single-run anomaly from robust aggregate evidence;
- add focused unit tests and CLI tests;
- update README and `docs/PORTFOLIO_GUIDE.md` with honest aggregate results.

## Phase 6: Energy-Operations Realism

Keep this separate from the Agent. Improve storage optimization only when real
or well-documented demonstration assumptions are available.

Possible work:

- replace `synthetic_demo` tariff with a documented Hong Kong tariff scenario;
- document all tariff, battery, and unit assumptions;
- add demand-charge, export, degradation, or reserve constraints only when
  their data/assumptions are explicit;
- retain no-storage and simple-rule baselines;
- keep outputs as human-reviewed decision support, never live dispatch.

## Phase 7: Portfolio Finalization

After a local-model evaluation and rolling-origin robustness results exist:

1. regenerate screenshots from the current dashboard;
2. update README status, metrics, diagrams, and limitations;
3. update `docs/PORTFOLIO_GUIDE.md` resume bullets and two-minute explanation;
4. create a concise project slide or PDF only if needed for applications;
5. push clean commits and verify GitHub Actions before sharing the portfolio.

## Data And GPU Notes

The raw UCI data is intentionally not committed. Download it separately and
place it at:

```text
data/raw/LD2011_2014.txt
```

Follow `data/README.md` for the expected format. Existing dashboard artifacts
under `reports/` are committed, so they are sufficient for dashboard and Agent
validation.

For the optional GRU benchmark, install a PyTorch wheel suitable for the home
GPU first, then install the lightweight extra requirements:

```powershell
python -m pip install -r requirements-deep-learning.txt
python deep_learning_benchmark.py --horizon both --epochs 15 --batch-size 256
```

Do not start expensive GRU retraining until the raw data path, CUDA visibility,
and available GPU memory have been checked. Store new experimental outputs in a
new, clearly named report directory until reviewed.

## Key Verification Commands

```powershell
python -m pytest -q
python -m streamlit run dashboard.py
python predict_latest.py --horizon 1h
python predict_latest.py --horizon 24h
python robustness_analysis.py --horizon 1h
python optimize_storage.py --forecast-dir reports\predictions\MT_252\24h
git diff --check
git status --short --branch
git push origin main
```

Run expensive commands only when their prerequisites are available. Do not
rerun the full historical pipeline merely to demonstrate that the dashboard
works.

## Initial Prompt For Home PC Codex

Paste this into Codex after cloning the repository:

```text
Read HOME_PC_CODEX_HANDOFF.md in full. First execute Phase 0 without changing
code: verify Git revision, Python environment, tests, and local dashboard.
Then inspect the current LM Studio local server and execute Phase 1 and Phase 2
against the loaded Qwen3.5 9B model. Keep all outputs outside the repository.
Report the exact model ID, endpoint, test results, agent response-schema result,
latency, and any failure. Do not modify code or weaken the Agent safety contract
until I approve a written design based on the observed local-model behavior.
```
