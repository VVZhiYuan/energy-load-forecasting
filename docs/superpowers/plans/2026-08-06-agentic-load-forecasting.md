# Agentic Load Forecasting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a disabled-by-default, provider-neutral AI Agent layer for forecast explanation and energy-operation recommendations.

**Architecture:** Keep `src/inference.py` as the numeric forecasting boundary. Add an `AISettings` configuration object, an `AIProvider` protocol with disabled, mock, and OpenAI-compatible implementations, and an `AgentContext`/`AgentResponse` contract. Add a small CLI that reads an existing forecast report and writes an agent analysis artifact.

**Tech Stack:** Python 3.12, standard-library HTTP/JSON, pandas, pytest, existing forecasting modules.

## Global Constraints

- Default provider is `disabled` and must not make network requests.
- API keys must come from environment variables and must not appear in logs or JSON artifacts.
- The AI layer may interpret forecasts but must not replace or mutate numeric predictions.
- Do not add a model download, local inference runtime, or required API dependency in this phase.
- Preserve existing uncommitted notebook and report changes.

---

### Task 1: Provider Configuration And Contract

**Files:**
- Create: `src/ai_config.py`
- Create: `src/ai_provider.py`
- Create: `tests/test_ai_provider.py`

**Interfaces:**
- `AISettings.from_env() -> AISettings`
- `build_provider(settings: AISettings) -> AIProvider`
- `AIProvider.analyze(context: AgentContext) -> AgentResponse`

- [ ] **Step 1: Write failing tests** for environment defaults, provider selection, disabled no-network behavior, and mock response shape.
- [ ] **Step 2: Run `pytest tests/test_ai_provider.py -q` and verify the new imports fail.
- [ ] **Step 3: Implement settings validation and three provider implementations.** The OpenAI-compatible implementation must POST JSON to `<base_url>/chat/completions`, use the configured model, and parse a JSON object from `choices[0].message.content`.
- [ ] **Step 4: Run `pytest tests/test_ai_provider.py -q` and verify all tests pass.
- [ ] **Step 5: Commit with `git add src/ai_config.py src/ai_provider.py tests/test_ai_provider.py && git commit -m "feat: add pluggable AI providers"`.

### Task 2: Forecast Agent Context And Orchestration

**Files:**
- Create: `src/agent.py`
- Create: `tests/test_agent.py`

**Interfaces:**
- `build_agent_context(run: ForecastRun, recent_points: int = 96) -> AgentContext`
- `analyze_forecast(run: ForecastRun, settings: AISettings | None = None) -> AgentResponse`

- [ ] **Step 1: Write failing tests** for JSON-safe context construction, preservation of numeric forecast values, and provider delegation.
- [ ] **Step 2: Run `pytest tests/test_agent.py -q` and verify the tests fail before implementation.
- [ ] **Step 3: Implement context construction and provider delegation.** Include selected model, horizon, forecast rows, comparison rows, recent load values, and peak/uncertainty summary fields. Do not mutate `ForecastRun.forecast`.
- [ ] **Step 4: Run `pytest tests/test_agent.py -q` and verify all tests pass.
- [ ] **Step 5: Commit with `git add src/agent.py tests/test_agent.py && git commit -m "feat: add forecast agent orchestration"`.

### Task 3: Offline Analysis CLI And Artifact

**Files:**
- Create: `analyze_latest.py`
- Create: `tests/test_analyze_latest.py`

**Interfaces:**
- CLI command: `python analyze_latest.py --report-dir reports/predictions/MT_252/1h --provider mock`
- Default output: `<report-dir>/agent_analysis.json`

- [ ] **Step 1: Write failing tests** for loading a saved report, writing disabled/mock output, and returning exit code 2 for invalid report files.
- [ ] **Step 2: Run `pytest tests/test_analyze_latest.py -q` and verify failure.
- [ ] **Step 3: Implement the CLI using `summary.json`, `forecast.csv`, and `model_comparison.csv`. Support `--provider`, `--base-url`, `--model`, and `--output`; do not accept API keys on the command line.
- [ ] **Step 4: Run the focused tests and a real mock command against the latest 1h report.
- [ ] **Step 5: Commit with `git add analyze_latest.py tests/test_analyze_latest.py && git commit -m "feat: add offline forecast agent CLI"`.

### Task 4: Configuration Documentation And Regression Verification

**Files:**
- Create: `.env.example`
- Modify: `.gitignore`
- Modify: `README.md`

- [ ] **Step 1: Add environment examples** for `disabled`, `mock`, and `openai-compatible`, using `ENERGY_AI_BASE_URL`, `ENERGY_AI_MODEL`, and `ENERGY_AI_API_KEY`.
- [ ] **Step 2: Document the hybrid architecture, offline command, future hosted API command, and local Ollama/vLLM handoff.
- [ ] **Step 3: Ignore `.env` and verify no key-like values are tracked.
- [ ] **Step 4: Run the complete test suite with `pytest -q`.
- [ ] **Step 5: Commit with `git add .env.example .gitignore README.md && git commit -m "docs: document agent provider setup"`.

