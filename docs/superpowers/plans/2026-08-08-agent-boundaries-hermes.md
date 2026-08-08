# Agent Boundaries and Hermes Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce a read-only, schema-validated Agent boundary for forecast interpretation while keeping Hermes optional and outside the default Dashboard runtime.

**Architecture:** Add a small standard-library contract module for bounded context and response validation. Keep `src/agent.py` responsible for context construction, `src/ai_provider.py` responsible for provider transport, and `dashboard.py` responsible only for rendering validated analysis. The Hermes handoff remains documentation and an optional future integration point; no Hermes package or executable is imported by the current application.

**Tech Stack:** Python 3.12, dataclasses, `urllib.request`, `urllib.parse`, pytest, Streamlit.

## Global Constraints

- Forecasting models remain the only source of numeric truth.
- The current project records recommendations only and has no execution path.
- `disabled` remains the default provider and performs no network request.
- `mock` remains deterministic and offline.
- The openai-compatible adapter sends analysis requests only; it sends no tool definitions and accepts no tool calls.
- Forecast rows and recent observed-load rows are each bounded to at most 96 rows.
- Evidence and recommendations are each bounded to at most 5 items.
- Every recommendation must contain `requires_human_approval: true`.
- The current Agent response must contain `forecast_unchanged: true` and `execution_enabled: false`.
- API keys come only from environment/deployment secrets and are never written to reports or source files.
- Hermes is not a runtime dependency and is not imported or started by the default application.
- Use existing dependencies only; do not add Pydantic or an agent framework for this boundary layer.

---

## File Map

- Create: `src/agent_contract.py` — constants, context limits, response schema validation, and safe response constructors.
- Modify: `src/agent.py` — allowlist and bound context fields before a provider sees them.
- Modify: `src/ai_config.py` — parse timeout, response-size, and explicit host allowlist settings.
- Modify: `src/ai_provider.py` — validate endpoints, bound HTTP responses, add boundary instructions, and validate every provider response.
- Modify: `dashboard.py` — render the validated response fields without assuming unvalidated model output.
- Modify: `analyze_latest.py` — expose the strict Agent response contract to the saved-report CLI consumer.
- Modify: `docs/DEPLOYMENT.md` — document the new environment variables and the read-only/approval boundary.
- Modify: `README.md` — add a concise architecture and safety note for portfolio reviewers.
- Modify: `tests/test_agent.py` — test context field filtering and 96-row bounds.
- Modify: `tests/test_ai_provider.py` — test response validation, endpoint policy, response-size limits, and no-tool requests.
- Modify: `tests/test_analyze_latest.py` — migrate CLI assertions from legacy mock fields to the read-only contract.
- Create: `tests/test_agent_contract.py` — focused tests for the response contract and safe provider payloads.

## Task 1: Add the Agent Contract Module

**Files:**
- Create: `src/agent_contract.py`
- Create: `tests/test_agent_contract.py`

**Interfaces:**
- Produces `AgentContractError(ValueError)`.
- Produces `validate_agent_response(content: object) -> dict[str, object]`.
- Produces `disabled_response(selected_model: object = None, message: str = ...) -> dict[str, object]`.
- Produces constants `MAX_CONTEXT_ROWS = 96`, `MAX_LIST_ITEMS = 5`, `MAX_TEXT_LENGTH = 500`, and `MAX_RESPONSE_BYTES = 65536`.

- [ ] **Step 1: Write failing contract tests**

Add tests covering a valid response, missing required fields, invalid enum values, non-boolean safety flags, too many evidence items, too many recommendations, an unapproved recommendation, and a recommendation containing an execution-like extra field.

```python
def test_validate_agent_response_accepts_read_only_contract():
    result = validate_agent_response(valid_content())
    assert result["forecast_unchanged"] is True
    assert result["execution_enabled"] is False


@pytest.mark.parametrize("field", ["status", "summary", "risk_level", "evidence", "recommendations", "forecast_unchanged", "execution_enabled"])
def test_validate_agent_response_rejects_missing_required_field(field):
    content = valid_content()
    content.pop(field)
    with pytest.raises(AgentContractError):
        validate_agent_response(content)


def test_validate_agent_response_rejects_unapproved_recommendation():
    content = valid_content()
    content["recommendations"][0]["requires_human_approval"] = False
    with pytest.raises(AgentContractError, match="human approval"):
        validate_agent_response(content)


def test_validate_agent_response_rejects_execution_field():
    content = valid_content()
    content["recommendations"][0]["tool_call"] = {"name": "dispatch"}
    with pytest.raises(AgentContractError):
        validate_agent_response(content)
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `python -m pytest -q tests/test_agent_contract.py`

Expected: FAIL because `src.agent_contract` does not exist yet.

- [ ] **Step 3: Implement the minimal validator**

Use standard-library type checks. Require exactly the top-level keys
`status`, `summary`, `risk_level`, `evidence`, `recommendations`,
`forecast_unchanged`, and `execution_enabled`. Require exactly the nested
recommendation keys `action`, `reason`, `priority`, and
`requires_human_approval`. Copy input containers before returning them so the
provider response cannot mutate caller-owned data. Reject non-string text,
unsupported enums, overlong text, more than five list items, non-boolean flags,
and any extra key.

The disabled constructor must return a complete valid response with
`status="disabled"`, `risk_level="low"`, empty evidence and recommendations,
`forecast_unchanged=True`, and `execution_enabled=False`.

- [ ] **Step 4: Run focused tests and verify they pass**

Run: `python -m pytest -q tests/test_agent_contract.py`

Expected: all contract tests pass.

- [ ] **Step 5: Commit the isolated contract change**

```powershell
git add src/agent_contract.py tests/test_agent_contract.py
git commit -m "feat: add read-only agent response contract"
```

## Task 2: Bound and Minimize Agent Context

**Files:**
- Modify: `src/agent.py`
- Modify: `tests/test_agent.py`

**Interfaces:**
- Consumes `MAX_CONTEXT_ROWS` from `src.agent_contract`.
- Keeps `build_agent_context(run: ForecastRun, recent_points: int = 96) -> AgentContext` unchanged for existing callers.
- Produces an `AgentContext` whose forecast and recent-load rows are at most 96 items and whose row fields are allowlisted.

- [ ] **Step 1: Add failing context-boundary tests**

Add tests that pass a forecast frame with an extra `secret` column and a
recent-load list with more than 96 rows. Assert the secret is absent, the
recent rows are capped at the latest 96 points, and the source DataFrame is
unchanged. Add a test that a forecast with more than 96 rows raises `ValueError`
instead of silently sending an unbounded horizon.

```python
def test_build_agent_context_filters_fields_and_caps_recent_rows():
    run = make_run_with_secret_column()
    context = build_agent_context(run, recent_points=200)
    assert len(context.forecast_rows) <= 96
    assert len(context.recent_load_rows) == 96
    assert all("secret" not in row for row in context.forecast_rows)


def test_build_agent_context_rejects_forecast_over_limit():
    run = make_run(forecast_steps=97)
    with pytest.raises(ValueError, match="96"):
        build_agent_context(run)
```

- [ ] **Step 2: Run the focused tests and verify the new tests fail**

Run: `python -m pytest -q tests/test_agent.py`

Expected: the new boundary tests fail because the current implementation
passes all forecast columns and does not enforce the row limit.

- [ ] **Step 3: Implement explicit field allowlists and row bounds**

In `src/agent.py`, define module-level tuples for allowed summary keys,
forecast row keys, comparison row keys, and recent-load row keys. Build records
from those keys only after `_json_safe` conversion. Reject forecasts with more
than `MAX_CONTEXT_ROWS` rows. For recent observations, keep only the latest 96
rows and normalize each row to `timestamp` and `load`. Preserve existing
`ValueError` behavior for missing forecast columns, empty frames, invalid
indexes, and negative `recent_points`.

- [ ] **Step 4: Run the focused Agent tests**

Run: `python -m pytest -q tests/test_agent.py tests/test_agent_contract.py`

Expected: all focused Agent tests pass and no source forecast frame is mutated.

- [ ] **Step 5: Commit the context boundary**

```powershell
git add src/agent.py tests/test_agent.py
git commit -m "feat: bound and minimize agent context"
```

## Task 3: Harden Provider Transport and Validate Provider Output

**Files:**
- Modify: `src/ai_config.py`
- Modify: `src/ai_provider.py`
- Modify: `tests/test_ai_provider.py`

**Interfaces:**
- Adds `AISettings.allowed_hosts: tuple[str, ...]`, `timeout_seconds: float`, and `max_response_bytes: int` with environment parsing.
- Consumes `validate_agent_response` and `disabled_response` from `src.agent_contract`.
- Keeps `build_provider(settings: AISettings) -> AIProvider` unchanged.

- [ ] **Step 1: Add failing provider safety tests**

Add tests for: disabled and mock responses satisfying the complete contract;
non-loopback HTTP URLs being rejected; remote HTTPS hosts requiring an explicit
allowlist; loopback HTTP being accepted for local Ollama/vLLM; response bodies
larger than the configured byte limit being rejected; and the outgoing request
containing only `model` and `messages` without tools or tool choice.

```python
def test_remote_host_requires_explicit_allowlist():
    settings = AISettings(
        provider="openai-compatible",
        base_url="https://llm.example/v1",
        model="forecast-writer",
    )
    with pytest.raises(ValueError, match="allowlist"):
        build_provider(settings).analyze(make_context())


def test_loopback_http_is_allowed_for_local_model(monkeypatch):
    settings = AISettings(
        provider="openai-compatible",
        base_url="http://127.0.0.1:11434/v1",
        model="local-model",
    )
    # Fake urlopen returns a valid contract response.
    assert build_provider(settings).analyze(make_context()).content["execution_enabled"] is False


def test_oversized_response_is_rejected(monkeypatch):
    settings = AISettings(
        provider="openai-compatible",
        base_url="http://127.0.0.1:11434/v1",
        model="local-model",
        max_response_bytes=32,
    )
    with pytest.raises(AIProviderError, match="size"):
        build_provider(settings).analyze(make_context())
```

Update existing remote-endpoint tests to pass
`allowed_hosts=("llm.example",)` so the tests express the explicit production
policy instead of bypassing it.

- [ ] **Step 2: Run provider tests and verify the new tests fail**

Run: `python -m pytest -q tests/test_ai_provider.py`

Expected: the new tests fail because the settings and provider currently have
no host policy, response-size limit, or complete response validation.

- [ ] **Step 3: Add bounded AI settings**

In `src/ai_config.py`, add environment names
`ENERGY_AI_ALLOWED_HOSTS`, `ENERGY_AI_TIMEOUT_SECONDS`, and
`ENERGY_AI_MAX_RESPONSE_BYTES`. Parse comma-separated lowercase hostnames,
positive finite timeout seconds, and a positive response-byte limit. Keep
defaults at 30 seconds and 65536 bytes. Reject malformed numeric values with a
clear `ValueError` when `build_provider` validates settings.

- [ ] **Step 4: Add endpoint validation**

In `src/ai_provider.py`, parse `base_url` with `urllib.parse.urlparse` before
any request. Allow `http` only for `localhost`, `127.0.0.1`, or `::1`. Require
`https` for non-loopback hosts, reject embedded credentials, query strings,
fragments, missing hosts, and hosts not present in `allowed_hosts`. Require an
explicit allowlist entry for remote hosts. Do not resolve or contact the host
until validation succeeds.

- [ ] **Step 5: Add bounded transport and output validation**

Use `settings.timeout_seconds` for `urlopen`. Read at most
`settings.max_response_bytes + 1` bytes and reject a larger body before JSON
parsing. Include an explicit system instruction that the response is a
read-only analysis, must not change forecasts, and must return the fixed JSON
contract. Parse the model content, then call `validate_agent_response` before
constructing `AgentResponse`. Update `DisabledAIProvider` and `MockAIProvider`
to produce complete contract responses.

- [ ] **Step 6: Run focused provider tests and the full suite**

Run: `python -m pytest -q tests/test_ai_provider.py tests/test_agent.py tests/test_agent_contract.py`

Expected: all focused tests pass.

Then run: `python -m pytest -q`

Expected: all existing tests remain green; the existing pandas warning may
remain present.

- [ ] **Step 7: Commit provider hardening**

```powershell
git add src/ai_config.py src/ai_provider.py tests/test_ai_provider.py
git commit -m "feat: enforce agent provider safety boundaries"
```

## Task 4: Preserve Dashboard Behavior and Document the Handoff

**Files:**
- Modify: `dashboard.py`
- Modify: `docs/DEPLOYMENT.md`
- Modify: `README.md`
- Modify: `tests/test_dashboard.py`

**Interfaces:**
- Consumes only validated `AgentResponse.content` from the provider layer.
- Keeps the current four Dashboard tabs and offline mock mode unchanged from a user perspective.
- Documents Hermes as optional and disabled by default.

- [ ] **Step 1: Add failing Dashboard regression tests**

Add a test that a validated mock response contains the safety flags shown in
the Agent section, and a test that an invalid provider response causes the
existing warning path without hiding the forecast report. Keep the current
seven Dashboard tests as regression coverage.

```python
def test_dashboard_agent_response_is_marked_read_only():
    response = dashboard._build_offline_agent_response(forecast, metadata, comparison)
    assert response.content["forecast_unchanged"] is True
    assert response.content["execution_enabled"] is False


def test_dashboard_keeps_forecast_when_agent_fails(monkeypatch):
    monkeypatch.setattr(dashboard, "_build_offline_agent_response", failing_agent)
    app = AppTest.from_file("dashboard.py").run()
    assert app.exception == []
    assert any("unavailable" in warning.value.lower() for warning in app.warning)
```

- [ ] **Step 2: Run Dashboard tests and verify the new tests fail**

Run: `python -m pytest -q tests/test_dashboard.py`

Expected: the safety-flag assertion fails until the mock response is upgraded;
the existing forecast rendering tests remain green.

- [ ] **Step 3: Update Dashboard rendering**

Render the validated `risk_level`, `forecast_unchanged`, and
`execution_enabled` values as compact read-only metadata. Keep the explicit
offline message. Do not add action buttons that imply real dispatch or device
control. Preserve the existing exception warning path and forecast/download
flow.

- [ ] **Step 4: Update project documentation**

Add a concise “Agent Safety Boundary” section to `README.md` and extend
`docs/DEPLOYMENT.md` with the new environment variables, host allowlist
example, request limits, and the rule that a cloud deployment must not point at
the user's home-computer `localhost`. State clearly that Hermes is not required
for the dashboard and is only a future isolated adapter.

- [ ] **Step 5: Run application-level verification**

Run:

```powershell
python -m pytest -q tests/test_dashboard.py tests/test_ai_provider.py tests/test_agent.py tests/test_agent_contract.py
python -m pytest -q
```

Expected: all tests pass and the Dashboard still has the `Forecast`, `Model
Comparison`, `Robustness`, and `Storage` tabs.

Start the local app for a smoke check:

```powershell
python -m streamlit run dashboard.py --server.headless true
```

Expected: the Forecast tab shows the offline Agent analysis, the safety flags,
and the forecast chart without an external model call.

- [ ] **Step 6: Commit the Dashboard and documentation changes**

```powershell
git add dashboard.py docs/DEPLOYMENT.md README.md tests/test_dashboard.py
git commit -m "docs: expose agent safety boundary and hermes handoff"
```

## Task 5: Final Review and GitHub Handoff

**Files:**
- Verify: all files from Tasks 1-4
- Verify: `.env.example`, `.gitignore`, `.github/workflows/ci.yml`

- [ ] **Step 1: Check for accidental secrets and generated files**

Run: `git status --short` and `rg -n "sk-[A-Za-z0-9]|api[_-]?key\\s*=\\s*[^$]" .env.example README.md docs src tests`

Expected: no real secret values and no unexpected generated files.

- [ ] **Step 2: Run the complete verification set**

Run: `python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 3: Review the final diff**

Run: `git diff origin/main...HEAD --stat` and `git diff origin/main...HEAD --check`.

Expected: only the Agent boundary implementation, tests, and related docs are
included; whitespace check passes.

- [ ] **Step 4: Push the completed commits**

```powershell
git push origin main
```

Expected: GitHub `main` advances and the existing CI workflow starts.

- [ ] **Step 5: Confirm CI and worktree state**

Run: `git status --short --branch` and inspect the latest GitHub Actions run.

Expected: branch is synchronized with `origin/main`, CI is green, and no
secrets or Hermes runtime dependency are present.
