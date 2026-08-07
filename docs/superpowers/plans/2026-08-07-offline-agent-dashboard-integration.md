# Offline Agent Dashboard Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, offline AI operations interpretation section to the Streamlit Forecast tab while preserving the existing provider-neutral handoff for a future local or company-approved model.

**Architecture:** Reuse `build_agent_context_from_frames` to convert the loaded forecast artifacts into the existing `AgentContext` contract. The dashboard will construct `AISettings(provider="mock", model="offline-mock")` explicitly and invoke the existing provider factory, so the default page never reads an API key or makes a network request. A small rendering helper will display the returned `AgentResponse` after the forecast chart; errors will become a warning while the numerical forecast remains available.

**Tech Stack:** Python 3.11+, Streamlit, pandas, Plotly, pytest, existing provider-neutral Agent layer.

## Global Constraints

- The dashboard must remain read-only: it reads committed CSV and JSON artifacts and does not retrain models or modify reports.
- The Dashboard Agent path must use the explicit offline `mock` provider and must not call external APIs.
- The numeric forecast, chart, CSV download, and existing four-tab layout must remain usable if Agent context construction or rendering fails.
- The existing `openai-compatible` provider and `AgentResponse` contract must remain unchanged for a future Ollama/vLLM or company API handoff.
- No API key, local model path, or provider secret may be written to source code or committed artifacts.
- The AI section must use existing Streamlit primitives and keep readable spacing on narrow layouts.

---

### Task 1: Add dashboard Agent adapter

**Files:**
- Modify: `C:\Users\clt\Documents\Codex\2026-08-03\lie\work\energy-load-forecasting\dashboard.py`
- Test: `C:\Users\clt\Documents\Codex\2026-08-03\lie\work\energy-load-forecasting\tests\test_dashboard.py`

**Interfaces:**
- Consumes: `forecast: pd.DataFrame`, `metadata: dict[str, Any]`, and `comparison: pd.DataFrame` from `_show_forecast`.
- Produces: `_build_offline_agent_response(forecast, metadata, comparison) -> AgentResponse`.

- [ ] **Step 1: Write the failing adapter test**

Add this test to `tests/test_dashboard.py`:

```python
def test_build_offline_agent_response_forces_mock_provider(monkeypatch):
    forecast = pd.DataFrame(
        {"step": [1, 2], "prediction": [100.0, 120.0], "p10": [90.0, 100.0], "p90": [110.0, 140.0]},
        index=pd.date_range("2025-01-01", periods=2, freq="h"),
    )
    comparison = pd.DataFrame(
        [{"model": "LightGBM", "selected": True, "test_mae": 2.0, "test_rmse": 3.0}]
    )

    monkeypatch.setenv("ENERGY_AI_PROVIDER", "openai-compatible")
    response = dashboard._build_offline_agent_response(
        forecast,
        {"selected_model": "LightGBM", "horizon": "1h"},
        comparison,
    )

    assert response.provider == "mock"
    assert response.model == "offline-mock"
    assert response.content["peak_prediction"] == 120.0
    assert response.content["mean_interval_width"] == 30.0
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_dashboard.py::test_build_offline_agent_response_forces_mock_provider -q
```

Expected: FAIL because `_build_offline_agent_response` does not exist yet.

- [ ] **Step 3: Add the adapter imports and implementation**

Add these imports to `dashboard.py`:

```python
from src.agent import build_agent_context_from_frames
from src.ai_config import AISettings
from src.ai_provider import AgentResponse, PROVIDER_MOCK, build_provider
```

Add this helper below `_metric_value`:

```python
def _build_offline_agent_response(
    forecast: pd.DataFrame,
    metadata: dict[str, Any],
    comparison: pd.DataFrame,
) -> AgentResponse:
    context = build_agent_context_from_frames(
        metadata,
        forecast,
        comparison,
        recent_load_rows=[],
    )
    settings = AISettings(provider=PROVIDER_MOCK, model="offline-mock")
    return build_provider(settings).analyze(context)
```

The helper intentionally does not call `AISettings.from_env()`. Environment
configuration remains available to the CLI and future local-model integration,
but the current Dashboard path is explicitly offline.

- [ ] **Step 4: Run the focused test and verify it passes**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_dashboard.py::test_build_offline_agent_response_forces_mock_provider -q
```

Expected: PASS.

- [ ] **Step 5: Commit the adapter**

```powershell
git add dashboard.py tests\test_dashboard.py
git commit -m "feat: add offline dashboard agent adapter"
```

### Task 2: Render the AI operations interpretation

**Files:**
- Modify: `C:\Users\clt\Documents\Codex\2026-08-03\lie\work\energy-load-forecasting\dashboard.py`
- Test: `C:\Users\clt\Documents\Codex\2026-08-03\lie\work\energy-load-forecasting\tests\test_dashboard.py`

**Interfaces:**
- Consumes: `AgentResponse` from `_build_offline_agent_response`.
- Produces: `_show_agent_analysis(forecast, metadata, comparison) -> None`, rendered in the Forecast tab.

- [ ] **Step 1: Extend the fake Streamlit surface used by tests**

Add these no-op methods to `_FakeStreamlit`:

```python
def subheader(self, *_args, **_kwargs):
    pass

def write(self, *_args, **_kwargs):
    pass
```

- [ ] **Step 2: Write the rendering and failure tests**

Add this test:

```python
def test_show_agent_analysis_renders_offline_summary():
    fake = _FakeStreamlit()
    forecast = pd.DataFrame(
        {"step": [1, 2], "prediction": [100.0, 120.0], "p10": [90.0, 100.0], "p90": [110.0, 140.0]},
        index=pd.date_range("2025-01-01", periods=2, freq="h"),
    )
    comparison = pd.DataFrame([{"model": "LightGBM", "selected": True}])

    with patch.object(dashboard, "st", fake):
        dashboard._show_agent_analysis(
            forecast,
            {"selected_model": "LightGBM", "horizon": "1h"},
            comparison,
        )

    assert any("Offline mock" in message for message in fake.infos)
    assert not fake.warnings


def test_show_agent_analysis_warns_without_breaking_forecast(monkeypatch):
    fake = _FakeStreamlit()
    forecast = pd.DataFrame(
        {"step": [1], "prediction": [100.0], "p10": [90.0], "p90": [110.0]},
        index=pd.date_range("2025-01-01", periods=1, freq="h"),
    )
    comparison = pd.DataFrame([{"model": "LightGBM", "selected": True}])

    def raise_context_error(*_args, **_kwargs):
        raise ValueError("invalid context")

    monkeypatch.setattr(dashboard, "_build_offline_agent_response", raise_context_error)

    with patch.object(dashboard, "st", fake):
        dashboard._show_agent_analysis(forecast, {}, comparison)

    assert fake.warnings == ["AI operations interpretation unavailable: invalid context"]
```

- [ ] **Step 3: Run the new tests and verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_dashboard.py::test_show_agent_analysis_renders_offline_summary tests\test_dashboard.py::test_show_agent_analysis_warns_without_breaking_forecast -q
```

Expected: FAIL because `_show_agent_analysis` does not exist yet.

- [ ] **Step 4: Implement the rendering helper**

Add this helper below `_forecast_figure`:

```python
def _show_agent_analysis(
    forecast: pd.DataFrame,
    metadata: dict[str, Any],
    comparison: pd.DataFrame,
) -> None:
    st.subheader("AI operations interpretation")
    try:
        response = _build_offline_agent_response(forecast, metadata, comparison)
    except (TypeError, ValueError, DashboardReportError, RuntimeError) as exc:
        st.warning(f"AI operations interpretation unavailable: {exc}")
        return

    content = response.content
    st.info("Offline mock analysis is active. No external API or local model was called.")
    columns = st.columns(3)
    columns[0].metric("Provider", f"{response.provider} / {response.model}")
    columns[1].metric("Forecast peak", f"{float(content['peak_prediction']):.1f} kW")
    columns[2].metric("Mean uncertainty", f"{float(content['mean_interval_width']):.1f} kW")

    peak_timestamp = pd.to_datetime(content["peak_timestamp"]).strftime("%Y-%m-%d %H:%M")
    st.caption(f"Peak window: {peak_timestamp}")
    st.write("Operational recommendations")
    for recommendation in content.get("recommendations", []):
        st.write(f"- {recommendation}")
```

The helper uses the existing response fields and leaves the numeric plot
untouched. The `st.info` message is the visible status indicator, so users do
not have to infer provider state from color or a hidden configuration.

- [ ] **Step 5: Render the helper from the Forecast tab**

In `_show_forecast`, keep the loaded `comparison` variable for both the
existing metrics and the new Agent section, then call:

```python
_show_agent_analysis(forecast, metadata, comparison)
```

Place the call immediately after the forecast chart and before the download
button, so the interpretation is visually attached to the chart.

- [ ] **Step 6: Run focused dashboard tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_dashboard.py -q
```

Expected: all dashboard tests pass.

- [ ] **Step 7: Commit the UI integration**

```powershell
git add dashboard.py tests\test_dashboard.py
git commit -m "feat: show offline agent analysis in dashboard"
```

### Task 3: Verify the running application and future handoff

**Files:**
- Modify: `C:\Users\clt\Documents\Codex\2026-08-03\lie\work\energy-load-forecasting\README.md`
- Modify: `C:\Users\clt\Documents\Codex\2026-08-03\lie\work\energy-load-forecasting\docs\PORTFOLIO_GUIDE.md`
- Test: `C:\Users\clt\Documents\Codex\2026-08-03\lie\work\energy-load-forecasting\tests\test_dashboard.py`

**Interfaces:**
- Consumes: the Dashboard Agent section and the existing provider configuration documentation.
- Produces: reproducible local demo instructions that state the current provider is offline mock and show the later local-model environment variables.

- [ ] **Step 1: Add a dashboard integration assertion**

Update `test_main_renders_four_tabs_and_storage_disclaimer` to assert that the
existing main render includes the new offline status:

```python
assert any("Offline mock" in message for message in fake.infos)
```

- [ ] **Step 2: Run the complete test suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: all tests pass; the existing pandas datetime warning and pytest
cache-permission warning may remain.

- [ ] **Step 3: Run Streamlit AppTest**

Run:

```powershell
.\.venv\Scripts\python.exe -c "from streamlit.testing.v1 import AppTest; app=AppTest.from_file('dashboard.py').run(); print({'exceptions': len(app.exception), 'tabs': [tab.label for tab in app.tabs], 'warnings': len(app.warning), 'infos': len(app.info)})"
```

Expected: `exceptions` is `0`, the tabs are
`['Forecast', 'Model Comparison', 'Robustness', 'Storage']`, and at least one
info message contains the offline Agent status.

- [ ] **Step 4: Check the live service**

Run:

```powershell
(Invoke-WebRequest -UseBasicParsing "http://localhost:8501/_stcore/health").Content
```

Expected: `ok`.

- [ ] **Step 5: Document the current and future provider modes**

Update the Dashboard section in `README.md` and the demo path in
`docs/PORTFOLIO_GUIDE.md` with:

```text
The Forecast tab includes an offline mock Agent interpretation. It does not
call an external API. A future local or company-approved OpenAI-compatible
endpoint can be enabled with ENERGY_AI_PROVIDER, ENERGY_AI_BASE_URL, and
ENERGY_AI_MODEL after the endpoint is available.
```

- [ ] **Step 6: Commit and push the completed integration**

```powershell
git add dashboard.py tests\test_dashboard.py README.md docs\PORTFOLIO_GUIDE.md
git commit -m "feat: integrate offline agent analysis into dashboard"
git push origin main
```

Verify:

```powershell
git status --short --branch
git ls-remote origin refs/heads/main
```

The worktree should be clean and the remote commit should match the local
`main` commit.
