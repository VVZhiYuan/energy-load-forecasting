# Offline Agent Dashboard Integration Design

## Goal

Expose the existing forecast Agent layer inside the Streamlit Forecast tab so
the portfolio demo visibly connects AI reasoning with energy operations. The
dashboard must remain fully usable on a company computer without a model
runtime, network access, or API key.

## Scope

The Forecast tab will render an `AI operations interpretation` section after
the forecast chart. It will use the currently selected horizon and forecast
family, and show:

- provider state and model label;
- forecast peak timestamp and predicted load;
- average P10-P90 interval width;
- deterministic operational recommendations.

The section will explicitly identify the current result as an offline mock
analysis. It will not retrain models, alter report artifacts, or add a chat
interface.

## Architecture

The dashboard will build an `AgentContext` from the already loaded forecast
frame, metadata, model comparison frame, and any available recent-load rows.
It will call the existing provider contract with an explicit
`AISettings(provider="mock")` configuration. This keeps the dashboard's
behavior deterministic and prevents accidental network calls.

The existing `openai-compatible` provider remains unchanged and remains the
future handoff point for an approved hosted API or a local Ollama/vLLM
endpoint. Enabling that provider later will be a configuration change outside
the default dashboard path; no API key will be added to source files or
committed artifacts.

## Data Flow

```text
committed forecast artifacts
        |
        v
dashboard loaders -> AgentContext -> mock provider -> AgentResponse
        |                                      |
        +-------------- forecast chart --------+
                                               v
                                     AI operations section
```

The numeric forecast and chart remain independent of Agent availability. If
context construction or Agent analysis fails, the dashboard will display a
clear warning and continue rendering the forecast and download controls.

## UI Behavior

The section will use compact metrics and a short recommendation list rather
than a nested card layout. The provider badge/message will make the offline
state clear, while the operational values will be traceable to the forecast
data already shown above. The layout must keep labels and values readable on
narrow Streamlit columns and must not reintroduce chart title/legend overlap.

## Testing

Add focused tests that verify:

1. the Forecast tab calls the offline mock provider with the selected forecast
   context;
2. the rendered Agent response includes peak and uncertainty values;
3. Agent failures become a dashboard warning without a traceback;
4. the existing four-tab dashboard and current forecast-layout regression
   tests continue to pass.

## Future Model Handoff

When a local model is available at home, configure the existing
OpenAI-compatible adapter with environment variables such as
`ENERGY_AI_PROVIDER=openai-compatible`,
`ENERGY_AI_BASE_URL=http://localhost:<port>/v1`, and
`ENERGY_AI_MODEL=<local-model>`. The dashboard integration should continue to
consume the same `AgentResponse` contract, so the UI does not need to know
whether the provider is mock, hosted, or local.
