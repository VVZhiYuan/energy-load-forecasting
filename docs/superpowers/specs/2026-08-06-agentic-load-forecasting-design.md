# Agentic Load Forecasting Design

## Goal

Extend the load forecasting portfolio project with an AI Agent layer that can
explain forecast results, identify operational risks, and produce energy
management recommendations without replacing the numerically evaluated load
forecasting models.

## Design

The project will use a two-layer hybrid design:

1. The existing forecasting layer remains responsible for numerical predictions
   and uncertainty scenarios. It continues to select and refit Naive, Seasonal
   Naive, Ridge, or LightGBM models using chronological validation.
2. The Agent layer consumes a small, explicit forecast context and uses tools or
   an LLM provider to interpret the result. It may report peak timing, interval
   width, model disagreement, and operational recommendations. It must not
   silently overwrite the numeric forecast.

The provider boundary is OpenAI-compatible but provider-neutral. The default
provider is `disabled`, so the repository runs on the company computer without
network access, a model download, or an API key. `mock` provides deterministic
offline demonstrations. `openai-compatible` sends a chat-completions request
to a configurable base URL, allowing a hosted API or a local Ollama/vLLM
server to be used later by changing environment variables.

## Data Flow

```text
ForecastRun -> AgentContext -> AIProvider -> AgentResponse -> JSON artifact
```

The context contains only JSON-safe summary fields, the forecast trajectory,
model comparison metrics, and recent observed-load values. API keys are read
from environment variables and never written to reports.

## Error Handling

- Invalid provider configuration fails before any network request.
- Disabled and mock providers never make network requests.
- Network, HTTP, malformed JSON, and missing response-content errors become a
  clear provider error at the CLI boundary.
- The numeric forecast remains usable when the Agent is disabled or unavailable.

## Success Criteria

- Existing forecasting tests remain green.
- The default configuration performs no external AI call.
- A deterministic mock analysis can be generated from a saved forecast report.
- A future OpenAI-compatible endpoint can be configured without changing the
  forecasting modules.
- The README documents the architecture, environment variables, commands,
  and the local-model handoff.

