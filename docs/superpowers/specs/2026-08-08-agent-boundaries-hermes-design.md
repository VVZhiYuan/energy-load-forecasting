# Agent Boundary and Hermes Handoff Design

## Goal

Harden the project's AI Agent layer for portfolio use while preserving the
forecasting models as the only source of numeric truth. The Agent may explain
forecast results and produce operational recommendations, but it must not
modify forecasts or execute real-world operations.

## Scope

This design covers the current read-only Agent provider layer and the future
handoff point for Hermes Agent. It does not add a live device controller,
storage dispatcher, pricing executor, Shell tool, MCP tool, browser tool, or
autonomous workflow.

## Architecture and Responsibilities

The forecasting layer remains responsible for data loading, feature
engineering, model training, model selection, point forecasts, and uncertainty
intervals.

The Agent layer receives an explicit `AgentContext`, interprets the forecast,
and returns a validated structured response. It may:

- identify load peaks and high-risk windows;
- explain uncertainty intervals and model disagreement;
- summarize evidence from the supplied forecast context;
- recommend energy-management actions for human review.

The Agent must not:

- modify or overwrite numeric forecast values;
- retrain or select forecasting models;
- modify files, datasets, reports, or project configuration;
- execute Shell, Python, database, MCP, browser, or device-control tools;
- directly control storage, equipment, prices, or grid dispatch.

The Dashboard continues to use the offline mock provider by default. The
numeric forecast and its visualizations remain usable if the Agent is disabled
or unavailable.

## Context Contract

The provider receives only JSON-safe, bounded forecast data:

```json
{
  "summary": {
    "meter": "MT_252",
    "horizon": "24h",
    "selected_model": "LightGBM",
    "forecast_origin": "2026-08-07T00:00:00",
    "peak_timestamp": "2026-08-07T09:00:00",
    "peak_prediction": 123.4,
    "mean_interval_width": 18.2
  },
  "forecast_rows": [],
  "comparison_rows": [],
  "recent_load_rows": []
}
```

Context rules:

- forecast rows are bounded to the selected horizon and at most 96 rows;
- recent observed-load rows are bounded to at most 96 rows;
- only load, timestamp, forecast, model, and error-related fields are passed;
- secrets, environment values, machine paths, and unrelated free text are not
  included;
- context values are treated as untrusted data and never as executable
  instructions.

## Response Contract

The model response must validate against this logical schema:

```json
{
  "status": "ok",
  "summary": "Future load remains broadly stable.",
  "risk_level": "low",
  "evidence": ["Peak occurs near 09:00"],
  "recommendations": [
    {
      "action": "Review flexible loads before the peak window",
      "reason": "The forecast indicates higher dispatch pressure",
      "priority": "medium",
      "requires_human_approval": true
    }
  ],
  "forecast_unchanged": true,
  "execution_enabled": false
}
```

Validation rules:

- `status` is one of `ok`, `disabled`, or `error`;
- `risk_level` is one of `low`, `medium`, or `high`;
- `evidence` contains at most five strings;
- `recommendations` contains at most five objects;
- every recommendation includes `requires_human_approval: true`;
- `forecast_unchanged` must be `true`;
- `execution_enabled` must be `false` in the current project;
- missing fields, invalid types, unsupported values, or execution-oriented
  payloads cause the response to be rejected.

## Provider and API Safety

The provider modes remain:

- `disabled`: deterministic response and no network access;
- `mock`: deterministic offline demonstration response;
- `openai-compatible`: optional future endpoint for a hosted or local model.

The openai-compatible adapter remains analysis-only. It does not send tool
definitions or accept tool calls. The implementation must enforce a finite
request timeout, bounded response size, and bounded output content. Provider
configuration must validate the endpoint scheme and host policy before making
requests; the default deployment must remain offline.

API keys are read only from environment or deployment secrets and are never
written to reports, logs, screenshots, or committed files.

## Failure Handling and Approval

Agent failures are fail-closed:

- disabled provider returns a disabled status;
- network, timeout, malformed JSON, schema, or response-size failures return a
  safe provider error;
- the Dashboard displays a warning and continues to render the forecast;
- no failure path retries as an executable action.

All recommendations carry `requires_human_approval: true`. The current
project records recommendations only and has no execution path. A future
operations executor must be a separately permissioned component with explicit
approval, audit logging, simulation mode, and an independent integration test
suite.

## Hermes Handoff

Hermes Agent is not added as a runtime dependency of the Streamlit dashboard.
It is a future optional orchestration layer. A future adapter may allow Hermes
to read committed forecast reports and request the same validated analysis
contract, but it must not inherit file-write, Shell, MCP, browser, memory,
cron, or device-control permissions by default.

If Hermes is evaluated later, it must run in an isolated environment with a
project-specific safe root, explicit tool allowlists, disabled autonomous
execution, and human approval for any operation-oriented action. The Hermes
integration must be opt-in and must not change the project's default offline
behavior.

## Testing and Acceptance

The implementation is complete when:

1. default configuration performs no external AI request;
2. mock and disabled providers satisfy the response contract;
3. malformed, oversized, or unsafe provider responses are rejected;
4. Agent analysis cannot mutate forecast inputs or outputs;
5. no Agent path executes files, Shell, MCP, browser, or equipment actions;
6. the Dashboard remains usable when Agent analysis fails;
7. existing project tests remain green;
8. Hermes is not imported, started, or required unless an explicit future
   adapter is enabled.

