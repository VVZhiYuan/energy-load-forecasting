# Task 3 Report: Provider Transport Safety and Output Validation

## Status

Implemented Task 3 in the allowed provider/configuration scope.

## Changes

- Added `AISettings.allowed_hosts`, `timeout_seconds`, and `max_response_bytes`.
- Added environment parsing for:
  - `ENERGY_AI_ALLOWED_HOSTS`
  - `ENERGY_AI_TIMEOUT_SECONDS`
  - `ENERGY_AI_MAX_RESPONSE_BYTES`
- Kept defaults at an empty remote-host allowlist, 30 seconds, and 65536 bytes.
- Enforced provider endpoint safety:
  - remote endpoints require HTTPS and an explicit hostname allowlist entry;
  - HTTP is allowed only for `localhost`, `127.0.0.1`, and `::1`;
  - credentials, query strings, fragments, missing hosts, and unsupported schemes are rejected;
  - validation occurs before transport.
- Applied bounded timeout and response reads (`max_response_bytes + 1`).
- Kept outbound chat requests to `model` and `messages` only, with no tools, tool choice, MCP, or shell capability.
- Added a read-only system instruction requiring the fixed response contract.
- Validated remote model output through `validate_agent_response` before returning it.
- Updated disabled and mock providers to return deterministic complete contract responses.
- Updated provider tests and remote fixtures to use the explicit allowlist and complete response contract.

## Verification

Focused command:

```text
.venv\Scripts\python.exe -m pytest -q tests/test_ai_provider.py tests/test_agent.py tests/test_agent_contract.py
58 passed, 1 warning
```

Provider-only command:

```text
.venv\Scripts\python.exe -m pytest -q tests/test_ai_provider.py
27 passed, 1 warning
```

Full-suite command:

```text
.venv\Scripts\python.exe -m pytest -q
249 passed, 3 failed, 3 warnings
```

The three full-suite failures are out-of-scope consumers in `tests/test_analyze_latest.py` and `tests/test_dashboard.py` that still require the legacy mock response fields `forecast_steps`, `peak_prediction`, `mean_interval_width`, and `peak_timestamp`. The strict Agent contract requires exact top-level keys, and the task explicitly forbids Dashboard changes, so those fixtures/consumers cannot be updated within Task 3's allowed edit scope.

Warnings were the existing pandas datetime parsing warning plus pytest cache permission warnings from the workspace's `.pytest_cache` directory.

## Commit

The implementation and this report are committed together after final verification.

## Fix Round 1

### Findings Resolved

- Moved endpoint and transport-limit validation into
  `OpenAICompatibleAIProvider.__init__`. Direct instantiation now rejects an
  unsafe endpoint, missing remote-host allowlist, or invalid timeout before
  `analyze()` can construct or send a request. `build_provider` delegates to
  this shared constructor validation.
- Extended unsafe-endpoint tests to patch `urlopen` with a failure callback.
  Added direct-provider parameterized coverage for remote HTTP, an unallowlisted
  remote HTTPS host, and an invalid transport timeout. Each asserts `ValueError`
  while proving no network call occurs.

### Fix Verification

```text
.venv\Scripts\python.exe -m pytest -q tests\test_ai_provider.py tests\test_agent.py tests\test_agent_contract.py
61 passed, 1 warning
```

The warning is the workspace's existing pytest-cache permission warning.
