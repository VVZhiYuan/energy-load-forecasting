# Deployment Guide

This project is ready for a read-only Streamlit Community Cloud deployment.
The deployed app uses committed report artifacts and the offline mock Agent
interpretation. It does not need API keys or a local model runtime.

Live deployment:
[https://energy-load-forecasting.streamlit.app/](https://energy-load-forecasting.streamlit.app/)

## Streamlit Community Cloud

Use these settings when creating the app:

| Field | Value |
| --- | --- |
| Repository | `VVZhiYuan/energy-load-forecasting` |
| Branch | `main` |
| Main file path | `dashboard.py` |
| Python version | `3.12` |
| Secrets | Leave empty for the current offline mock dashboard |

Deployment steps:

1. Open `https://share.streamlit.io`.
2. Click **Create app**.
3. Choose **Yup, I have an app**.
4. Select or paste the GitHub repository.
5. Set branch to `main` and main file path to `dashboard.py`.
6. Open **Advanced settings** and select Python `3.12`.
7. Leave secrets empty unless a future approved API endpoint is being enabled.
8. Deploy the app and copy the generated `streamlit.app` URL.
9. Add the final URL to `README.md` and your resume/portfolio page.

## Current Runtime Mode

The dashboard's Forecast tab uses the explicit offline mock Agent path:

```text
dashboard forecast artifacts -> AgentContext -> mock provider -> AgentResponse
```

Do not set `ENERGY_AI_PROVIDER=openai-compatible` in the cloud deployment until
you also have an approved reachable endpoint and a safe secret-management
setup. For the current portfolio demo, no secrets are required.

The Agent boundary is read-only. It can summarize evidence and make
human-approved recommendations, but it cannot modify a forecast, dispatch
equipment, or execute tools. The dashboard presents its `risk_level`,
`forecast_unchanged`, and `execution_enabled` metadata without operation
buttons. If Agent analysis fails, forecast charts and CSV downloads remain
available.

## Future Local Or Company API Handoff

When a local Ollama/vLLM model or company-approved OpenAI-compatible API is
available, configure these environment variables in the deployment platform's
secrets or environment settings:

```powershell
ENERGY_AI_PROVIDER = "openai-compatible"
ENERGY_AI_BASE_URL = "http://localhost:11434/v1"
ENERGY_AI_MODEL = "your-local-or-approved-model"
ENERGY_AI_API_KEY = "your-api-key-if-required"
ENERGY_AI_ALLOWED_HOSTS = "api.example.com"
ENERGY_AI_TIMEOUT_SECONDS = "30"
ENERGY_AI_MAX_RESPONSE_BYTES = "65536"
```

`ENERGY_AI_ALLOWED_HOSTS` is a comma-separated allowlist for remote provider
hostnames. Remote providers must use HTTPS and be listed explicitly. HTTP is
allowed only for loopback local runtimes (`localhost`, `127.0.0.1`, or `::1`).
`ENERGY_AI_TIMEOUT_SECONDS` must be a positive finite value, and
`ENERGY_AI_MAX_RESPONSE_BYTES` must be a positive byte limit; their defaults
are 30 seconds and 65536 bytes.

For a cloud deployment, `localhost` means the cloud container itself, not your
home computer. Do not point a cloud deployment at a home-computer `localhost`;
use a reachable, approved HTTPS endpoint instead. Hermes is not required by
the dashboard or deployed runtime. It remains an optional future adapter that
must be isolated behind the existing validated, read-only provider contract.

## Troubleshooting

- If the app shows missing report files, confirm the `reports/` directory is
  committed and present on GitHub.
- If dependency installation fails, check the Community Cloud logs and update
  root `requirements.txt`.
- If the app starts but charts are blank, reboot the app from Streamlit Cloud
  after dependency installation completes.
- If secrets are added later, never commit them to `.env`, `.streamlit`, or
  any source file.
