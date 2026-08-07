# Deployment Guide

This project is ready for a read-only Streamlit Community Cloud deployment.
The deployed app uses committed report artifacts and the offline mock Agent
interpretation. It does not need API keys or a local model runtime.

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

## Future Local Or Company API Handoff

When a local Ollama/vLLM model or company-approved OpenAI-compatible API is
available, configure these environment variables in the deployment platform's
secrets or environment settings:

```powershell
ENERGY_AI_PROVIDER = "openai-compatible"
ENERGY_AI_BASE_URL = "http://localhost:11434/v1"
ENERGY_AI_MODEL = "your-local-or-approved-model"
ENERGY_AI_API_KEY = "your-api-key-if-required"
```

For a cloud deployment, `localhost` means the cloud container itself, not your
home computer. Use a reachable HTTPS endpoint for hosted deployments.

## Troubleshooting

- If the app shows missing report files, confirm the `reports/` directory is
  committed and present on GitHub.
- If dependency installation fails, check the Community Cloud logs and update
  root `requirements.txt`.
- If the app starts but charts are blank, reboot the app from Streamlit Cloud
  after dependency installation completes.
- If secrets are added later, never commit them to `.env`, `.streamlit`, or
  any source file.
