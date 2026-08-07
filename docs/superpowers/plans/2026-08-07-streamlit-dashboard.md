# Offline Energy Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build a local read-only Streamlit dashboard for the project's forecast, GRU, robustness, and storage artifacts.

**Architecture:** dashboard.py stays a thin Streamlit entrypoint. src/dashboard_data.py validates and loads report artifacts through cached functions. The dashboard renders each workflow in one tab from immutable reports and never invokes model training.

**Tech Stack:** Streamlit, Plotly, pandas, pytest, existing reports.

## Global Constraints

- Add streamlit>=1.40.0 to requirements.txt.
- Read artifacts only from reports/; never trigger prediction, training, robustness, or optimization commands from the UI.
- Keep source fixed to existing MT_252 reports for this portfolio release.
- Validate all required report columns and show an in-app unavailable state for missing or malformed reports.
- Preserve the synthetic_demo disclaimer in storage charts and metrics.
- Dashboard must fit desktop and mobile without clipped labels.

### Task 1: Cached Artifact Loaders

**Files:**
- Create: src/dashboard_data.py
- Create: tests/test_dashboard_data.py
- Modify: requirements.txt

**Interfaces:**
- Produces: DashboardReportError, load_forecast_report, load_model_comparison, load_robustness_report, load_storage_report, and load_gru_metrics.
- Consumes: report directories rooted at src.config.REPORTS_DIR.

- [ ] **Step 1: Write failing loader tests**

Use temporary report directories. Test valid forecast CSV/JSON loads as a timestamp-indexed DataFrame and metadata dict. Test missing forecast.csv, invalid datetime, missing p10/p50/p90, and malformed JSON raise DashboardReportError with the filename. Test storage loader returns p50 optimized dispatch and summary; test robustness loader accepts required metric columns.

- [ ] **Step 2: Implement validated cached loaders**

Use st.cache_data only inside a lazy Streamlit import helper so the module remains testable. Read JSON as an object, parse timestamp columns, require finite numeric P10/P50/P90 and ordered intervals. Return copied DataFrames to avoid cache mutation. Map every OSError, JSON decode, and value problem to DashboardReportError naming its source file.

- [ ] **Step 3: Run and commit Task 1**

Run: .\.venv\Scripts\python.exe -m pytest tests\test_dashboard_data.py -q
Commit: git add requirements.txt src\dashboard_data.py tests\test_dashboard_data.py && git commit -m "feat: load dashboard report artifacts"

### Task 2: Streamlit Workbench

**Files:**
- Create: dashboard.py
- Create: tests/test_dashboard.py

**Interfaces:**
- Consumes: dashboard_data loaders.
- Produces: a Streamlit page with Forecast, Model Comparison, Robustness, and Storage tabs.

- [ ] **Step 1: Write failing dashboard smoke tests**

Patch Streamlit with a recording fake. Verify dashboard.main renders four tabs, uses selected horizon and forecast family, and catches DashboardReportError into st.warning/st.error without a traceback. Test that the storage section includes synthetic-demo text.

- [ ] **Step 2: Implement the dashboard entrypoint**

Set page configuration and a compact operational title. Sidebar controls are horizon, forecast family, and storage scenario. Use tabs:
- Forecast: metric strip plus Plotly load and P10/P50/P90 chart, CSV download.
- Model Comparison: filtered metric table and grouped test-MAE/test-RMSE Plotly bars.
- Robustness: sorted degradation bar chart and scenario table.
- Storage: P50 or selected quantile dispatch, grid/load chart, battery power chart, SOC/tariff chart, cost and peak metrics, synthetic_demo warning, CSV download.

Each tab independently handles DashboardReportError so another view remains usable. Avoid training buttons and external APIs.

- [ ] **Step 3: Run and commit Task 2**

Run: .\.venv\Scripts\python.exe -m pytest tests\test_dashboard_data.py tests\test_dashboard.py -q
Commit: git add dashboard.py tests\test_dashboard.py && git commit -m "feat: add energy operations dashboard"

### Task 3: Visual Verification And Documentation

**Files:**
- Modify: README.md
- Modify: reports only if Streamlit generates no artifacts; otherwise none.

- [ ] **Step 1: Install and run Streamlit**

Run: .\.venv\Scripts\python.exe -m pip install -r requirements.txt
Run: .\.venv\Scripts\python.exe -m streamlit run dashboard.py --server.headless true --server.port 8501

Use Playwright to open localhost:8501, switch tabs and controls, and capture desktop/mobile screenshots. Verify no failed artifact state appears for committed MT_252 reports and text remains readable.

- [ ] **Step 2: Document local usage**

Add README dashboard command, purpose, four available views, local URL, and explicit read-only behavior. Mark dashboard complete in status/roadmap. Link to Streamlit's official installation guide.

- [ ] **Step 3: Full regression and publish**

Run: .\.venv\Scripts\python.exe -m pytest -q
Run: git diff --check
Commit: git add README.md && git commit -m "docs: publish energy dashboard"
Publish: git push origin main; git status --short --branch; git ls-remote origin refs/heads/main

## Plan Self-Review

- Spec coverage: loaders, validation, all four workflows, unavailable states, visual verification, documentation, and publication are assigned.
- No training or external API path is included.
- Public report contracts are read-only and unchanged.

