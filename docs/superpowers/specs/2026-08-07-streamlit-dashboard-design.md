# Offline Energy Operations Dashboard Design

## Goal

Create a local Streamlit dashboard that presents the completed electricity-load forecasting, GRU benchmark, robustness, and storage-optimization artifacts in one portfolio-ready workflow.

## Scope

The dashboard reads committed CSV and JSON report artifacts. It never retrains models, downloads data, calls an AI API, or modifies reports during a browser session. Missing optional report families render a clear unavailable state instead of crashing the app.

The four views are:

1. Forecast: LightGBM or GRU P10/P50/P90 trajectory, selected horizon, and forecast metadata.
2. Model comparison: validation/test MAE and RMSE for Naive, Seasonal Naive, Ridge, LightGBM, and GRU when present.
3. Robustness: scenario MAE/RMSE deltas and degradation ranking for the selected horizon.
4. Storage: P50 no-storage, rule-based, and MILP strategies with cost, peak, SOC, and dispatch charts.

## Inputs And Controls

The sidebar provides source meter MT_252, horizon 1h/24h, forecast family Classical/GRU, and storage scenario p10/p50/p90. The app maps these selections to fixed report paths under reports/. It shows the selected report timestamp, model, device, and assumption source.

## Data Safety

Use st.cache_data for read-only CSV/JSON loaders. Validate required columns, datetime indexes, finite numeric values, and non-empty reports before plotting. Do not infer missing metrics or silently substitute a different horizon. Synthetic tariff and battery parameters must remain visibly labelled in the storage view.

## Visual Design

Use a restrained operational dashboard: compact title and status strip, four metric columns, tabs for the four workflows, and Plotly charts with clear axis units. Forecast views show load and P10/P50/P90. Model comparison shows grouped test MAE/RMSE bars. Robustness shows scenario degradation bars. Storage shows grid import versus load, battery power, and SOC/price. Tables are limited to useful rows and support download of the selected CSV.

## Error Handling

A missing report shows a warning naming the expected path and a short action message. Malformed JSON/CSV shows an error with the filename. The app remains usable for other views when one report family is missing. No stack trace is shown in the main UI.

## Acceptance Criteria

- streamlit run dashboard.py starts locally.
- Sidebar controls change the displayed report family and horizon without retraining.
- Forecast, comparison, robustness, and storage views render from real committed artifacts.
- Missing reports and malformed inputs are handled without app termination.
- At least one automated smoke test verifies app imports and core loaders.
- Existing full pytest suite remains green.
- README documents installation and local run commands.

## Portfolio Positioning

The dashboard demonstrates an end-to-end workflow: AI load forecast, uncertainty comparison, robustness evidence, and forecast-driven battery dispatch. It is a read-only decision-support prototype, not a production grid-control interface.

