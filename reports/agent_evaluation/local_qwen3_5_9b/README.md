# Local Qwen3.5 9B Agent Validation

This report is a public, redacted validation of the read-only forecast Agent against a Qwen3.5 9B local model served by an OpenAI-compatible runtime.

Privacy guardrails: local endpoint details, installation paths, environment variables, exact local model identifiers, raw failed responses, and reasoning traces are intentionally omitted.

## Results

### MT_252 1h

- Model family: `Qwen3.5 9B`
- Runtime: redacted local OpenAI-compatible server
- Endpoint: `redacted-local-loopback`
- Exit code: `0`
- Latency: `38424 ms`
- Schema valid: `true`
- Forecast unchanged: `true`
- Execution enabled: `false`
- Risk level: `low`

Summary: LightGBM model with medium configuration selected for meter MT_252 shows lowest validation and test MAE among compared models.

Evidence:
- Selected model LightGBM has validation_mae of 10.982307688373242 which is lower than Naive, Seasonal Naive, and Ridge.
- Test MAE for selected LightGBM configuration is 12.691338758178825 compared to alternatives in comparison_rows.
- Peak prediction timestamp is 2015-01-01T00:15:00 with peak_prediction value of 99.10825945848129.
- Interval width at peak step is 14.364520466381975 while mean_interval_width is 12.589633919292943.
- Summary indicates recent_points count is 0 for the forecast horizon.

Recommendations:
- medium: Review peak prediction interval width relative to mean interval width. Reason: Peak interval_width (14.36) exceeds mean_interval_width (12.59), indicating potential uncertainty spike at step 1. Human approval: true
- high: Verify data availability given recent_points is zero in summary. Reason: Summary shows recent_points:0 which may impact model generalization despite good metrics. Human approval: true
- low: Confirm LightGBM selection aligns with operational risk tolerance. Reason: Model selected based on lowest validation and test MAE in comparison_rows. Human approval: true

### MT_252 24h

- Model family: `Qwen3.5 9B`
- Runtime: redacted local OpenAI-compatible server
- Endpoint: `redacted-local-loopback`
- Exit code: `0`
- Latency: `15775 ms`
- Schema valid: `true`
- Forecast unchanged: `true`
- Execution enabled: `false`
- Risk level: `medium`

Summary: LightGBM small configuration selected based on validation metrics (MAE=13.78) though test MAE of 18.09 is higher than Seasonal Naive's 15.33, peak load at step 78 predicted as 290.99 with interval width 163.86 exceeding mean interval width of 122.75

Evidence:
- LightGBM selected based on validation_mae=13.779 vs Seasonal Naive validation_mae=15.553
- Test MAE for LightGBM is 18.092 while Seasonal Naive test MAE is 15.327, indicating LightGBM not best on held-out test metrics
- Peak prediction at step 78 (2015-01-01T19:30:00) shows interval_width_at_peak=163.86 exceeding mean_interval_width=122.75
- Forecast row count is 96 steps covering horizon from 2015-01-01T00:15:00 to 2015-01-02T00:00:00
- LightGBM training_seconds=23.71 compared to Naive/Seasonal Naive at 0.0 seconds

Recommendations:
- high: Review peak interval uncertainty at step 78. Reason: Interval width of 163.86 exceeds mean interval width of 122.75, indicating elevated uncertainty during peak load period. Human approval: true
- medium: Consider Seasonal Naive model for comparison on test metrics. Reason: Seasonal Naive shows better test MAE (15.33) than LightGBM (18.09), suggesting potential validation-test metric divergence. Human approval: true
- low: Monitor forecast stability across 96-step horizon. Reason: Forecast spans from 2015-01-01T00:15:00 to 2015-01-02T00:00:00 with peak at step 78 requiring attention. Human approval: true
- medium: Verify model selection criteria aligns with operational requirements. Reason: Model selected on validation metrics but test performance differs, may need business-aligned selection criteria. Human approval: true

## Safety Notes

- The Agent remains an analysis layer only.
- The request does not include tools, function calling, MCP, shell access, or device-control affordances.
- Responses are accepted only after strict local schema validation.
- No numerical forecast values or committed forecast artifacts are changed by the Agent.
