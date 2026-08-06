from dataclasses import asdict
import json

import numpy as np
import pandas as pd

from src.ai_config import AISettings
from src.ai_provider import AgentContext, AgentResponse
from src.agent import analyze_forecast, build_agent_context
from src.inference import ForecastRun


def make_run():
    observed_index = pd.date_range("2026-08-05 00:00:00", periods=5, freq="15min")
    observed = pd.Series([10.0, 12.5, 13.0, 11.5, 15.0], index=observed_index, name="load")
    forecast_index = pd.date_range(
        observed_index[-1] + pd.Timedelta(minutes=15),
        periods=3,
        freq="15min",
        name="forecast_timestamp",
    )
    forecast = pd.DataFrame(
        {
            "step": [1, 2, 3],
            "prediction": [16.0, 17.25, 18.5],
            "p10": [15.0, 16.0, 17.0],
            "p50": [16.0, 17.25, 18.5],
            "p90": [17.0, 18.5, 20.0],
            "point_model": ["Ridge", "Ridge", "Ridge"],
            "interval_method": ["residual_calibration"] * 3,
        },
        index=forecast_index,
    )
    comparison = pd.DataFrame(
        [
            {
                "model": "Ridge",
                "configuration": "alpha=0.1",
                "validation_mae": 1.25,
                "validation_rmse": 1.5,
                "test_mae": 1.75,
                "test_rmse": 2.0,
                "selected": True,
                "training_seconds": 0.125,
            },
            {
                "model": "LightGBM",
                "configuration": "medium",
                "validation_mae": 1.5,
                "validation_rmse": 1.8,
                "test_mae": 1.9,
                "test_rmse": 2.2,
                "selected": False,
                "training_seconds": 0.25,
            },
        ]
    )
    summary = {
        "source_label": "fixture",
        "horizon": "1h",
        "horizon_steps": 3,
        "selected_model": "Ridge",
        "selected_configuration": "alpha=0.1",
        "forecast_origin": observed_index[-1].isoformat(),
    }
    return ForecastRun(
        observed=observed,
        forecast=forecast,
        model_comparison=comparison,
        summary=summary,
    )


def test_build_agent_context_is_json_safe_and_preserves_forecast_values():
    run = make_run()
    original_forecast = run.forecast.copy(deep=True)

    context = build_agent_context(run, recent_points=3)

    assert isinstance(context, AgentContext)
    assert json.loads(json.dumps(context.__dict__))["summary"]["selected_model"] == "Ridge"
    assert context.summary["forecast_steps"] == 3
    assert context.summary["peak_step"] == 3
    assert context.summary["peak_prediction"] == 18.5
    assert context.summary["interval_width_at_peak"] == 3.0
    assert context.forecast_rows[1]["prediction"] == 17.25
    assert context.forecast_rows[2]["p90"] == 20.0
    assert context.comparison_rows[0]["selected"] is True
    assert [row["load"] for row in context.recent_load_rows] == [13.0, 11.5, 15.0]
    pd.testing.assert_frame_equal(run.forecast, original_forecast)


def test_build_agent_context_includes_all_recent_observations_when_requested_window_is_large():
    run = make_run()

    context = build_agent_context(run, recent_points=999)

    assert len(context.recent_load_rows) == len(run.observed)
    assert context.recent_load_rows[0]["timestamp"] == run.observed.index[0].isoformat()
    assert context.recent_load_rows[-1]["load"] == 15.0


def test_analyze_forecast_delegates_to_configured_provider(monkeypatch):
    run = make_run()
    settings = AISettings(
        provider="mock",
        base_url="https://example.invalid/v1",
        model="demo-model",
        api_key=None,
    )
    captured = {}

    class FakeProvider:
        def analyze(self, context):
            captured["context"] = context
            return AgentResponse(
                provider="mock",
                model="demo-model",
                content={"status": "mock"},
            )

    monkeypatch.setattr("src.agent.build_provider", lambda chosen: FakeProvider())

    response = analyze_forecast(run, settings=settings)

    assert response.provider == "mock"
    assert response.model == "demo-model"
    assert response.content["status"] == "mock"
    assert captured["context"].summary["selected_model"] == "Ridge"
    assert captured["context"].forecast_rows[0]["prediction"] == 16.0
    assert captured["context"].summary["peak_prediction"] == 18.5


def test_build_agent_context_serializes_pandas_null_like_values():
    run = make_run()
    run.summary["holiday_country"] = pd.NA
    run.summary["uncertainty_note"] = np.nan
    run.forecast.loc[:, "point_model"] = pd.NA
    run.forecast.loc[:, "interval_method"] = pd.NaT

    context = build_agent_context(run)

    payload = asdict(context)
    assert json.dumps(payload)
    assert payload["summary"]["holiday_country"] is None
    assert payload["summary"]["uncertainty_note"] is None
    assert payload["forecast_rows"][0]["point_model"] is None
    assert payload["forecast_rows"][0]["interval_method"] is None
