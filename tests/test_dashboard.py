from unittest.mock import patch

import pandas as pd

import dashboard
from src.dashboard_data import DashboardReportError


class _Container:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def metric(self, *_args, **_kwargs):
        pass


class _FakeStreamlit:
    def __init__(self):
        self.sidebar = _Container()
        self.column_config = type("ColumnConfig", (), {"NumberColumn": lambda *_args, **_kwargs: None})
        self.warnings = []
        self.infos = []
        self.tabs_seen = []

    def set_page_config(self, **_kwargs):
        pass

    def title(self, *_args, **_kwargs):
        pass

    def subheader(self, *_args, **_kwargs):
        pass

    def write(self, *_args, **_kwargs):
        pass

    def caption(self, *_args, **_kwargs):
        pass

    def header(self, *_args, **_kwargs):
        pass

    def selectbox(self, label, options, **_kwargs):
        return list(options)[0]

    def segmented_control(self, _label, _options, **_kwargs):
        return "Classical"

    def tabs(self, labels):
        self.tabs_seen = labels
        return [_Container() for _ in labels]

    def columns(self, count):
        return [_Container() for _ in range(count)]

    def plotly_chart(self, *_args, **_kwargs):
        pass

    def download_button(self, *_args, **_kwargs):
        pass

    def expander(self, *_args, **_kwargs):
        return _Container()

    def json(self, *_args, **_kwargs):
        pass

    def dataframe(self, *_args, **_kwargs):
        pass

    def warning(self, message):
        self.warnings.append(message)

    def info(self, message):
        self.infos.append(message)


def test_build_offline_agent_response_forces_mock_provider(monkeypatch):
    forecast = pd.DataFrame(
        {"step": [1, 2], "prediction": [100.0, 120.0], "p10": [90.0, 100.0], "p90": [110.0, 140.0]},
        index=pd.date_range("2025-01-01", periods=2, freq="h"),
    )
    comparison = pd.DataFrame(
        [{"model": "LightGBM", "selected": True, "test_mae": 2.0, "test_rmse": 3.0}]
    )

    monkeypatch.setenv("ENERGY_AI_PROVIDER", "openai-compatible")
    response = dashboard._build_offline_agent_response(
        forecast,
        {"selected_model": "LightGBM", "horizon": "1h"},
        comparison,
    )

    assert response.provider == "mock"
    assert response.model == "offline-mock"
    assert response.content["peak_prediction"] == 120.0
    assert response.content["mean_interval_width"] == 30.0


def test_show_agent_analysis_renders_offline_summary():
    fake = _FakeStreamlit()
    forecast = pd.DataFrame(
        {"step": [1, 2], "prediction": [100.0, 120.0], "p10": [90.0, 100.0], "p90": [110.0, 140.0]},
        index=pd.date_range("2025-01-01", periods=2, freq="h"),
    )
    comparison = pd.DataFrame([{"model": "LightGBM", "selected": True}])

    with patch.object(dashboard, "st", fake):
        dashboard._show_agent_analysis(
            forecast,
            {"selected_model": "LightGBM", "horizon": "1h"},
            comparison,
        )

    assert any("Offline mock" in message for message in fake.infos)
    assert not fake.warnings


def test_show_agent_analysis_warns_without_breaking_forecast(monkeypatch):
    fake = _FakeStreamlit()
    forecast = pd.DataFrame(
        {"step": [1], "prediction": [100.0], "p10": [90.0], "p90": [110.0]},
        index=pd.date_range("2025-01-01", periods=1, freq="h"),
    )
    comparison = pd.DataFrame([{"model": "LightGBM", "selected": True}])

    def raise_context_error(*_args, **_kwargs):
        raise ValueError("invalid context")

    monkeypatch.setattr(dashboard, "_build_offline_agent_response", raise_context_error)

    with patch.object(dashboard, "st", fake):
        dashboard._show_agent_analysis(forecast, {}, comparison)

    assert fake.warnings == ["AI operations interpretation unavailable: invalid context"]


def test_show_forecast_gru_survives_comparison_failure(monkeypatch):
    fake = _FakeStreamlit()
    forecast = pd.DataFrame(
        {"step": [1], "prediction": [100.0], "p10": [90.0], "p50": [100.0], "p90": [110.0]},
        index=pd.to_datetime(["2025-01-01 00:15:00"]),
    )
    metadata = {"test_metrics": {"MAE": 3.0, "RMSE": 4.0}}
    monkeypatch.setattr(dashboard, "_load_gru_forecast", lambda _horizon: forecast)
    monkeypatch.setattr(dashboard, "load_gru_metrics", lambda _meter, _horizon: metadata)
    monkeypatch.setattr(
        dashboard,
        "load_model_comparison",
        lambda _meter, _horizon: (_ for _ in ()).throw(DashboardReportError("comparison.csv missing")),
    )

    with (
        patch.object(dashboard, "st", fake),
        patch.object(fake, "plotly_chart", wraps=fake.plotly_chart) as plotly_chart,
        patch.object(fake, "download_button", wraps=fake.download_button) as download_button,
        patch.object(fake, "json", wraps=fake.json) as json,
    ):
        dashboard._show_forecast("1h", "GRU")

    assert plotly_chart.call_count == 1
    assert download_button.call_count == 1
    assert json.call_count == 1
    assert any("comparison.csv missing" in message for message in fake.warnings)


def test_forecast_figure_reserves_space_between_title_and_legend():
    forecast = pd.DataFrame(
        {"p10": [90.0, 91.0], "p50": [100.0, 101.0], "p90": [110.0, 111.0]},
        index=pd.date_range("2025-01-01", periods=2, freq="h"),
    )

    figure = dashboard._forecast_figure(forecast, "LightGBM forecast for MT_252 (1h)")

    assert figure.layout.margin.t >= 80
    assert figure.layout.title.y < figure.layout.legend.y
    assert figure.layout.title.x == 0
    assert figure.layout.legend.x == 0


def test_main_renders_four_tabs_and_storage_disclaimer():
    fake = _FakeStreamlit()
    forecast = pd.DataFrame(
        {"step": [1], "prediction": [100.0], "p10": [90.0], "p50": [100.0], "p90": [110.0]},
        index=pd.to_datetime(["2025-01-01 00:15:00"]),
    )
    comparison = pd.DataFrame(
        [{"model": "LightGBM", "configuration": "small", "validation_mae": 1.0, "validation_rmse": 2.0, "test_mae": 3.0, "test_rmse": 4.0, "selected": True, "training_seconds": 1.0}]
    )
    robustness = pd.DataFrame(
        [{"scenario": "clean", "affected_points": 0, "imputed_points": 0, "mae": 1.0, "rmse": 2.0, "mae_delta": 0.0, "mae_degradation_pct": 0.0}]
    )
    dispatch = pd.DataFrame(
        [{"forecast_load_kw": 100.0, "grid_import_kw": 90.0, "charge_kw": 0.0, "discharge_kw": 10.0, "soc": 0.5, "energy_price": 0.6}],
        index=pd.to_datetime(["2025-01-01 00:15:00"]),
    )
    gru = {"validation_metrics": {"MAE": 1.0, "RMSE": 2.0}, "test_metrics": {"MAE": 3.0, "RMSE": 4.0}}
    summary = {"results": [{"scenario": "p10", "strategy": "optimized", "total_energy_cost": 1.0, "cost_savings": 1.0, "peak_reduction_kw": 1.0, "battery_throughput_kwh": 1.0}]}

    with patch.object(dashboard, "st", fake), patch.object(dashboard, "load_forecast_report", return_value=(forecast, {})), patch.object(dashboard, "load_model_comparison", return_value=comparison), patch.object(dashboard, "load_gru_metrics", return_value=gru), patch.object(dashboard, "load_robustness_report", return_value=(robustness, {})), patch.object(dashboard, "load_storage_report", return_value=(dispatch, summary)):
        dashboard.main()

    assert fake.tabs_seen == ["Forecast", "Model Comparison", "Robustness", "Storage"]
    assert any("Offline mock" in message for message in fake.infos)
    assert any("Synthetic-demo" in message for message in fake.infos)


def test_forecast_unavailable_is_a_warning_not_a_traceback():
    fake = _FakeStreamlit()

    with patch.object(dashboard, "st", fake), patch.object(dashboard, "load_forecast_report", side_effect=DashboardReportError("forecast.csv missing")):
        dashboard._show_forecast("1h", "Classical")

    assert fake.warnings == ["Forecast report unavailable: forecast.csv missing"]
