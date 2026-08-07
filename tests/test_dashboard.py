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


def test_main_renders_four_tabs_and_storage_disclaimer():
    fake = _FakeStreamlit()
    forecast = pd.DataFrame(
        {"prediction": [100.0], "p10": [90.0], "p50": [100.0], "p90": [110.0]},
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
    assert any("Synthetic-demo" in message for message in fake.infos)


def test_forecast_unavailable_is_a_warning_not_a_traceback():
    fake = _FakeStreamlit()

    with patch.object(dashboard, "st", fake), patch.object(dashboard, "load_forecast_report", side_effect=DashboardReportError("forecast.csv missing")):
        dashboard._show_forecast("1h", "Classical")

    assert fake.warnings == ["Forecast report unavailable: forecast.csv missing"]
