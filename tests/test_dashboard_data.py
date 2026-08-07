import json

import pandas as pd
import pytest

from src.dashboard_data import (
    DashboardReportError,
    load_forecast_report,
    load_gru_metrics,
    load_model_comparison,
    load_robustness_report,
    load_storage_report,
)


def _write_json(path, contents):
    path.write_text(json.dumps(contents), encoding="utf-8")


def _write_forecast_report(reports_dir, *, horizon="24h"):
    report_dir = reports_dir / "predictions" / "MT_252" / horizon
    report_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "forecast_timestamp": ["2025-01-01 00:15:00", "2025-01-01 00:30:00"],
            "step": [1, 2],
            "prediction": [101.0, 102.0],
            "p10": [90.0, 91.0],
            "p50": [100.0, 101.0],
            "p90": [110.0, 111.0],
        }
    ).to_csv(report_dir / "forecast.csv", index=False)
    _write_json(report_dir / "summary.json", {"horizon": horizon, "selected_model": "LightGBM"})
    return report_dir


def _write_comparison_report(reports_dir, *, horizon="24h"):
    report_dir = reports_dir / "predictions" / "MT_252" / horizon
    report_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "model": ["LightGBM"],
            "configuration": ["small"],
            "validation_mae": [1.0],
            "validation_rmse": [2.0],
            "test_mae": [3.0],
            "test_rmse": [4.0],
            "selected": [True],
            "training_seconds": [5.0],
        }
    ).to_csv(report_dir / "model_comparison.csv", index=False)


def _write_robustness_report(reports_dir, *, horizon="24h"):
    report_dir = reports_dir / "robustness" / "MT_252" / horizon
    report_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "scenario": ["clean"],
            "scenario_kind": ["clean"],
            "forecast_origin": ["2025-01-01 00:00:00"],
            "horizon": [horizon],
            "horizon_steps": [96],
            "selected_model": ["LightGBM"],
            "affected_points": [0],
            "imputed_points": [0],
            "mae": [1.0],
            "rmse": [2.0],
            "mape": [3.0],
            "clean_mae": [1.0],
            "mae_delta": [0.0],
            "mae_degradation_pct": [0.0],
        }
    ).to_csv(report_dir / "robustness_metrics.csv", index=False)
    _write_json(report_dir / "robustness_summary.json", {"horizon": horizon})


def _write_storage_report(reports_dir, *, horizon="24h"):
    report_dir = reports_dir / "optimization" / "MT_252" / horizon
    report_dir.mkdir(parents=True)
    rows = []
    for scenario in ("p10", "p50"):
        for strategy in ("no_storage", "optimized"):
            rows.append(
                {
                    "forecast_timestamp": "2025-01-01 00:15:00",
                    "scenario": scenario,
                    "strategy": strategy,
                    "forecast_load_kw": 100.0,
                    "grid_import_kw": 90.0,
                    "charge_kw": 0.0,
                    "discharge_kw": 10.0,
                    "battery_energy_kwh": 250.0,
                    "soc": 0.5,
                    "tariff_period": "off_peak",
                    "energy_price": 0.6,
                    "interval_energy_cost": 13.5,
                }
            )
    pd.DataFrame(rows).to_csv(report_dir / "dispatch.csv", index=False)
    _write_json(report_dir / "optimization_summary.json", {"primary_scenario": "p50"})


def _write_gru_metrics(reports_dir, *, horizon="24h"):
    report_dir = reports_dir / "deep_learning" / "MT_252" / horizon
    report_dir.mkdir(parents=True)
    _write_json(
        report_dir / "metrics.json",
        {
            "horizon": horizon,
            "validation_metrics": {"MAE": 1.0, "RMSE": 2.0},
            "test_metrics": {"MAE": 3.0, "RMSE": 4.0},
        },
    )


def test_load_forecast_report_returns_timestamp_indexed_frame_and_metadata(tmp_path):
    _write_forecast_report(tmp_path)

    forecast, summary = load_forecast_report("MT_252", "24h", reports_dir=tmp_path)

    assert isinstance(forecast.index, pd.DatetimeIndex)
    assert list(forecast.columns) == ["step", "prediction", "p10", "p50", "p90"]
    assert summary["selected_model"] == "LightGBM"
    forecast.loc[:, "p50"] = 0.0
    summary["selected_model"] = "changed"
    fresh_forecast, fresh_summary = load_forecast_report("MT_252", "24h", reports_dir=tmp_path)
    assert fresh_forecast["p50"].iloc[0] == 100.0
    assert fresh_summary["selected_model"] == "LightGBM"


def test_load_forecast_report_maps_missing_csv_to_artifact_error(tmp_path):
    report_dir = tmp_path / "predictions" / "MT_252" / "24h"
    report_dir.mkdir(parents=True)
    _write_json(report_dir / "summary.json", {})

    with pytest.raises(DashboardReportError, match="forecast.csv"):
        load_forecast_report("MT_252", "24h", reports_dir=tmp_path)


def test_load_forecast_report_rejects_invalid_datetime(tmp_path):
    report_dir = _write_forecast_report(tmp_path, horizon="bad-datetime")
    frame = pd.read_csv(report_dir / "forecast.csv")
    frame.loc[0, "forecast_timestamp"] = "not-a-timestamp"
    frame.to_csv(report_dir / "forecast.csv", index=False)

    with pytest.raises(DashboardReportError, match="forecast.csv"):
        load_forecast_report("MT_252", "bad-datetime", reports_dir=tmp_path)


def test_load_forecast_report_rejects_missing_interval_column(tmp_path):
    report_dir = _write_forecast_report(tmp_path, horizon="missing-interval")
    frame = pd.read_csv(report_dir / "forecast.csv").drop(columns="p90")
    frame.to_csv(report_dir / "forecast.csv", index=False)

    with pytest.raises(DashboardReportError, match="forecast.csv"):
        load_forecast_report("MT_252", "missing-interval", reports_dir=tmp_path)


def test_load_forecast_report_maps_malformed_json_to_artifact_error(tmp_path):
    report_dir = _write_forecast_report(tmp_path, horizon="bad-json")
    (report_dir / "summary.json").write_text("{", encoding="utf-8")

    with pytest.raises(DashboardReportError, match="summary.json"):
        load_forecast_report("MT_252", "bad-json", reports_dir=tmp_path)


def test_load_model_comparison_validates_required_finite_metrics(tmp_path):
    _write_comparison_report(tmp_path)

    comparison = load_model_comparison("MT_252", "24h", reports_dir=tmp_path)

    assert comparison.loc[0, "model"] == "LightGBM"


def test_load_robustness_report_accepts_required_metric_columns(tmp_path):
    _write_robustness_report(tmp_path)

    metrics, summary = load_robustness_report("MT_252", "24h", reports_dir=tmp_path)

    assert isinstance(metrics["forecast_origin"].iloc[0], pd.Timestamp)
    assert summary["horizon"] == "24h"


def test_load_storage_report_returns_selected_optimized_dispatch_and_summary(tmp_path):
    _write_storage_report(tmp_path)

    dispatch, summary = load_storage_report("MT_252", "24h", reports_dir=tmp_path)

    assert isinstance(dispatch.index, pd.DatetimeIndex)
    assert set(dispatch["scenario"]) == {"p50"}
    assert set(dispatch["strategy"]) == {"optimized"}
    assert summary["primary_scenario"] == "p50"


def test_load_gru_metrics_validates_required_finite_values(tmp_path):
    _write_gru_metrics(tmp_path)

    metrics = load_gru_metrics("MT_252", "24h", reports_dir=tmp_path)

    assert metrics["test_metrics"]["RMSE"] == 4.0
