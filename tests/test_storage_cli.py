import json

import numpy as np
import pandas as pd

import optimize_storage


def make_forecast_frame() -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=96, freq="15min")
    base = np.linspace(100.0, 195.0, num=96)
    return pd.DataFrame(
        {"p10": base, "p50": base + 10.0, "p90": base + 20.0}, index=index
    )


def write_forecast_report(directory, *, horizon="24h"):
    directory.mkdir()
    make_forecast_frame().to_csv(directory / "forecast.csv")
    (directory / "summary.json").write_text(
        json.dumps(
            {"source_label": "fixture_meter", "horizon": horizon, "selected_model": "Ridge"}
        ),
        encoding="utf-8",
    )


def test_parser_defaults_to_portfolio_battery():
    args = optimize_storage.build_parser().parse_args([])

    assert args.capacity_kwh == 500.0
    assert args.max_charge_kw == 100.0
    assert args.max_discharge_kw == 100.0


def test_main_writes_complete_optimization_report(tmp_path, capsys):
    forecast_dir = tmp_path / "forecast"
    write_forecast_report(forecast_dir)
    output_dir = tmp_path / "optimization"

    assert optimize_storage.main(
        ["--forecast-dir", str(forecast_dir), "--output-dir", str(output_dir)]
    ) == 0

    dispatch_path = output_dir / "dispatch.csv"
    summary_path = output_dir / "optimization_summary.json"
    chart_path = output_dir / "storage_dispatch.png"
    assert dispatch_path.is_file()
    assert summary_path.is_file()
    assert chart_path.is_file()
    assert chart_path.stat().st_size > 0
    assert len(pd.read_csv(dispatch_path)) == 96 * 9
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["source_label"] == "fixture_meter"
    assert summary["solver_method"] == "scipy_highs_linprog"
    assert len(summary["results"]) == 9
    assert "Complete" in capsys.readouterr().out


def test_main_rejects_non_24_hour_source_report(tmp_path, capsys):
    forecast_dir = tmp_path / "forecast"
    write_forecast_report(forecast_dir, horizon="1h")

    assert optimize_storage.main(["--forecast-dir", str(forecast_dir)]) == 2

    assert "requires a 24h" in capsys.readouterr().err


def test_main_returns_two_for_invalid_forecast(tmp_path, capsys):
    forecast_dir = tmp_path / "forecast"
    forecast_dir.mkdir()
    pd.DataFrame({"p50": [100.0]}).to_csv(forecast_dir / "forecast.csv")

    assert optimize_storage.main(["--forecast-dir", str(forecast_dir)]) == 2

    assert "DatetimeIndex" in capsys.readouterr().err
