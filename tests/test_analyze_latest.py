import json

import pandas as pd

import analyze_latest


def write_report(report_dir):
    report_dir.mkdir()
    (report_dir / "summary.json").write_text(
        json.dumps(
            {
                "source_label": "fixture",
                "horizon": "1h",
                "selected_model": "LightGBM",
                "forecast_origin": "2026-08-06T12:00:00",
            }
        ),
        encoding="utf-8",
    )
    forecast = pd.DataFrame(
        {
            "forecast_timestamp": pd.date_range(
                "2026-08-06 12:15", periods=2, freq="15min"
            ),
            "step": [1, 2],
            "prediction": [100.0, 120.0],
            "p10": [95.0, 110.0],
            "p50": [100.0, 120.0],
            "p90": [105.0, 130.0],
        }
    )
    forecast.to_csv(report_dir / "forecast.csv", index=False)
    pd.DataFrame(
        {
            "model": ["LightGBM", "Ridge"],
            "validation_mae": [1.0, 2.0],
            "selected": [True, False],
        }
    ).to_csv(report_dir / "model_comparison.csv", index=False)


def test_mock_provider_writes_agent_analysis(tmp_path):
    report_dir = tmp_path / "report"
    write_report(report_dir)

    assert (
        analyze_latest.main(
            ["--report-dir", str(report_dir), "--provider", "mock"]
        )
        == 0
    )

    output = json.loads(
        (report_dir / "agent_analysis.json").read_text(encoding="utf-8")
    )
    assert output["provider"] == "mock"
    assert output["content"]["forecast_steps"] == 2


def test_disabled_provider_is_written_without_network(tmp_path, monkeypatch):
    report_dir = tmp_path / "report"
    write_report(report_dir)

    def forbidden(*args, **kwargs):
        raise AssertionError("disabled mode must not make a network request")

    monkeypatch.setattr("src.ai_provider.urlopen", forbidden)

    assert (
        analyze_latest.main(
            ["--report-dir", str(report_dir), "--provider", "disabled"]
        )
        == 0
    )
    output = json.loads(
        (report_dir / "agent_analysis.json").read_text(encoding="utf-8")
    )
    assert output["content"]["status"] == "disabled"


def test_missing_report_file_returns_two(tmp_path, capsys):
    report_dir = tmp_path / "missing"
    report_dir.mkdir()

    assert analyze_latest.main(["--report-dir", str(report_dir)]) == 2
    assert "summary.json" in capsys.readouterr().err

