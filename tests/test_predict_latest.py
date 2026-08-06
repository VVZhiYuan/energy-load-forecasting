from pathlib import Path

import pytest

import predict_latest
from src.config import PREDICTIONS_DIR


def test_parser_requires_supported_horizon():
    args = predict_latest.build_parser().parse_args(["--horizon", "24h"])
    assert args.horizon == "24h"
    assert args.search is False


def test_default_input_uses_uci_meter_and_portugal_holidays(monkeypatch):
    captured = {}

    def fake_load(path, meter):
        captured["meter"] = meter
        return type("Loaded", (), {"source_label": "MT_252"})()

    def fake_run(loaded, horizon_label, holiday_country, **kwargs):
        captured["country"] = holiday_country
        captured["horizon"] = horizon_label
        captured["search"] = kwargs["search"]
        return object()

    def fake_write(run, output):
        captured["output"] = output
        return {"summary_json": Path("summary.json")}

    monkeypatch.setattr(predict_latest, "load_forecast_series", fake_load)
    monkeypatch.setattr(predict_latest, "run_latest_forecast", fake_run)
    monkeypatch.setattr(predict_latest, "write_forecast_artifacts", fake_write)
    assert predict_latest.main(["--horizon", "1h"]) == 0
    assert captured == {
        "meter": "MT_252",
        "country": "PT",
        "horizon": "1h",
        "search": False,
        "output": PREDICTIONS_DIR / "MT_252" / "1h",
    }


def test_custom_input_has_no_implicit_holiday_country(monkeypatch, tmp_path):
    path = tmp_path / "custom.csv"
    path.write_text("timestamp,load\n2025-01-01,1\n", encoding="utf-8")
    captured = {}

    def fake_run(loaded, horizon_label, holiday_country, **kwargs):
        captured["country"] = holiday_country
        return object()

    monkeypatch.setattr(predict_latest, "load_forecast_series", lambda path, meter: object())
    monkeypatch.setattr(predict_latest, "run_latest_forecast", fake_run)
    monkeypatch.setattr(
        predict_latest,
        "write_forecast_artifacts",
        lambda run, output: {"summary_json": Path("summary.json")},
    )
    assert predict_latest.main(["--input", str(path), "--horizon", "1h"]) == 0
    assert captured["country"] is None


def test_user_input_error_returns_two(monkeypatch, capsys):
    monkeypatch.setattr(
        predict_latest,
        "load_forecast_series",
        lambda path, meter: (_ for _ in ()).throw(ValueError("bad timestamps")),
    )
    code = predict_latest.main(["--horizon", "1h"])
    assert code == 2
    assert "bad timestamps" in capsys.readouterr().err


def test_custom_options_forward_to_interfaces_and_progress_is_ordered(
    monkeypatch, tmp_path, capsys
):
    input_path = tmp_path / "custom.csv"
    output_dir = tmp_path / "reports"
    captured = {}

    def fake_load(path, meter):
        captured["load"] = (path, meter)
        return type("Loaded", (), {"source_label": "custom"})()

    def fake_run(loaded, horizon_label, holiday_country, search, progress):
        captured["run"] = {
            "loaded": loaded,
            "horizon": horizon_label,
            "country": holiday_country,
            "search": search,
            "progress": progress,
        }
        progress("model progress")
        return object()

    def fake_write(run, output):
        captured["write"] = (run, output)
        return {"summary_json": output / "summary.json"}

    monkeypatch.setattr(predict_latest, "load_forecast_series", fake_load)
    monkeypatch.setattr(predict_latest, "run_latest_forecast", fake_run)
    monkeypatch.setattr(predict_latest, "write_forecast_artifacts", fake_write)

    assert predict_latest.main(
        [
            "--input",
            str(input_path),
            "--meter",
            "MT_CUSTOM",
            "--horizon",
            "24h",
            "--search",
            "--holiday-country",
            "HK",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    assert captured["load"] == (input_path, "MT_CUSTOM")
    assert captured["run"]["horizon"] == "24h"
    assert captured["run"]["country"] == "HK"
    assert captured["run"]["search"] is True
    assert captured["run"]["progress"] is print
    assert captured["write"][1] == output_dir

    output = capsys.readouterr().out
    lines = output.splitlines()
    assert lines.index(f"[1/4] Loading {input_path}") < lines.index(
        "[2/4] Building leakage-safe backtest and selecting model"
    )
    assert lines.index(
        "[2/4] Building leakage-safe backtest and selecting model"
    ) < lines.index("model progress")
    assert lines.index("model progress") < lines.index("[3/4] Rendering forecast reports")
    assert lines.index("[3/4] Rendering forecast reports") < lines.index(
        f"[4/4] Complete: {output_dir / 'summary.json'}"
    )


@pytest.mark.parametrize("error_type", [FileExistsError, PermissionError, OSError])
def test_output_filesystem_error_returns_two(monkeypatch, tmp_path, capsys, error_type):
    output_dir = tmp_path / "reports"
    monkeypatch.setattr(
        predict_latest,
        "load_forecast_series",
        lambda path, meter: type("Loaded", (), {"source_label": "fixture"})(),
    )
    monkeypatch.setattr(
        predict_latest, "run_latest_forecast", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        predict_latest,
        "write_forecast_artifacts",
        lambda run, output: (_ for _ in ()).throw(error_type("cannot publish")),
    )

    assert predict_latest.main(
        ["--input", str(tmp_path / "input.csv"), "--horizon", "1h", "--output-dir", str(output_dir)]
    ) == 2
    error = capsys.readouterr().err
    assert "Error: cannot publish" in error
    assert "Traceback" not in error


def test_unexpected_programming_error_keeps_traceback(monkeypatch):
    monkeypatch.setattr(
        predict_latest,
        "load_forecast_series",
        lambda path, meter: type("Loaded", (), {"source_label": "fixture"})(),
    )
    monkeypatch.setattr(
        predict_latest,
        "run_latest_forecast",
        lambda *args, **kwargs: (_ for _ in ()).throw(TypeError("programming bug")),
    )

    with pytest.raises(TypeError, match="programming bug"):
        predict_latest.main(["--horizon", "1h"])
