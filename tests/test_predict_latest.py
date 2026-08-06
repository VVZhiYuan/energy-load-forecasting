from pathlib import Path

import predict_latest


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
        return object()

    monkeypatch.setattr(predict_latest, "load_forecast_series", fake_load)
    monkeypatch.setattr(predict_latest, "run_latest_forecast", fake_run)
    monkeypatch.setattr(
        predict_latest,
        "write_forecast_artifacts",
        lambda run, output: {"summary_json": Path("summary.json")},
    )
    assert predict_latest.main(["--horizon", "1h"]) == 0
    assert captured == {"meter": "MT_252", "country": "PT"}


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
