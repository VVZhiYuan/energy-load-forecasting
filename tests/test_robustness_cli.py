import json
from pathlib import Path

import pandas as pd

import robustness_analysis


def test_parser_has_seed_and_default_output_options():
    args = robustness_analysis.build_parser().parse_args(["--horizon", "1h"])

    assert args.horizon == "1h"
    assert args.seed == 42
    assert args.scenarios is None


def test_main_writes_robustness_artifacts(monkeypatch, tmp_path, capsys):
    input_path = tmp_path / "fixture.csv"
    output_dir = tmp_path / "robustness"
    captured = {}

    def fake_load(path, meter):
        captured["load"] = (path, meter)
        return type("Loaded", (), {"source_label": "fixture"})()

    def fake_run(loaded, horizon_label, holiday_country, **kwargs):
        captured["run"] = {
            "horizon": horizon_label,
            "country": holiday_country,
            "scenarios": kwargs["scenarios"],
            "seed": kwargs["seed"],
        }
        return pd.DataFrame(
            {
                "scenario": ["clean", "sensor_noise_5pct"],
                "mae": [1.0, 2.0],
                "clean_mae": [1.0, 1.0],
                "mae_delta": [0.0, 1.0],
                "mae_degradation_pct": [0.0, 100.0],
            }
        )

    monkeypatch.setattr(robustness_analysis, "load_forecast_series", fake_load)
    monkeypatch.setattr(robustness_analysis, "run_robustness_experiment", fake_run)

    assert (
        robustness_analysis.main(
            [
                "--input",
                str(input_path),
                "--meter",
                "MT_CUSTOM",
                "--horizon",
                "1h",
                "--holiday-country",
                "HK",
                "--seed",
                "7",
                "--scenarios",
                "sensor_noise_5pct,missing_blocks_1pct",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    assert captured["load"] == (input_path, "MT_CUSTOM")
    assert captured["run"] == {
        "horizon": "1h",
        "country": "HK",
        "scenarios": ["sensor_noise_5pct", "missing_blocks_1pct"],
        "seed": 7,
    }
    assert (output_dir / "robustness_metrics.csv").is_file()
    assert (output_dir / "robustness_summary.json").is_file()
    assert (output_dir / "robustness_mae.png").is_file()
    summary = json.loads(
        (output_dir / "robustness_summary.json").read_text(encoding="utf-8")
    )
    assert summary["seed"] == 7
    assert "Complete" in capsys.readouterr().out


def test_user_input_error_returns_two(monkeypatch, capsys):
    monkeypatch.setattr(
        robustness_analysis,
        "load_forecast_series",
        lambda path, meter: (_ for _ in ()).throw(ValueError("bad input")),
    )

    assert robustness_analysis.main(["--horizon", "1h"]) == 2
    assert "bad input" in capsys.readouterr().err

