import json

import numpy as np
import pandas as pd

from src.inference import ForecastRun
from src.reporting import write_forecast_artifacts


def make_run():
    observed_index = pd.date_range("2025-01-01", periods=700, freq="15min")
    forecast_index = pd.date_range(
        observed_index[-1] + pd.Timedelta(minutes=15),
        periods=4,
        freq="15min",
        name="forecast_timestamp",
    )
    forecast = pd.DataFrame(
        {
            "step": [1, 2, 3, 4],
            "prediction": [10.0, 11.0, 12.0, 13.0],
            "p10": [9.0, 10.0, 11.0, 12.0],
            "p50": [10.0, 11.0, 12.0, 13.0],
            "p90": [11.0, 12.0, 13.0, 14.0],
            "point_model": "Ridge",
            "interval_method": "residual_calibration",
        },
        index=forecast_index,
    )
    comparison = pd.DataFrame(
        [
            {
                "model": "Ridge",
                "configuration": "alpha=0.1",
                "validation_mae": 1.0,
                "validation_rmse": 1.2,
                "test_mae": 1.1,
                "test_rmse": 1.3,
                "selected": True,
                "training_seconds": 0.1,
            }
        ]
    )
    return ForecastRun(
        observed=pd.Series(np.arange(700), index=observed_index, name="load"),
        forecast=forecast,
        model_comparison=comparison,
        summary={"source_label": "fixture", "horizon": "1h", "selected_model": "Ridge"},
    )


def test_writes_complete_report_set(tmp_path):
    output = tmp_path / "report"
    paths = write_forecast_artifacts(make_run(), output)

    assert set(paths) == {"forecast_csv", "comparison_csv", "summary_json", "png", "html"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())
    assert len(pd.read_csv(paths["forecast_csv"])) == 4
    assert json.loads(paths["summary_json"].read_text(encoding="utf-8"))["selected_model"] == "Ridge"
    html = paths["html"].read_text(encoding="utf-8")
    assert "plotly" in html.lower()
    assert "Ridge" in html
