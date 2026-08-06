"""Serialize and render the artifacts produced by a latest forecast run."""

from __future__ import annotations

import html
import json
import platform
import shutil
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import numpy as np
import pandas as pd
import plotly.graph_objects as go

if TYPE_CHECKING:
    from src.inference import ForecastRun


ARTIFACT_NAMES = {
    "forecast_csv": "forecast.csv",
    "comparison_csv": "model_comparison.csv",
    "png": "forecast.png",
    "html": "forecast.html",
    "summary_json": "summary.json",
}
PUBLISH_ORDER = ("forecast_csv", "comparison_csv", "png", "html", "summary_json")
FORECAST_VALUE_COLUMNS = ("prediction", "p10", "p50", "p90")


def _package_versions() -> dict[str, str]:
    packages = ("numpy", "pandas", "matplotlib", "plotly")
    result = {"python": platform.python_version()}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "unavailable"
    return result


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _validate_forecast(run: ForecastRun) -> pd.DataFrame:
    forecast = run.forecast.copy()
    missing = [column for column in FORECAST_VALUE_COLUMNS if column not in forecast]
    if missing:
        raise ValueError(f"forecast is missing required columns: {', '.join(missing)}")
    if forecast.empty:
        raise ValueError("forecast must contain at least one row")

    values = forecast.loc[:, FORECAST_VALUE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("forecast values must be finite")
    quantiles = values[:, 1:]
    if not np.all(quantiles[:, 1:] >= quantiles[:, :-1]):
        raise ValueError("forecast quantiles must satisfy p10 <= p50 <= p90")
    if forecast.index.name is None:
        forecast.index.name = "forecast_timestamp"
    return forecast


def _render_png(
    observed: pd.Series,
    forecast: pd.DataFrame,
    summary: dict[str, object],
    destination: Path,
) -> None:
    history = observed.iloc[-min(len(observed), 7 * 96) :]
    source = str(summary.get("source_label", "Forecast"))
    horizon = str(summary.get("horizon", ""))
    title = f"{source} {horizon} Latest Load Forecast"

    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    axis.plot(history.index, history.to_numpy(dtype=float), label="Recent history")
    axis.plot(
        forecast.index,
        forecast["prediction"].to_numpy(dtype=float),
        label="Point forecast",
    )
    axis.fill_between(
        forecast.index,
        forecast["p10"].to_numpy(dtype=float),
        forecast["p90"].to_numpy(dtype=float),
        alpha=0.25,
        label="P10-P90 interval",
    )
    axis.set_xlabel("Timestamp")
    axis.set_ylabel("Load")
    axis.set_title(title)
    axis.legend()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def _render_html(
    observed: pd.Series,
    forecast: pd.DataFrame,
    comparison: pd.DataFrame,
    summary: dict[str, object],
    destination: Path,
) -> None:
    history = observed.iloc[-min(len(observed), 7 * 96) :]
    selected_model = str(summary["selected_model"])
    source = str(summary.get("source_label", "Forecast"))
    horizon = str(summary.get("horizon", ""))
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=history.index,
            y=history.to_numpy(dtype=float),
            mode="lines",
            name="Recent history",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast["prediction"].to_numpy(dtype=float),
            mode="lines+markers",
            name=f"Point forecast ({selected_model})",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast["p90"].to_numpy(dtype=float),
            mode="lines",
            line={"width": 0},
            name="P90",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast["p10"].to_numpy(dtype=float),
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(31, 119, 180, 0.2)",
            name="P10-P90 interval",
        )
    )
    figure.update_layout(
        title=f"{source} {horizon} Latest Load Forecast",
        xaxis_title="Timestamp",
        yaxis_title="Load",
        template="plotly_white",
    )
    trajectory_html = figure.to_html(
        full_html=False,
        include_plotlyjs=True,
        config={"displaylogo": False, "responsive": True},
    )
    leaderboard_html = comparison.sort_values("validation_mae").to_html(
        index=False,
        float_format=lambda value: f"{value:.4f}",
        border=0,
    )
    interval_method = html.escape(str(summary.get("interval_method", "")))
    html_document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Latest Load Forecast</title></head><body>
<main><h1>Latest Load Forecast</h1>
<p>Selected model: {html.escape(selected_model)}<br>Interval method: {interval_method}</p>
{trajectory_html}<h2>Model Comparison</h2>{leaderboard_html}</main>
</body></html>"""
    destination.write_text(html_document, encoding="utf-8")


def _validate_staged_artifacts(
    staging_dir: Path,
    forecast: pd.DataFrame,
    comparison: pd.DataFrame,
    summary: dict[str, object],
) -> None:
    forecast_readback = pd.read_csv(
        staging_dir / ARTIFACT_NAMES["forecast_csv"],
        index_col=0,
        parse_dates=[0],
    )
    comparison_readback = pd.read_csv(staging_dir / ARTIFACT_NAMES["comparison_csv"])
    if len(forecast_readback) != len(forecast):
        raise ValueError("forecast.csv did not round-trip with the expected row count")
    if list(forecast_readback.columns) != list(forecast.columns):
        raise ValueError("forecast.csv did not round-trip with the expected columns")
    expected_comparison = comparison.reset_index(drop=True)
    if len(comparison_readback) != len(expected_comparison):
        raise ValueError(
            "model_comparison.csv did not round-trip with the expected row count"
        )
    try:
        pd.testing.assert_frame_equal(
            comparison_readback,
            expected_comparison,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise ValueError(
            "model_comparison.csv did not round-trip with the expected contents"
        ) from exc
    if json.loads(
        (staging_dir / ARTIFACT_NAMES["summary_json"]).read_text(encoding="utf-8")
    ) != summary:
        raise ValueError("summary.json did not round-trip with the expected metadata")

    png = mpimg.imread(staging_dir / ARTIFACT_NAMES["png"])
    if png.shape[0] == 0 or png.shape[1] == 0:
        raise ValueError("forecast.png must have nonzero dimensions")
    html_document = (staging_dir / ARTIFACT_NAMES["html"]).read_text(encoding="utf-8")
    selected_model = str(summary["selected_model"])
    if "plotly" not in html_document.lower() or selected_model not in html_document:
        raise ValueError("forecast.html must contain Plotly and the selected model")
    if any(
        not (staging_dir / filename).is_file()
        or (staging_dir / filename).stat().st_size == 0
        for filename in ARTIFACT_NAMES.values()
    ):
        raise ValueError("all report artifacts must be non-empty before publication")


def write_forecast_artifacts(
    run: ForecastRun, output_dir: str | Path
) -> dict[str, Path]:
    """Write and publish the complete report set for one ForecastRun."""

    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir: Path | None = None
    forecast = _validate_forecast(run)
    comparison = run.model_comparison.copy()
    summary = dict(run.summary)
    summary["package_versions"] = _package_versions()
    summary["artifacts"] = dict(ARTIFACT_NAMES)

    try:
        staging_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}.staging-",
                dir=output_dir.parent,
            )
        )
        forecast.to_csv(
            staging_dir / ARTIFACT_NAMES["forecast_csv"],
            index=True,
            index_label=forecast.index.name,
        )
        comparison.to_csv(
            staging_dir / ARTIFACT_NAMES["comparison_csv"],
            index=False,
        )
        _render_png(run.observed, forecast, summary, staging_dir / ARTIFACT_NAMES["png"])
        _render_html(
            run.observed,
            forecast,
            comparison,
            summary,
            staging_dir / ARTIFACT_NAMES["html"],
        )
        (staging_dir / ARTIFACT_NAMES["summary_json"]).write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
            encoding="utf-8",
        )
        _validate_staged_artifacts(staging_dir, forecast, comparison, summary)

        paths = {key: output_dir / filename for key, filename in ARTIFACT_NAMES.items()}
        for key in PUBLISH_ORDER:
            source = staging_dir / ARTIFACT_NAMES[key]
            destination = output_dir / ARTIFACT_NAMES[key]
            source.replace(destination)
        return paths
    finally:
        if staging_dir is not None:
            shutil.rmtree(staging_dir, ignore_errors=True)
