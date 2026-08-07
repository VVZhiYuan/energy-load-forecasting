"""Read-only Streamlit workbench for the portfolio report artifacts.

Run with: python -m streamlit run dashboard.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import REPORTS_DIR
from src.dashboard_data import (
    DashboardReportError,
    load_forecast_report,
    load_gru_metrics,
    load_model_comparison,
    load_robustness_report,
    load_storage_report,
)


METER = "MT_252"
PLOTLY_LAYOUT = {
    "template": "plotly_white",
    "margin": {"l": 42, "r": 18, "t": 88, "b": 38},
    "legend": {"orientation": "h", "y": 1.12, "x": 0, "xanchor": "left"},
}


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=True).encode("utf-8")


def _metric_value(metrics: dict[str, Any], section: str, metric: str) -> float:
    return float(metrics[section][metric])


def _load_gru_forecast(horizon: str) -> pd.DataFrame:
    """Load the committed GRU trajectory without invoking a model runtime."""

    path = Path(REPORTS_DIR) / "deep_learning" / METER / horizon / "forecast.csv"
    try:
        frame = pd.read_csv(path)
        required = {"forecast_timestamp", "prediction", "p10", "p50", "p90"}
        missing = required.difference(frame.columns)
        if frame.empty or missing:
            raise ValueError(f"missing required columns: {', '.join(sorted(missing))}")
        frame["forecast_timestamp"] = pd.to_datetime(frame["forecast_timestamp"], errors="raise")
        return frame.set_index("forecast_timestamp").sort_index()
    except (OSError, TypeError, ValueError) as exc:
        raise DashboardReportError(f"Unable to load dashboard artifact {path.name}: {exc}") from exc


def _forecast_figure(forecast: pd.DataFrame, title: str) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast["p90"],
            mode="lines",
            line={"width": 0, "color": "rgba(43, 108, 176, 0.18)"},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast["p10"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(43, 108, 176, 0.16)",
            line={"width": 0, "color": "rgba(43, 108, 176, 0.18)"},
            name="P10-P90 interval",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast["p50"],
            mode="lines+markers",
            line={"color": "#1f5d8c", "width": 3},
            marker={"size": 6},
            name="P50 forecast",
        )
    )
    figure.update_layout(
        title={"text": title, "x": 0, "xanchor": "left", "y": 0.99, "yanchor": "top"},
        yaxis_title="Load (kW)",
        **PLOTLY_LAYOUT,
    )
    return figure


def _show_forecast(horizon: str, family: str) -> None:
    try:
        if family == "GRU":
            forecast = _load_gru_forecast(horizon)
            metadata = load_gru_metrics(METER, horizon)
            test_mae = _metric_value(metadata, "test_metrics", "MAE")
            test_rmse = _metric_value(metadata, "test_metrics", "RMSE")
            model_label = "GRU"
        else:
            forecast, metadata = load_forecast_report(METER, horizon)
            comparison = load_model_comparison(METER, horizon)
            selected = comparison.loc[comparison["selected"].astype(bool)]
            best = selected.iloc[0] if not selected.empty else comparison.iloc[0]
            test_mae = float(best["test_mae"])
            test_rmse = float(best["test_rmse"])
            model_label = str(metadata.get("selected_model", "Classical model"))
    except DashboardReportError as exc:
        st.warning(f"Forecast report unavailable: {exc}")
        return

    columns = st.columns(4)
    columns[0].metric("Model", model_label)
    columns[1].metric("Horizon", horizon)
    columns[2].metric("Test MAE", f"{test_mae:.2f} kW")
    columns[3].metric("Test RMSE", f"{test_rmse:.2f} kW")
    st.plotly_chart(
        _forecast_figure(forecast, f"{model_label} forecast for {METER} ({horizon})"),
        use_container_width=True,
    )
    st.download_button(
        "Download forecast CSV",
        data=_csv_bytes(forecast),
        file_name=f"{METER}_{family.lower()}_{horizon}_forecast.csv",
        mime="text/csv",
    )
    with st.expander("Report metadata"):
        st.json(metadata)


def _show_model_comparison(horizon: str) -> None:
    try:
        comparison = load_model_comparison(METER, horizon)
        gru_metrics = load_gru_metrics(METER, horizon)
    except DashboardReportError as exc:
        st.warning(f"Model comparison report unavailable: {exc}")
        return

    gru_row = pd.DataFrame(
        [
            {
                "model": "GRU",
                "configuration": "context=96, hidden=64",
                "validation_mae": _metric_value(gru_metrics, "validation_metrics", "MAE"),
                "validation_rmse": _metric_value(gru_metrics, "validation_metrics", "RMSE"),
                "test_mae": _metric_value(gru_metrics, "test_metrics", "MAE"),
                "test_rmse": _metric_value(gru_metrics, "test_metrics", "RMSE"),
                "selected": False,
                "training_seconds": float(gru_metrics.get("runtime_seconds", 0.0)),
            }
        ]
    )
    comparison = pd.concat([comparison, gru_row], ignore_index=True)
    metrics = comparison.melt(
        id_vars="model", value_vars=["test_mae", "test_rmse"], var_name="metric", value_name="kW"
    )
    chart = px.bar(
        metrics,
        x="model",
        y="kW",
        color="metric",
        barmode="group",
        color_discrete_map={"test_mae": "#1f5d8c", "test_rmse": "#e46f45"},
        title=f"Test error comparison ({horizon})",
    )
    chart.update_layout(yaxis_title="Error (kW)", **PLOTLY_LAYOUT)
    st.plotly_chart(chart, use_container_width=True)
    st.dataframe(
        comparison.sort_values("test_mae"),
        column_config={
            "validation_mae": st.column_config.NumberColumn("Validation MAE", format="%.2f"),
            "validation_rmse": st.column_config.NumberColumn("Validation RMSE", format="%.2f"),
            "test_mae": st.column_config.NumberColumn("Test MAE", format="%.2f"),
            "test_rmse": st.column_config.NumberColumn("Test RMSE", format="%.2f"),
        },
        hide_index=True,
        use_container_width=True,
    )
    st.download_button(
        "Download model comparison CSV",
        data=comparison.to_csv(index=False).encode("utf-8"),
        file_name=f"{METER}_{horizon}_model_comparison.csv",
        mime="text/csv",
    )


def _show_robustness(horizon: str) -> None:
    try:
        metrics, summary = load_robustness_report(METER, horizon)
    except DashboardReportError as exc:
        st.warning(f"Robustness report unavailable: {exc}")
        return

    ranked = metrics.sort_values("mae_degradation_pct", ascending=False)
    chart = px.bar(
        ranked,
        x="mae_degradation_pct",
        y="scenario",
        orientation="h",
        color="mae_degradation_pct",
        color_continuous_scale=["#2a9d8f", "#f4a261", "#d1495b"],
        title=f"MAE degradation under data-quality scenarios ({horizon})",
    )
    chart.update_layout(xaxis_title="MAE degradation (%)", yaxis_title="Scenario", **PLOTLY_LAYOUT)
    st.plotly_chart(chart, use_container_width=True)
    st.dataframe(
        ranked[["scenario", "affected_points", "imputed_points", "mae", "rmse", "mae_delta", "mae_degradation_pct"]],
        column_config={
            "mae": st.column_config.NumberColumn("MAE (kW)", format="%.2f"),
            "rmse": st.column_config.NumberColumn("RMSE (kW)", format="%.2f"),
            "mae_delta": st.column_config.NumberColumn("MAE delta (kW)", format="%.2f"),
            "mae_degradation_pct": st.column_config.NumberColumn("Degradation (%)", format="%.1f"),
        },
        hide_index=True,
        use_container_width=True,
    )
    st.caption(f"Selected model: {summary.get('selected_models', {}).get('clean', 'unavailable')}")


def _storage_summary(summary: dict[str, Any], scenario: str) -> dict[str, Any] | None:
    for result in summary.get("results", []):
        if result.get("scenario") == scenario and result.get("strategy") == "optimized":
            return result
    return None


def _show_storage(horizon: str, scenario: str) -> None:
    try:
        dispatch, summary = load_storage_report(METER, horizon, scenario=scenario)
    except DashboardReportError as exc:
        st.warning(f"Storage report unavailable: {exc}")
        return

    st.info("Synthetic-demo tariff and battery assumptions are used in this storage simulation.")
    result = _storage_summary(summary, scenario)
    if result:
        columns = st.columns(4)
        columns[0].metric("Energy cost", f"{float(result['total_energy_cost']):.0f}")
        columns[1].metric("Cost savings", f"{float(result['cost_savings']):.0f}")
        columns[2].metric("Peak reduction", f"{float(result['peak_reduction_kw']):.1f} kW")
        columns[3].metric("Battery throughput", f"{float(result['battery_throughput_kwh']):.1f} kWh")

    grid_chart = go.Figure()
    grid_chart.add_trace(go.Scatter(x=dispatch.index, y=dispatch["forecast_load_kw"], name="Forecast load", line={"color": "#343a40"}))
    grid_chart.add_trace(go.Scatter(x=dispatch.index, y=dispatch["grid_import_kw"], name="Grid import", line={"color": "#1f5d8c", "width": 3}))
    grid_chart.update_layout(title="Forecast load and optimized grid import", yaxis_title="Power (kW)", **PLOTLY_LAYOUT)
    st.plotly_chart(grid_chart, use_container_width=True)

    battery_chart = go.Figure()
    battery_chart.add_trace(go.Bar(x=dispatch.index, y=dispatch["charge_kw"], name="Charge", marker_color="#2a9d8f"))
    battery_chart.add_trace(go.Bar(x=dispatch.index, y=-dispatch["discharge_kw"], name="Discharge", marker_color="#e46f45"))
    battery_chart.update_layout(title="Battery dispatch", barmode="relative", yaxis_title="Power (kW)", **PLOTLY_LAYOUT)
    st.plotly_chart(battery_chart, use_container_width=True)

    state_chart = go.Figure()
    state_chart.add_trace(go.Scatter(x=dispatch.index, y=dispatch["soc"], name="State of charge", line={"color": "#5e548e", "width": 3}))
    state_chart.add_trace(go.Scatter(x=dispatch.index, y=dispatch["energy_price"], name="Energy price", yaxis="y2", line={"color": "#e9c46a", "width": 2}))
    state_chart.update_layout(
        title="Battery state of charge and tariff",
        yaxis={"title": "SOC"},
        yaxis2={"title": "Price", "overlaying": "y", "side": "right"},
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(state_chart, use_container_width=True)
    st.download_button(
        "Download optimized dispatch CSV",
        data=_csv_bytes(dispatch),
        file_name=f"{METER}_{horizon}_{scenario}_optimized_dispatch.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(page_title="Energy Operations Workbench", page_icon="bar_chart", layout="wide")
    st.title("Energy Operations Workbench")
    st.caption("Read-only portfolio dashboard for forecast, robustness, and forecast-driven storage evidence.")

    with st.sidebar:
        st.header("Report controls")
        st.selectbox("Meter", [METER], disabled=True)
        horizon = st.selectbox("Forecast horizon", ["1h", "24h"], index=1)
        family = st.segmented_control("Forecast family", ["Classical", "GRU"], default="Classical")
        if family is None:
            family = "Classical"
        scenario = st.selectbox("Storage scenario", ["p10", "p50", "p90"], index=1)
        st.caption("All views read committed reports only. No model training runs in this app.")

    forecast_tab, comparison_tab, robustness_tab, storage_tab = st.tabs(
        ["Forecast", "Model Comparison", "Robustness", "Storage"]
    )
    with forecast_tab:
        _show_forecast(horizon, family)
    with comparison_tab:
        _show_model_comparison(horizon)
    with robustness_tab:
        _show_robustness(horizon)
    with storage_tab:
        _show_storage(horizon, scenario)


if __name__ == "__main__":
    main()
