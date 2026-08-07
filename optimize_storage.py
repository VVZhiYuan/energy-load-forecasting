"""Turn a saved 24-hour load forecast into an auditable battery dispatch report."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import OPTIMIZATION_DIR, PREDICTIONS_DIR
from src.storage_optimization import (
    BatteryConfig,
    TariffConfig,
    run_storage_scenarios,
)


DEFAULT_FORECAST_DIR = PREDICTIONS_DIR / "MT_252" / "24h"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize a battery dispatch from a saved 24-hour load forecast."
    )
    parser.add_argument("--forecast-dir", type=Path, default=DEFAULT_FORECAST_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--capacity-kwh", type=float, default=500.0)
    parser.add_argument("--max-charge-kw", type=float, default=100.0)
    parser.add_argument("--max-discharge-kw", type=float, default=100.0)
    parser.add_argument("--initial-soc", type=float, default=0.50)
    parser.add_argument("--min-soc", type=float, default=0.10)
    parser.add_argument("--max-soc", type=float, default=0.90)
    parser.add_argument("--terminal-soc", type=float, default=0.50)
    parser.add_argument("--round-trip-efficiency", type=float, default=0.90)
    parser.add_argument("--off-peak-price", type=float, default=0.60)
    parser.add_argument("--shoulder-price", type=float, default=1.00)
    parser.add_argument("--peak-price", type=float, default=1.50)
    parser.add_argument("--peak-import-penalty", type=float, default=5.00)
    parser.add_argument("--throughput-cost", type=float, default=0.02)
    return parser


def _read_forecast(forecast_dir: Path) -> pd.DataFrame:
    path = forecast_dir / "forecast.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Required forecast file not found: {path}")
    frame = pd.read_csv(path, index_col=0, parse_dates=[0])
    frame.index.name = "forecast_timestamp"
    return frame


def _read_source_summary(forecast_dir: Path) -> dict[str, object]:
    path = forecast_dir / "summary.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("forecast summary.json must contain a JSON object")
    horizon = payload.get("horizon")
    if horizon is not None and horizon != "24h":
        raise ValueError("storage optimization requires a 24h forecast report")
    return payload


def _battery_from_args(args: argparse.Namespace) -> BatteryConfig:
    return BatteryConfig(
        capacity_kwh=args.capacity_kwh,
        max_charge_kw=args.max_charge_kw,
        max_discharge_kw=args.max_discharge_kw,
        initial_soc=args.initial_soc,
        min_soc=args.min_soc,
        max_soc=args.max_soc,
        terminal_soc=args.terminal_soc,
        round_trip_efficiency=args.round_trip_efficiency,
    )


def _tariff_from_args(args: argparse.Namespace) -> TariffConfig:
    return TariffConfig(
        off_peak_price=args.off_peak_price,
        shoulder_price=args.shoulder_price,
        peak_price=args.peak_price,
        peak_import_penalty=args.peak_import_penalty,
        throughput_cost=args.throughput_cost,
    )


def _package_versions() -> dict[str, str]:
    packages = ("matplotlib", "numpy", "pandas", "scipy")
    versions = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _render_dispatch_chart(dispatch: pd.DataFrame, output_path: Path) -> None:
    p50 = dispatch.loc[
        (dispatch["scenario"] == "p50") & (dispatch["strategy"] == "optimized")
    ].copy()
    if len(p50) != 96:
        raise ValueError("optimized p50 dispatch must contain exactly 96 intervals")

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, layout="constrained")
    fig.suptitle("Forecast-Driven Battery Dispatch: P50 Scenario", fontsize=15)

    axes[0].plot(p50.index, p50["forecast_load_kw"], label="Forecast load", color="#1f77b4")
    axes[0].plot(p50.index, p50["grid_import_kw"], label="Grid import", color="#d62728")
    axes[0].set_ylabel("Power (kW)")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.25)

    axes[1].step(p50.index, p50["charge_kw"], where="mid", label="Charge", color="#2ca02c")
    axes[1].step(
        p50.index,
        -p50["discharge_kw"],
        where="mid",
        label="Discharge",
        color="#ff7f0e",
    )
    axes[1].axhline(0.0, color="#4d4d4d", linewidth=0.8)
    axes[1].set_ylabel("Battery power (kW)")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.25)

    axes[2].plot(p50.index, p50["soc"] * 100.0, label="State of charge", color="#9467bd")
    axes[2].set_ylabel("SOC (%)")
    axes[2].set_ylim(0.0, 100.0)
    axes[2].grid(alpha=0.25)
    price_axis = axes[2].twinx()
    price_axis.step(
        p50.index,
        p50["energy_price"],
        where="mid",
        label="Synthetic tariff",
        color="#8c564b",
        linestyle="--",
    )
    price_axis.set_ylabel("Energy price (units/kWh)")
    handles, labels = axes[2].get_legend_handles_labels()
    price_handles, price_labels = price_axis.get_legend_handles_labels()
    axes[2].legend(handles + price_handles, labels + price_labels, loc="upper left")
    axes[2].set_xlabel("Forecast timestamp")

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_artifacts(
    output_dir: Path,
    dispatch: pd.DataFrame,
    summary: dict[str, object],
) -> dict[str, Path]:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    paths = {
        "dispatch_csv": staging_dir / "dispatch.csv",
        "summary_json": staging_dir / "optimization_summary.json",
        "chart_png": staging_dir / "storage_dispatch.png",
    }
    try:
        dispatch.to_csv(paths["dispatch_csv"], index_label="forecast_timestamp")
        paths["summary_json"].write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _render_dispatch_chart(dispatch, paths["chart_png"])

        readback = pd.read_csv(paths["dispatch_csv"])
        if readback.empty or len(readback) != len(dispatch):
            raise RuntimeError("dispatch artifact failed readback validation")
        json.loads(paths["summary_json"].read_text(encoding="utf-8"))
        if paths["chart_png"].stat().st_size == 0:
            raise RuntimeError("chart artifact is empty")

        output_dir.mkdir(parents=True, exist_ok=True)
        published = {name: output_dir / path.name for name, path in paths.items()}
        for name, staged_path in paths.items():
            staged_path.replace(published[name])
        return published
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        forecast = _read_forecast(args.forecast_dir)
        source_summary = _read_source_summary(args.forecast_dir)
        source_label = str(source_summary.get("source_label", args.forecast_dir.parent.name))
        output_dir = args.output_dir or OPTIMIZATION_DIR / source_label / "24h"
        print(f"[1/3] Loading forecast report from {args.forecast_dir}")
        dispatch, summary = run_storage_scenarios(
            forecast,
            _battery_from_args(args),
            _tariff_from_args(args),
        )
        summary.update(
            {
                "artifacts": {
                    "dispatch_csv": "dispatch.csv",
                    "summary_json": "optimization_summary.json",
                    "chart_png": "storage_dispatch.png",
                },
                "source_forecast_dir": str(args.forecast_dir),
                "source_label": source_label,
                "forecast_horizon": "24h",
                "forecast_model": source_summary.get("selected_model"),
                "package_versions": _package_versions(),
            }
        )
        print("[2/3] Solving no-storage, rule-based, and optimized dispatch scenarios")
        paths = _write_artifacts(output_dir, dispatch, summary)
        print(f"[3/3] Complete: {paths['summary_json']}")
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
