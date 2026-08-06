"""Run historical robustness experiments for the load forecaster."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.config import PROJECT_ROOT, REPORTS_DIR
from src.data_loader import get_default_raw_data_path, load_forecast_series
from src.robustness import run_robustness_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate load forecasting robustness under data stress scenarios."
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--meter")
    parser.add_argument("--horizon", choices=("1h", "24h"), required=True)
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--holiday-country")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenarios")
    parser.add_argument("--output-dir", type=Path)
    return parser


def _scenario_names(value: str | None) -> list[str] | None:
    if value is None:
        return None
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise ValueError("--scenarios must contain at least one scenario name")
    return names


def _render_chart(metrics: pd.DataFrame, destination: Path) -> None:
    non_clean = metrics.loc[metrics["scenario"] != "clean"].copy()
    labels = {
        "sensor_noise_5pct": "Sensor noise\n(5% std)",
        "missing_blocks_1pct": "Missing blocks\n(1% of history)",
        "spikes_1pct": "Abnormal spikes\n(1% of history)",
        "distribution_shift_10pct": "Distribution shift\n(+10% recent load)",
    }
    values = non_clean["mae_degradation_pct"].to_numpy(dtype=float)
    colors = ["#c84b31" if value > 0 else "#247b7b" for value in values]
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    bars = axis.bar(
        [labels.get(name, name) for name in non_clean["scenario"]],
        values,
        color=colors,
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("MAE degradation versus clean (%)")
    axis.set_xlabel("Scenario")
    axis.set_title("Forecast robustness under data stress")
    axis.bar_label(bars, fmt="%.1f%%", padding=3)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def _write_outputs(metrics: pd.DataFrame, output_dir: Path, metadata: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "robustness_metrics.csv", index=False)
    (output_dir / "robustness_summary.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _render_chart(metrics, output_dir / "robustness_mae.png")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    using_default = args.input is None
    input_path = args.input or get_default_raw_data_path(PROJECT_ROOT)
    meter = args.meter or ("MT_252" if using_default else None)
    country = args.holiday_country or ("PT" if using_default else None)
    try:
        print(f"[1/3] Loading {input_path}")
        loaded = load_forecast_series(input_path, meter=meter)
        print("[2/3] Running historical robustness scenarios")
        names = _scenario_names(args.scenarios)
        metrics = run_robustness_experiment(
            loaded,
            horizon_label=args.horizon,
            holiday_country=country,
            scenarios=names,
            seed=args.seed,
            search=args.search,
            progress=print,
        )
        source_label = loaded.source_label
        output_dir = args.output_dir or REPORTS_DIR / "robustness" / source_label / args.horizon
        metadata = {
            "source_label": source_label,
            "input_format": getattr(loaded, "input_format", None),
            "meter": getattr(loaded, "meter", meter),
            "horizon": args.horizon,
            "holiday_country": country,
            "seed": args.seed,
            "scenarios": metrics["scenario"].tolist(),
            "selected_models": (
                dict(
                    zip(
                        metrics["scenario"],
                        metrics["selected_model"],
                        strict=True,
                    )
                )
                if "selected_model" in metrics
                else {}
            ),
            "metrics": metrics.to_dict(orient="records"),
        }
        print("[3/3] Writing robustness reports")
        _write_outputs(metrics, output_dir, metadata)
        print(f"Complete: {output_dir / 'robustness_metrics.csv'}")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
