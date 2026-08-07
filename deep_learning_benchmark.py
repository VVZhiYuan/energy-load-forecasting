"""Run and publish one or both optional GRU electricity-load benchmarks."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src.config import (
    DEFAULT_TARGET_HORIZON_1H,
    DEFAULT_TARGET_HORIZON_24H,
    PREDICTIONS_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
)
from src.data_loader import get_default_raw_data_path, load_forecast_series
from src.deep_learning import GRUConfig, run_gru_benchmark
from src.deep_learning_reporting import read_saved_comparison, write_gru_artifacts


HORIZONS = {
    "1h": DEFAULT_TARGET_HORIZON_1H,
    "24h": DEFAULT_TARGET_HORIZON_24H,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for direct GRU benchmarks."""

    defaults = GRUConfig()
    parser = argparse.ArgumentParser(
        description="Benchmark a direct PyTorch GRU electricity-load forecaster."
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--meter")
    parser.add_argument("--horizon", choices=("1h", "24h", "both"), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--context-steps", type=int, default=defaults.context_steps)
    parser.add_argument("--hidden-size", type=int, default=defaults.hidden_size)
    parser.add_argument("--num-layers", type=int, default=defaults.num_layers)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    return parser


def _config_from_args(args: argparse.Namespace) -> GRUConfig:
    return GRUConfig(
        context_steps=args.context_steps,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seed=args.seed,
    )


def _horizon_labels(value: str) -> tuple[str, ...]:
    return ("1h", "24h") if value == "both" else (value,)


def _output_dir(args: argparse.Namespace, source_label: str, label: str) -> Path:
    if args.output_dir is None:
        return REPORTS_DIR / "deep_learning" / source_label / label
    return args.output_dir / label if args.horizon == "both" else args.output_dir


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    using_default = args.input is None
    input_path = args.input or get_default_raw_data_path(PROJECT_ROOT)
    meter = args.meter or ("MT_252" if using_default else None)
    try:
        config = _config_from_args(args)
        print(f"[1/3] Loading {input_path}")
        loaded = load_forecast_series(input_path, meter=meter)
        source_label = getattr(loaded, "source_label", input_path.stem)
        source_metadata = {
            "input_format": getattr(loaded, "input_format", None),
            "meter": getattr(loaded, "meter", meter),
            "negative_load_count": getattr(loaded, "negative_load_count", None),
        }
        for label in _horizon_labels(args.horizon):
            print(f"[2/3] Running {label} GRU benchmark")
            started = time.perf_counter()
            result = run_gru_benchmark(loaded.series, HORIZONS[label], config)
            runtime_seconds = time.perf_counter() - started
            output_dir = _output_dir(args, source_label, label)
            comparison = read_saved_comparison(PREDICTIONS_DIR / source_label / label)
            print(f"[3/3] Publishing {label} GRU report")
            paths = write_gru_artifacts(
                result,
                loaded.series,
                output_dir,
                source_label=source_label,
                horizon_label=label,
                runtime_seconds=runtime_seconds,
                classical_comparison=comparison,
                source_metadata=source_metadata,
            )
            print(f"Complete: {paths['metrics_json']}")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
