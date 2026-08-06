from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import PREDICTIONS_DIR, PROJECT_ROOT
from src.data_loader import get_default_raw_data_path, load_forecast_series
from src.inference import run_latest_forecast
from src.reporting import write_forecast_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Forecast the latest electricity load trajectory."
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--meter")
    parser.add_argument("--horizon", choices=("1h", "24h"), required=True)
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--holiday-country")
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    using_default = args.input is None
    input_path = args.input or get_default_raw_data_path(PROJECT_ROOT)
    meter = args.meter or ("MT_252" if using_default else None)
    country = args.holiday_country or ("PT" if using_default else None)
    try:
        print(f"[1/4] Loading {input_path}")
        loaded = load_forecast_series(input_path, meter=meter)
        if country is None:
            print(
                "Warning: holiday features are disabled for custom input.",
                file=sys.stderr,
            )
        print("[2/4] Building leakage-safe backtest and selecting model")
        run = run_latest_forecast(
            loaded,
            horizon_label=args.horizon,
            holiday_country=country,
            search=args.search,
            progress=print,
        )
        source_label = getattr(loaded, "source_label", input_path.stem)
        output_dir = args.output_dir or PREDICTIONS_DIR / source_label / args.horizon
        print("[3/4] Rendering forecast reports")
        paths = write_forecast_artifacts(run, output_dir)
        print(f"[4/4] Complete: {paths['summary_json']}")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
