"""Run the forecast Agent against an existing report directory."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.ai_config import AISettings
from src.ai_provider import (
    SUPPORTED_PROVIDERS,
    AIProviderError,
    AgentContext,
    build_provider,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze a saved load forecast with the optional AI Agent."
    )
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--provider", choices=sorted(SUPPORTED_PROVIDERS))
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path)
    return parser


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Required report file not found: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Report file must contain a JSON object: {path.name}")
    return payload


def _read_records(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required report file not found: {path.name}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Report table must contain at least one row: {path.name}")
    # pandas' JSON encoder converts timestamps, NumPy scalars, and nulls into
    # values that can be passed through the provider contract safely.
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _load_context(report_dir: Path) -> AgentContext:
    summary = _read_json(report_dir / "summary.json")
    return AgentContext(
        summary=summary,
        forecast_rows=_read_records(report_dir / "forecast.csv"),
        comparison_rows=_read_records(report_dir / "model_comparison.csv"),
        recent_load_rows=[],
    )


def _settings_from_args(args: argparse.Namespace) -> AISettings:
    settings = AISettings.from_env()
    overrides = {}
    if args.provider is not None:
        overrides["provider"] = args.provider
    if args.base_url is not None:
        overrides["base_url"] = args.base_url
    if args.model is not None:
        overrides["model"] = args.model
    return replace(settings, **overrides)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report_dir = args.report_dir
    output_path = args.output or report_dir / "agent_analysis.json"
    try:
        context = _load_context(report_dir)
        response = build_provider(_settings_from_args(args)).analyze(context)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "provider": response.provider,
                    "model": response.model,
                    "content": response.content,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Agent analysis written to {output_path}")
        return 0
    except (OSError, ValueError, AIProviderError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

