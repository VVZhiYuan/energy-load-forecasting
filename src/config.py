"""Central project configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

RAW_DATA_FILENAME = "LD2011_2014.txt"
DEFAULT_TARGET_HORIZON_1H = 4
DEFAULT_TARGET_HORIZON_24H = 96
DEFAULT_FREQUENCY = "15min"


@dataclass(frozen=True)
class ForecastConfig:
    """Configuration for one forecasting setup."""

    target_horizon_steps: int
    forecast_label: str
    target_column: str = "load"


ONE_HOUR_CONFIG = ForecastConfig(
    target_horizon_steps=DEFAULT_TARGET_HORIZON_1H,
    forecast_label="1h",
)

TWENTY_FOUR_HOUR_CONFIG = ForecastConfig(
    target_horizon_steps=DEFAULT_TARGET_HORIZON_24H,
    forecast_label="24h",
)

