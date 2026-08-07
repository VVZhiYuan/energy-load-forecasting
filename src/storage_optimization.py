"""Configuration and input validation for storage dispatch optimization."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd


FORECAST_QUANTILES = ("p10", "p50", "p90")
FORECAST_STEPS = 96
FORECAST_FREQUENCY = "15min"


def _is_finite_real(value: object) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and bool(np.isfinite(value))
    )


@dataclass(frozen=True)
class BatteryConfig:
    """Battery and planning-horizon parameters."""

    capacity_kwh: float = 500.0
    max_charge_kw: float = 100.0
    max_discharge_kw: float = 100.0
    initial_soc: float = 0.50
    min_soc: float = 0.10
    max_soc: float = 0.90
    terminal_soc: float = 0.50
    round_trip_efficiency: float = 0.90
    interval_hours: float = 0.25

    @property
    def initial_energy_kwh(self) -> float:
        return self.capacity_kwh * self.initial_soc

    @property
    def terminal_energy_kwh(self) -> float:
        return self.capacity_kwh * self.terminal_soc

    @property
    def charge_efficiency(self) -> float:
        return float(np.sqrt(self.round_trip_efficiency))

    @property
    def discharge_efficiency(self) -> float:
        return float(np.sqrt(self.round_trip_efficiency))

    def validate(self) -> None:
        """Raise ValueError when the battery configuration is infeasible."""

        positive_fields = (
            "capacity_kwh",
            "max_charge_kw",
            "max_discharge_kw",
            "interval_hours",
        )
        for field in positive_fields:
            value = getattr(self, field)
            if not _is_finite_real(value) or value <= 0:
                raise ValueError(f"{field} must be finite and positive")

        if not _is_finite_real(self.round_trip_efficiency) or not (
            0 < self.round_trip_efficiency <= 1
        ):
            raise ValueError(
                "round_trip_efficiency must be finite and in (0, 1]"
            )

        for field in ("initial_soc", "terminal_soc", "min_soc", "max_soc"):
            value = getattr(self, field)
            if not _is_finite_real(value) or not 0 <= value <= 1:
                raise ValueError(f"{field} must be finite and between 0 and 1")

        if self.min_soc >= self.max_soc:
            raise ValueError("min_soc must be less than max_soc")
        if not self.min_soc <= self.initial_soc <= self.max_soc:
            raise ValueError("initial_soc must be within min_soc and max_soc")
        if not self.min_soc <= self.terminal_soc <= self.max_soc:
            raise ValueError("terminal_soc must be within min_soc and max_soc")


@dataclass(frozen=True)
class TariffConfig:
    """Three-period time-of-use tariff parameters."""

    off_peak_price: float = 0.60
    shoulder_price: float = 1.00
    peak_price: float = 1.50
    peak_import_penalty: float = 5.00
    throughput_cost: float = 0.02
    source: str = "synthetic_demo"

    def validate(self) -> None:
        """Raise ValueError when a tariff value is invalid."""

        for field in (
            "off_peak_price",
            "shoulder_price",
            "peak_price",
            "peak_import_penalty",
            "throughput_cost",
        ):
            value = getattr(self, field)
            if not _is_finite_real(value) or value < 0:
                raise ValueError(f"{field} must be finite and non-negative")


def validate_forecast_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a 24-hour quantile forecast table."""

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("forecast index must be a DatetimeIndex")
    if frame.index.hasnans:
        raise ValueError("forecast timestamps must be present")
    if not frame.index.is_unique:
        raise ValueError("forecast timestamps must be unique")
    if len(frame) != FORECAST_STEPS:
        raise ValueError(f"forecast must contain exactly {FORECAST_STEPS} rows")

    validated = frame.sort_index().copy()
    expected_index = pd.date_range(
        start=validated.index[0],
        periods=FORECAST_STEPS,
        freq=FORECAST_FREQUENCY,
    )
    if not validated.index.equals(expected_index):
        raise ValueError("forecast timestamps must have 15-minute continuity")

    missing = [column for column in FORECAST_QUANTILES if column not in validated]
    if missing:
        raise ValueError(
            "forecast is missing required columns: " + ", ".join(missing)
        )

    for column in FORECAST_QUANTILES:
        validated[column] = pd.to_numeric(
            validated[column], errors="coerce"
        ).astype(float)

    values = validated.loc[:, FORECAST_QUANTILES].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("forecast quantiles must contain only finite values")
    if (values < 0).any():
        raise ValueError("forecast loads cannot be negative")
    if not np.all(values[:, 0] <= values[:, 1]) or not np.all(
        values[:, 1] <= values[:, 2]
    ):
        raise ValueError("forecast quantiles must satisfy p10 <= p50 <= p90")

    return validated


def build_tariff_schedule(
    index: pd.DatetimeIndex, tariff: TariffConfig
) -> pd.DataFrame:
    """Build a timestamp-aligned time-of-use tariff schedule."""

    tariff.validate()
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("tariff schedule index must be a DatetimeIndex")
    if index.hasnans:
        raise ValueError("tariff schedule timestamps must be present")
    if not index.is_unique:
        raise ValueError("tariff schedule timestamps must be unique")

    hours = index.hour + index.minute / 60.0
    periods = np.select(
        [
            (hours < 7) | (hours >= 23),
            ((hours >= 7) & (hours < 17)) | ((hours >= 21) & (hours < 23)),
            (hours >= 17) & (hours < 21),
        ],
        ["off_peak", "shoulder", "peak"],
        default="off_peak",
    )
    prices = np.select(
        [periods == "off_peak", periods == "shoulder", periods == "peak"],
        [tariff.off_peak_price, tariff.shoulder_price, tariff.peak_price],
    ).astype(float)
    return pd.DataFrame(
        {"tariff_period": periods, "energy_price": prices}, index=index
    )
