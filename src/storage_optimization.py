"""Configuration and input validation for storage dispatch optimization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Real

import numpy as np
import pandas as pd
from scipy.optimize import linprog


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


@dataclass(frozen=True)
class DispatchResult:
    """One feasible battery dispatch and its operational metrics."""

    schedule: pd.DataFrame
    metrics: dict[str, float | int]
    solver: dict[str, object]


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


def _validate_dispatch_inputs(
    load: pd.Series,
    tariff_schedule: pd.DataFrame,
    battery: BatteryConfig,
    tariff: TariffConfig,
) -> tuple[pd.Series, pd.DataFrame]:
    battery.validate()
    tariff.validate()
    if not isinstance(load, pd.Series) or not isinstance(load.index, pd.DatetimeIndex):
        raise ValueError("load must be a timestamp-indexed pandas Series")
    if load.empty or load.index.hasnans or not load.index.is_unique:
        raise ValueError("load must have non-empty unique timestamps")
    numeric_load = pd.to_numeric(load, errors="coerce").astype(float)
    if not np.isfinite(numeric_load.to_numpy()).all() or (numeric_load < 0).any():
        raise ValueError("load must contain finite non-negative values")
    if not isinstance(tariff_schedule, pd.DataFrame):
        raise ValueError("tariff_schedule must be a pandas DataFrame")
    required = ("tariff_period", "energy_price")
    missing = [column for column in required if column not in tariff_schedule]
    if missing:
        raise ValueError("tariff_schedule is missing: " + ", ".join(missing))
    if not tariff_schedule.index.equals(numeric_load.index):
        raise ValueError("tariff_schedule must be aligned with load timestamps")
    schedule = tariff_schedule.loc[:, required].copy()
    schedule["energy_price"] = pd.to_numeric(
        schedule["energy_price"], errors="coerce"
    ).astype(float)
    if (
        schedule["tariff_period"].isna().any()
        or not np.isfinite(schedule["energy_price"].to_numpy()).all()
        or (schedule["energy_price"] < 0).any()
    ):
        raise ValueError("tariff_schedule contains invalid tariff values")
    numeric_load.name = "forecast_load_kw"
    return numeric_load, schedule


def summarize_dispatch(
    schedule: pd.DataFrame,
    battery: BatteryConfig,
    tariff: TariffConfig,
) -> dict[str, float | int]:
    """Calculate comparable energy, peak, throughput, and SOC metrics."""

    battery.validate()
    tariff.validate()
    required = (
        "grid_import_kw",
        "charge_kw",
        "discharge_kw",
        "soc",
        "energy_price",
    )
    missing = [column for column in required if column not in schedule]
    if missing:
        raise ValueError("dispatch schedule is missing: " + ", ".join(missing))
    values = schedule.loc[:, required].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("dispatch schedule must contain finite values")
    if (values.loc[:, ("grid_import_kw", "charge_kw", "discharge_kw")] < 0).any().any():
        raise ValueError("dispatch powers must be non-negative")

    interval_hours = battery.interval_hours
    energy_cost = float(
        (
            values["grid_import_kw"]
            * values["energy_price"]
            * interval_hours
        ).sum()
    )
    charge_energy = float((values["charge_kw"] * interval_hours).sum())
    discharge_energy = float((values["discharge_kw"] * interval_hours).sum())
    throughput = charge_energy + discharge_energy
    peak_import = float(values["grid_import_kw"].max())
    simultaneous = (
        (values["charge_kw"] > 1e-9) & (values["discharge_kw"] > 1e-9)
    ).sum()
    return {
        "total_energy_cost": energy_cost,
        "peak_import_kw": peak_import,
        "objective_value": float(
            energy_cost
            + tariff.peak_import_penalty * peak_import
            + tariff.throughput_cost * throughput
        ),
        "battery_charge_kwh": charge_energy,
        "battery_discharge_kwh": discharge_energy,
        "battery_throughput_kwh": throughput,
        "min_soc": float(values["soc"].min()),
        "max_soc": float(values["soc"].max()),
        "terminal_soc": float(values["soc"].iloc[-1]),
        "simultaneous_activity_count": int(simultaneous),
    }


def _build_dispatch_result(
    load: pd.Series,
    tariff_schedule: pd.DataFrame,
    charge_kw: np.ndarray,
    discharge_kw: np.ndarray,
    battery_energy_kwh: np.ndarray,
    battery: BatteryConfig,
    tariff: TariffConfig,
    method: str,
) -> DispatchResult:
    grid_import = load.to_numpy(dtype=float) + charge_kw - discharge_kw
    if (grid_import < -1e-8).any():
        raise ValueError("dispatch would require negative grid import")
    schedule = pd.DataFrame(
        {
            "forecast_load_kw": load.to_numpy(dtype=float),
            "grid_import_kw": np.maximum(grid_import, 0.0),
            "charge_kw": charge_kw,
            "discharge_kw": discharge_kw,
            "battery_energy_kwh": battery_energy_kwh,
            "soc": battery_energy_kwh / battery.capacity_kwh,
            "tariff_period": tariff_schedule["tariff_period"].to_numpy(),
            "energy_price": tariff_schedule["energy_price"].to_numpy(dtype=float),
        },
        index=load.index,
    )
    schedule.index.name = load.index.name or "forecast_timestamp"
    schedule["interval_energy_cost"] = (
        schedule["grid_import_kw"]
        * schedule["energy_price"]
        * battery.interval_hours
    )
    metrics = summarize_dispatch(schedule, battery, tariff)
    return DispatchResult(
        schedule=schedule,
        metrics=metrics,
        solver={"method": method, "success": True},
    )


def no_storage_dispatch(
    load: pd.Series,
    tariff_schedule: pd.DataFrame,
    battery: BatteryConfig,
    tariff: TariffConfig,
) -> DispatchResult:
    """Create the no-storage cost and peak baseline."""

    numeric_load, schedule = _validate_dispatch_inputs(
        load, tariff_schedule, battery, tariff
    )
    zeros = np.zeros(len(numeric_load), dtype=float)
    energy = np.full(len(numeric_load), battery.initial_energy_kwh, dtype=float)
    return _build_dispatch_result(
        numeric_load,
        schedule,
        zeros,
        zeros,
        energy,
        battery,
        tariff,
        method="no_storage",
    )


def rule_based_dispatch(
    load: pd.Series,
    tariff_schedule: pd.DataFrame,
    battery: BatteryConfig,
    tariff: TariffConfig,
) -> DispatchResult:
    """Charge off-peak and discharge at peak while preserving terminal reachability."""

    numeric_load, schedule = _validate_dispatch_inputs(
        load, tariff_schedule, battery, tariff
    )
    count = len(numeric_load)
    loads = numeric_load.to_numpy(dtype=float)
    periods = schedule["tariff_period"].to_numpy(dtype=object)
    charges = np.zeros(count, dtype=float)
    discharges = np.zeros(count, dtype=float)
    energy = np.zeros(count, dtype=float)
    current_energy = battery.initial_energy_kwh
    min_energy = battery.capacity_kwh * battery.min_soc
    max_energy = battery.capacity_kwh * battery.max_soc
    target_energy = battery.terminal_energy_kwh
    charge_increment = (
        battery.charge_efficiency
        * battery.max_charge_kw
        * battery.interval_hours
    )

    for position in range(count):
        remaining_loads = loads[position + 1 :]
        remaining_charge_capacity = (count - position - 1) * charge_increment
        remaining_discharge_capacity = float(
            np.minimum(remaining_loads, battery.max_discharge_kw).sum()
            * battery.interval_hours
            / battery.discharge_efficiency
        )
        reachable_lower = max(
            min_energy, target_energy - remaining_charge_capacity
        )
        reachable_upper = min(
            max_energy, target_energy + remaining_discharge_capacity
        )
        max_charge_energy = min(
            max_energy - current_energy,
            charge_increment,
        )
        max_discharge_energy = min(
            current_energy - min_energy,
            battery.max_discharge_kw
            * battery.interval_hours
            / battery.discharge_efficiency,
            loads[position]
            * battery.interval_hours
            / battery.discharge_efficiency,
        )
        action_lower = current_energy - max_discharge_energy
        action_upper = current_energy + max_charge_energy
        lower = max(action_lower, reachable_lower)
        upper = min(action_upper, reachable_upper)
        if lower > upper + 1e-8:
            raise ValueError(
                "rule-based strategy cannot reach terminal SOC with this load"
            )

        if periods[position] == "off_peak":
            requested_energy = action_upper
        elif periods[position] == "peak":
            requested_energy = action_lower
        else:
            requested_energy = current_energy
        next_energy = float(np.clip(requested_energy, lower, upper))
        energy_change = next_energy - current_energy
        if energy_change >= 0.0:
            charges[position] = energy_change / (
                battery.charge_efficiency * battery.interval_hours
            )
        else:
            discharges[position] = -energy_change * battery.discharge_efficiency / (
                battery.interval_hours
            )
        energy[position] = next_energy
        current_energy = next_energy

    if not np.isclose(current_energy, target_energy, atol=1e-8):
        raise ValueError("rule-based strategy did not reach terminal SOC")
    return _build_dispatch_result(
        numeric_load,
        schedule,
        charges,
        discharges,
        energy,
        battery,
        tariff,
        method="rule_based",
    )


def optimize_dispatch(
    load: pd.Series,
    tariff_schedule: pd.DataFrame,
    battery: BatteryConfig,
    tariff: TariffConfig,
) -> DispatchResult:
    """Minimize energy, peak, and throughput costs with a linear battery model."""

    numeric_load, schedule = _validate_dispatch_inputs(
        load, tariff_schedule, battery, tariff
    )
    count = len(numeric_load)
    loads = numeric_load.to_numpy(dtype=float)
    prices = schedule["energy_price"].to_numpy(dtype=float)
    charge_start = 0
    discharge_start = count
    grid_start = 2 * count
    energy_start = 3 * count
    peak_position = 4 * count
    variable_count = peak_position + 1
    interval_hours = battery.interval_hours

    objective = np.zeros(variable_count, dtype=float)
    objective[charge_start:discharge_start] = (
        tariff.throughput_cost * interval_hours
    )
    objective[discharge_start:grid_start] = (
        tariff.throughput_cost * interval_hours
    )
    objective[grid_start:energy_start] = prices * interval_hours
    objective[peak_position] = tariff.peak_import_penalty

    equality = np.zeros((count + 1, variable_count), dtype=float)
    equality_rhs = np.zeros(count + 1, dtype=float)
    for position in range(count):
        equality[position, grid_start + position] = 1.0
        equality[position, charge_start + position] = -1.0
        equality[position, discharge_start + position] = 1.0
        equality_rhs[position] = loads[position]

    dynamics = np.zeros((count + 1, variable_count), dtype=float)
    dynamics_rhs = np.zeros(count + 1, dtype=float)
    for position in range(count):
        dynamics[position, energy_start + position] = 1.0
        dynamics[position, charge_start + position] = (
            -battery.charge_efficiency * interval_hours
        )
        dynamics[position, discharge_start + position] = (
            interval_hours / battery.discharge_efficiency
        )
        if position == 0:
            dynamics_rhs[position] = battery.initial_energy_kwh
        else:
            dynamics[position, energy_start + position - 1] = -1.0
    dynamics[count, energy_start + count - 1] = 1.0
    dynamics_rhs[count] = battery.terminal_energy_kwh

    equality = np.vstack((equality, dynamics))
    equality_rhs = np.concatenate((equality_rhs, dynamics_rhs))
    inequality = np.zeros((count, variable_count), dtype=float)
    for position in range(count):
        inequality[position, grid_start + position] = 1.0
        inequality[position, peak_position] = -1.0

    min_energy = battery.capacity_kwh * battery.min_soc
    max_energy = battery.capacity_kwh * battery.max_soc
    bounds = (
        [(0.0, battery.max_charge_kw)] * count
        + [(0.0, battery.max_discharge_kw)] * count
        + [(0.0, None)] * count
        + [(min_energy, max_energy)] * count
        + [(0.0, None)]
    )
    result = linprog(
        objective,
        A_ub=inequality,
        b_ub=np.zeros(count, dtype=float),
        A_eq=equality,
        b_eq=equality_rhs,
        bounds=bounds,
        method="highs",
    )
    if not result.success or result.x is None:
        raise ValueError(f"storage optimization failed: {result.message}")

    values = np.asarray(result.x, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("storage optimization failed: solver returned non-finite values")
    charges = np.maximum(values[charge_start:discharge_start], 0.0)
    discharges = np.maximum(values[discharge_start:grid_start], 0.0)
    energy = values[energy_start:peak_position]
    dispatch = _build_dispatch_result(
        numeric_load,
        schedule,
        charges,
        discharges,
        energy,
        battery,
        tariff,
        method="scipy_highs_linprog",
    )
    return DispatchResult(
        schedule=dispatch.schedule,
        metrics=dispatch.metrics,
        solver={
            "method": "scipy_highs_linprog",
            "success": True,
            "status": int(result.status),
            "message": str(result.message),
            "objective_value": float(result.fun),
        },
    )


def run_storage_scenarios(
    forecast: pd.DataFrame,
    battery: BatteryConfig,
    tariff: TariffConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Run every strategy against the P10, P50, and P90 forecast scenarios."""

    validated = validate_forecast_frame(forecast)
    battery.validate()
    tariff.validate()
    tariff_schedule = build_tariff_schedule(validated.index, tariff)
    dispatches: list[pd.DataFrame] = []
    result_rows: list[dict[str, object]] = []
    strategies = {
        "no_storage": no_storage_dispatch,
        "rule_based": rule_based_dispatch,
        "optimized": optimize_dispatch,
    }

    for scenario in FORECAST_QUANTILES:
        load = validated[scenario]
        baseline = no_storage_dispatch(load, tariff_schedule, battery, tariff)
        baseline_metrics = baseline.metrics
        for strategy_name, strategy in strategies.items():
            result = strategy(load, tariff_schedule, battery, tariff)
            dispatch = result.schedule.copy()
            dispatch.insert(0, "strategy", strategy_name)
            dispatch.insert(0, "scenario", scenario)
            dispatches.append(dispatch)
            metrics = dict(result.metrics)
            energy_cost = float(metrics["total_energy_cost"])
            peak_import = float(metrics["peak_import_kw"])
            baseline_cost = float(baseline_metrics["total_energy_cost"])
            baseline_peak = float(baseline_metrics["peak_import_kw"])
            result_rows.append(
                {
                    "scenario": scenario,
                    "strategy": strategy_name,
                    **metrics,
                    "cost_savings": baseline_cost - energy_cost,
                    "cost_savings_pct": (
                        (baseline_cost - energy_cost) / baseline_cost * 100.0
                        if baseline_cost > 0.0
                        else 0.0
                    ),
                    "peak_reduction_kw": baseline_peak - peak_import,
                    "peak_reduction_pct": (
                        (baseline_peak - peak_import) / baseline_peak * 100.0
                        if baseline_peak > 0.0
                        else 0.0
                    ),
                    "solver": result.solver,
                }
            )

    combined = pd.concat(dispatches).sort_index(kind="stable")
    summary: dict[str, object] = {
        "assumption_source": tariff.source,
        "primary_scenario": "p50",
        "solver_method": "scipy_highs_linprog",
        "battery_config": asdict(battery),
        "tariff_config": asdict(tariff),
        "results": result_rows,
    }
    return combined, summary
