import numpy as np
import pandas as pd
import pytest

from src.storage_optimization import (
    BatteryConfig,
    DispatchResult,
    TariffConfig,
    build_tariff_schedule,
    no_storage_dispatch,
    optimize_dispatch,
    rule_based_dispatch,
    run_storage_scenarios,
    summarize_dispatch,
    validate_forecast_frame,
)
from src.config import OPTIMIZATION_DIR


def make_forecast_frame() -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=96, freq="15min")
    base = np.linspace(100.0, 195.0, num=96)
    return pd.DataFrame(
        {
            "p10": base,
            "p50": base + 10.0,
            "p90": base + 20.0,
        },
        index=index,
    )


def test_optimization_directory_is_under_reports():
    assert OPTIMIZATION_DIR.name == "optimization"
    assert OPTIMIZATION_DIR.parent.name == "reports"


def test_default_battery_configuration_is_valid():
    battery = BatteryConfig()
    battery.validate()

    assert battery.capacity_kwh == 500.0
    assert battery.initial_energy_kwh == 250.0
    assert battery.terminal_energy_kwh == 250.0
    assert battery.charge_efficiency == pytest.approx(np.sqrt(0.90))
    assert battery.discharge_efficiency == pytest.approx(np.sqrt(0.90))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("capacity_kwh", 0.0),
        ("max_charge_kw", 0.0),
        ("max_discharge_kw", -1.0),
    ],
)
def test_battery_rejects_non_positive_capacity_or_power(field, value):
    battery = BatteryConfig(**{field: value})

    with pytest.raises(ValueError, match=field):
        battery.validate()


def test_battery_rejects_invalid_soc_ordering():
    with pytest.raises(ValueError, match="min_soc"):
        BatteryConfig(min_soc=0.90, max_soc=0.90).validate()

    with pytest.raises(ValueError, match="max_soc"):
        BatteryConfig(min_soc=0.95, max_soc=0.90).validate()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("initial_soc", 0.05),
        ("initial_soc", 0.95),
        ("terminal_soc", 0.05),
        ("terminal_soc", 0.95),
    ],
)
def test_battery_rejects_initial_or_terminal_soc_outside_bounds(field, value):
    battery = BatteryConfig(**{field: value})

    with pytest.raises(ValueError, match=field):
        battery.validate()


@pytest.mark.parametrize(
    "round_trip_efficiency",
    [0.0, -0.1, 1.01, np.nan, np.inf],
)
def test_battery_rejects_efficiency_outside_open_closed_unit_interval(
    round_trip_efficiency,
):
    battery = BatteryConfig(round_trip_efficiency=round_trip_efficiency)

    with pytest.raises(ValueError, match="round_trip_efficiency"):
        battery.validate()


@pytest.mark.parametrize("interval_hours", [0.0, -0.25])
def test_battery_rejects_non_positive_interval_duration(interval_hours):
    battery = BatteryConfig(interval_hours=interval_hours)

    with pytest.raises(ValueError, match="interval_hours"):
        battery.validate()


@pytest.mark.parametrize(
    "field",
    ["capacity_kwh", "round_trip_efficiency", "initial_soc"],
)
def test_battery_rejects_string_configuration_values(field):
    battery = BatteryConfig(**{field: "not-a-number"})

    with pytest.raises(ValueError, match=field):
        battery.validate()


def test_default_tariff_configuration_is_valid():
    tariff = TariffConfig()
    tariff.validate()
    assert tariff.source == "synthetic_demo"


@pytest.mark.parametrize(
    "field",
    [
        "off_peak_price",
        "shoulder_price",
        "peak_price",
        "peak_import_penalty",
        "throughput_cost",
    ],
)
def test_tariff_rejects_negative_prices_penalties_and_throughput(field):
    tariff = TariffConfig(**{field: -0.01})

    with pytest.raises(ValueError, match=field):
        tariff.validate()


@pytest.mark.parametrize(
    "field",
    [
        "off_peak_price",
        "shoulder_price",
        "peak_price",
        "peak_import_penalty",
        "throughput_cost",
    ],
)
def test_tariff_rejects_string_configuration_values(field):
    tariff = TariffConfig(**{field: "not-a-number"})

    with pytest.raises(ValueError, match=field):
        tariff.validate()


def test_forecast_requires_a_unique_datetime_index():
    frame = make_forecast_frame()
    frame.index = frame.index.tolist()[:-1] + [frame.index[-2]]

    with pytest.raises(ValueError, match="unique"):
        validate_forecast_frame(frame)


def test_forecast_is_sorted_before_validation_and_returned_sorted():
    frame = make_forecast_frame().iloc[::-1]

    validated = validate_forecast_frame(frame)

    assert validated.index.is_monotonic_increasing
    pd.testing.assert_frame_equal(validated, make_forecast_frame())


@pytest.mark.parametrize("row_count", [95, 97])
def test_forecast_requires_exactly_96_rows(row_count):
    frame = make_forecast_frame().reindex(
        pd.date_range("2025-01-01", periods=row_count, freq="15min")
    )

    with pytest.raises(ValueError, match="96"):
        validate_forecast_frame(frame)


def test_forecast_requires_15_minute_continuity():
    frame = make_forecast_frame()
    irregular_index = frame.index.copy()
    irregular_index = irregular_index.delete(40).insert(
        40, irregular_index[39] + pd.Timedelta(minutes=20)
    )
    frame.index = irregular_index

    with pytest.raises(ValueError, match="15-minute"):
        validate_forecast_frame(frame)


@pytest.mark.parametrize("missing_column", ["p10", "p50", "p90"])
def test_forecast_requires_all_quantile_columns(missing_column):
    frame = make_forecast_frame().drop(columns=missing_column)

    with pytest.raises(ValueError, match=missing_column):
        validate_forecast_frame(frame)


def test_forecast_coerces_quantiles_to_finite_floats():
    frame = make_forecast_frame().astype(str)

    validated = validate_forecast_frame(frame)

    assert all(pd.api.types.is_float_dtype(validated[column]) for column in frame)
    pd.testing.assert_frame_equal(validated, make_forecast_frame())


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_forecast_rejects_non_finite_quantile_values(bad_value):
    frame = make_forecast_frame()
    frame.loc[frame.index[10], "p50"] = bad_value

    with pytest.raises(ValueError, match="finite"):
        validate_forecast_frame(frame)


@pytest.mark.parametrize("column", ["p10", "p50", "p90"])
def test_forecast_rejects_negative_loads(column):
    frame = make_forecast_frame()
    frame.loc[frame.index[10], column] = -0.01

    with pytest.raises(ValueError, match="negative"):
        validate_forecast_frame(frame)


@pytest.mark.parametrize(
    ("p10", "p50", "p90"),
    [
        (110.0, 100.0, 120.0),
        (100.0, 120.0, 110.0),
        (110.0, 120.0, 100.0),
    ],
)
def test_forecast_requires_non_decreasing_quantiles(p10, p50, p90):
    frame = make_forecast_frame()
    frame.loc[frame.index[10], ["p10", "p50", "p90"]] = [p10, p50, p90]

    with pytest.raises(ValueError, match="p10.*p50.*p90"):
        validate_forecast_frame(frame)


def test_tariff_assigns_reproducible_periods_and_prices():
    index = pd.date_range("2025-01-01", periods=96, freq="15min")
    schedule = build_tariff_schedule(index, TariffConfig())

    assert schedule.loc["2025-01-01 02:00", "tariff_period"] == "off_peak"
    assert schedule.loc["2025-01-01 07:00", "tariff_period"] == "shoulder"
    assert schedule.loc["2025-01-01 12:00", "tariff_period"] == "shoulder"
    assert schedule.loc["2025-01-01 17:00", "tariff_period"] == "peak"
    assert schedule.loc["2025-01-01 21:00", "tariff_period"] == "shoulder"
    assert schedule.loc["2025-01-01 23:00", "tariff_period"] == "off_peak"
    assert schedule.loc["2025-01-01 23:15", "energy_price"] == 0.60
    assert schedule.loc["2025-01-01 12:00", "energy_price"] == 1.00
    assert schedule.loc["2025-01-01 18:00", "energy_price"] == 1.50

    repeated = build_tariff_schedule(index, TariffConfig())
    pd.testing.assert_frame_equal(schedule, repeated)


def test_tariff_schedule_preserves_index():
    index = pd.date_range("2025-01-01", periods=4, freq="15min", tz="UTC")

    schedule = build_tariff_schedule(index, TariffConfig())

    pd.testing.assert_index_equal(schedule.index, index)


def test_tariff_schedule_rejects_missing_timestamps():
    index = pd.DatetimeIndex([pd.Timestamp("2025-01-01"), pd.NaT])

    with pytest.raises(ValueError, match="present"):
        build_tariff_schedule(index, TariffConfig())


def make_load_and_tariff() -> tuple[pd.Series, pd.DataFrame]:
    forecast = make_forecast_frame()
    return forecast["p50"], build_tariff_schedule(forecast.index, TariffConfig())


def test_no_storage_preserves_load_and_has_zero_battery_activity():
    load, tariff_schedule = make_load_and_tariff()

    result = no_storage_dispatch(
        load, tariff_schedule, BatteryConfig(), TariffConfig()
    )

    assert isinstance(result, DispatchResult)
    np.testing.assert_allclose(result.schedule["grid_import_kw"], load)
    assert result.schedule["charge_kw"].eq(0.0).all()
    assert result.schedule["discharge_kw"].eq(0.0).all()
    assert result.schedule["battery_energy_kwh"].eq(250.0).all()
    assert result.metrics["terminal_soc"] == pytest.approx(0.50)
    assert result.metrics["battery_throughput_kwh"] == pytest.approx(0.0)
    assert result.solver["success"] is True


def test_no_storage_cost_and_objective_are_calculated_from_grid_import():
    load, tariff_schedule = make_load_and_tariff()

    result = no_storage_dispatch(
        load, tariff_schedule, BatteryConfig(), TariffConfig()
    )

    expected_energy_cost = (
        result.schedule["grid_import_kw"]
        * result.schedule["energy_price"]
        * 0.25
    ).sum()
    expected_peak = result.schedule["grid_import_kw"].max()
    assert result.metrics["total_energy_cost"] == pytest.approx(expected_energy_cost)
    assert result.metrics["peak_import_kw"] == pytest.approx(expected_peak)
    assert result.metrics["objective_value"] == pytest.approx(
        expected_energy_cost + 5.0 * expected_peak
    )


def test_rule_strategy_is_feasible_and_returns_to_terminal_soc():
    load, tariff_schedule = make_load_and_tariff()

    result = rule_based_dispatch(
        load, tariff_schedule, BatteryConfig(), TariffConfig()
    )
    schedule = result.schedule

    np.testing.assert_allclose(
        schedule["grid_import_kw"],
        schedule["forecast_load_kw"]
        + schedule["charge_kw"]
        - schedule["discharge_kw"],
        atol=1e-9,
    )
    assert (schedule["grid_import_kw"] >= 0.0).all()
    assert (schedule["charge_kw"] <= 100.0 + 1e-9).all()
    assert (schedule["discharge_kw"] <= 100.0 + 1e-9).all()
    assert schedule["soc"].between(0.10 - 1e-9, 0.90 + 1e-9).all()
    assert result.metrics["terminal_soc"] == pytest.approx(0.50, abs=1e-9)
    assert result.metrics["simultaneous_activity_count"] == 0


def test_rule_strategy_uses_off_peak_charging_and_peak_discharging():
    load, tariff_schedule = make_load_and_tariff()

    result = rule_based_dispatch(
        load, tariff_schedule, BatteryConfig(), TariffConfig()
    )

    schedule = result.schedule
    assert schedule.loc[schedule["tariff_period"].eq("off_peak"), "charge_kw"].gt(
        0.0
    ).any()
    assert schedule.loc[schedule["tariff_period"].eq("peak"), "discharge_kw"].gt(
        0.0
    ).any()


def test_rule_strategy_stored_energy_follows_battery_dynamics():
    load, tariff_schedule = make_load_and_tariff()
    battery = BatteryConfig()

    result = rule_based_dispatch(load, tariff_schedule, battery, TariffConfig())
    schedule = result.schedule
    expected = np.empty(len(schedule))
    previous = battery.initial_energy_kwh
    for position, row in enumerate(schedule.itertuples()):
        previous += (
            battery.charge_efficiency * row.charge_kw * battery.interval_hours
            - row.discharge_kw
            * battery.interval_hours
            / battery.discharge_efficiency
        )
        expected[position] = previous

    np.testing.assert_allclose(schedule["battery_energy_kwh"], expected, atol=1e-9)


def test_rule_strategy_objective_includes_throughput_cost():
    load, tariff_schedule = make_load_and_tariff()
    tariff = TariffConfig(throughput_cost=0.50)

    result = rule_based_dispatch(load, tariff_schedule, BatteryConfig(), tariff)

    assert result.metrics["battery_throughput_kwh"] > 0.0
    expected = (
        result.metrics["total_energy_cost"]
        + tariff.peak_import_penalty * result.metrics["peak_import_kw"]
        + tariff.throughput_cost * result.metrics["battery_throughput_kwh"]
    )
    assert result.metrics["objective_value"] == pytest.approx(expected)


def test_rule_strategy_rejects_unreachable_terminal_soc_without_grid_export():
    load, tariff_schedule = make_load_and_tariff()
    low_load = pd.Series(1.0, index=load.index)
    battery = BatteryConfig(initial_soc=0.90, terminal_soc=0.10)

    with pytest.raises(ValueError, match="cannot reach terminal SOC"):
        rule_based_dispatch(low_load, tariff_schedule, battery, TariffConfig())


def test_summarize_rejects_simultaneous_charge_and_discharge():
    load, tariff_schedule = make_load_and_tariff()
    result = no_storage_dispatch(
        load, tariff_schedule, BatteryConfig(), TariffConfig()
    )
    invalid = result.schedule.copy()
    invalid.loc[invalid.index[0], "charge_kw"] = 1.0
    invalid.loc[invalid.index[0], "discharge_kw"] = 1.0

    summary = summarize_dispatch(invalid, BatteryConfig(), TariffConfig())

    assert summary["simultaneous_activity_count"] == 1


def test_dispatch_rejects_misaligned_tariff_schedule():
    load, tariff_schedule = make_load_and_tariff()
    misaligned = tariff_schedule.copy()
    misaligned.index = misaligned.index + pd.Timedelta(minutes=15)

    with pytest.raises(ValueError, match="aligned"):
        no_storage_dispatch(load, misaligned, BatteryConfig(), TariffConfig())


def test_optimizer_satisfies_balance_bounds_and_terminal_energy():
    load, tariff_schedule = make_load_and_tariff()
    battery = BatteryConfig()

    result = optimize_dispatch(load, tariff_schedule, battery, TariffConfig())
    schedule = result.schedule

    np.testing.assert_allclose(
        schedule["grid_import_kw"],
        schedule["forecast_load_kw"]
        + schedule["charge_kw"]
        - schedule["discharge_kw"],
        atol=1e-7,
    )
    assert (schedule["grid_import_kw"] >= -1e-8).all()
    assert (schedule["charge_kw"] <= battery.max_charge_kw + 1e-8).all()
    assert (schedule["discharge_kw"] <= battery.max_discharge_kw + 1e-8).all()
    assert schedule["soc"].between(
        battery.min_soc - 1e-8, battery.max_soc + 1e-8
    ).all()
    assert result.metrics["terminal_soc"] == pytest.approx(
        battery.terminal_soc, abs=1e-7
    )
    assert result.solver["success"] is True


def test_optimizer_uses_milp_and_strictly_excludes_simultaneous_activity():
    load, tariff_schedule = make_load_and_tariff()
    result = optimize_dispatch(
        load,
        tariff_schedule,
        BatteryConfig(),
        TariffConfig(throughput_cost=0.0),
    )

    schedule = result.schedule
    assert result.solver["method"] == "scipy_highs_milp"
    assert result.metrics["simultaneous_activity_count"] == 0
    assert not (
        (schedule["charge_kw"] > 1e-8) & (schedule["discharge_kw"] > 1e-8)
    ).any()


def test_optimized_objective_is_not_worse_than_no_storage():
    load, tariff_schedule = make_load_and_tariff()
    battery = BatteryConfig()
    tariff = TariffConfig()

    baseline = no_storage_dispatch(load, tariff_schedule, battery, tariff)
    optimized = optimize_dispatch(load, tariff_schedule, battery, tariff)

    assert optimized.metrics["objective_value"] <= (
        baseline.metrics["objective_value"] + 1e-7
    )


def test_optimizer_is_deterministic_for_identical_inputs():
    load, tariff_schedule = make_load_and_tariff()

    first = optimize_dispatch(load, tariff_schedule, BatteryConfig(), TariffConfig())
    second = optimize_dispatch(load, tariff_schedule, BatteryConfig(), TariffConfig())

    pd.testing.assert_frame_equal(first.schedule, second.schedule)
    assert first.metrics == second.metrics
    assert first.solver["method"] == "scipy_highs_milp"


def test_runner_reports_three_scenarios_and_three_strategies():
    dispatch, summary = run_storage_scenarios(
        make_forecast_frame(), BatteryConfig(), TariffConfig()
    )

    assert set(dispatch["scenario"]) == {"p10", "p50", "p90"}
    assert set(dispatch["strategy"]) == {
        "no_storage",
        "rule_based",
        "optimized",
    }
    assert len(dispatch) == 96 * 9
    assert len(summary["results"]) == 9
    assert summary["primary_scenario"] == "p50"
    assert summary["solver_method"] == "scipy_highs_milp"
    assert all("cost_savings" in row for row in summary["results"])
    assert all("peak_reduction_kw" in row for row in summary["results"])


def test_optimizer_rejects_infeasible_terminal_energy():
    load, tariff_schedule = make_load_and_tariff()
    low_load = pd.Series(1.0, index=load.index)
    battery = BatteryConfig(initial_soc=0.90, terminal_soc=0.10)

    with pytest.raises(ValueError, match="optimization failed"):
        optimize_dispatch(low_load, tariff_schedule, battery, TariffConfig())
