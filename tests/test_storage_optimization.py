import numpy as np
import pandas as pd
import pytest

from src.storage_optimization import (
    BatteryConfig,
    TariffConfig,
    build_tariff_schedule,
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
