import numpy as np
import pandas as pd

from src.robustness import apply_scenario, scenario_catalog


def make_series(length=240):
    index = pd.date_range("2024-01-01", periods=length, freq="15min")
    values = 100.0 + 15.0 * np.sin(np.arange(length) * 2 * np.pi / 96)
    return pd.Series(values, index=index, name="load")


def test_scenario_catalog_has_clean_and_four_stress_cases():
    assert [scenario.name for scenario in scenario_catalog()] == [
        "clean",
        "sensor_noise_5pct",
        "missing_blocks_1pct",
        "spikes_1pct",
        "distribution_shift_10pct",
    ]


def test_clean_scenario_is_an_exact_copy_and_does_not_mutate_input():
    series = make_series()
    original = series.copy(deep=True)
    scenario = next(item for item in scenario_catalog() if item.name == "clean")

    result = apply_scenario(series, scenario, seed=42)

    pd.testing.assert_series_equal(result.series, original)
    pd.testing.assert_series_equal(series, original)
    assert result.affected_points == 0
    assert result.imputed_points == 0


def test_noise_is_deterministic_and_keeps_loads_non_negative():
    scenario = next(
        item for item in scenario_catalog() if item.name == "sensor_noise_5pct"
    )
    first = apply_scenario(make_series(), scenario, seed=42)
    second = apply_scenario(make_series(), scenario, seed=42)

    pd.testing.assert_series_equal(first.series, second.series)
    assert first.affected_points == len(first.series)
    assert first.imputed_points == 0
    assert (first.series >= 0).all()


def test_missing_blocks_are_repaired_and_reported():
    scenario = next(
        item for item in scenario_catalog() if item.name == "missing_blocks_1pct"
    )

    result = apply_scenario(make_series(), scenario, seed=42)

    assert result.affected_points > 0
    assert result.imputed_points == result.affected_points
    assert result.series.notna().all()
    assert result.series.index.equals(make_series().index)


def test_spikes_and_distribution_shift_change_only_expected_values():
    series = make_series()
    spikes = next(item for item in scenario_catalog() if item.name == "spikes_1pct")
    shift = next(
        item for item in scenario_catalog() if item.name == "distribution_shift_10pct"
    )

    spike_result = apply_scenario(series, spikes, seed=42)
    shift_result = apply_scenario(series, shift, seed=42)

    assert spike_result.affected_points > 0
    assert shift_result.affected_points == int(np.ceil(len(series) * 0.2))
    assert shift_result.series.iloc[-1] == series.iloc[-1] * 1.1

