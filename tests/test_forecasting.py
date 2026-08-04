import numpy as np
import pandas as pd
import pytest

from src.forecasting import make_multistep_targets, split_supervised_by_time


def make_series(length: int = 1000) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=length, freq="15min")
    return pd.Series(np.arange(length, dtype=float), index=index, name="load")


def test_make_multistep_targets_contains_ordered_future_values():
    series = make_series(20)
    targets = make_multistep_targets(series, horizon=4)

    assert list(targets.columns) == [
        "target_step_1",
        "target_step_2",
        "target_step_3",
        "target_step_4",
    ]
    assert targets.iloc[5].tolist() == [6.0, 7.0, 8.0, 9.0]
    assert targets.iloc[-1].isna().all()


@pytest.mark.parametrize("horizon", [0, 97])
def test_make_multistep_targets_rejects_unsupported_horizon(horizon):
    with pytest.raises(ValueError, match="between 1 and 96"):
        make_multistep_targets(make_series(), horizon=horizon)


def test_make_multistep_targets_rejects_irregular_index():
    series = make_series(20).drop(make_series(20).index[5])
    with pytest.raises(ValueError, match="15-minute"):
        make_multistep_targets(series, horizon=4)


def test_split_keeps_every_target_inside_its_partition():
    series = make_series(1000)
    targets = make_multistep_targets(series, horizon=96).dropna()
    features = pd.DataFrame({"current_load": series}, index=series.index).loc[targets.index]

    splits = split_supervised_by_time(
        features,
        targets,
        full_index=series.index,
        horizon=96,
    )

    train_end_position = int(len(series) * 0.7)
    validation_end_position = int(len(series) * 0.85)
    positions = {name: series.index.get_indexer(X.index) for name, (X, _) in splits.items()}

    assert np.all(positions["train"] + 96 < train_end_position)
    assert np.all(positions["validation"] >= train_end_position)
    assert np.all(positions["validation"] + 96 < validation_end_position)
    assert np.all(positions["test"] >= validation_end_position)
    assert np.all(positions["test"] + 96 < len(series))


def test_validation_features_can_use_pre_boundary_history():
    series = make_series(1000)
    targets = make_multistep_targets(series, horizon=4).dropna()
    features = pd.DataFrame({"lag_672": series.shift(672)}, index=series.index).dropna()
    common_index = features.index.intersection(targets.index)

    splits = split_supervised_by_time(
        features.loc[common_index],
        targets.loc[common_index],
        full_index=series.index,
        horizon=4,
    )

    X_validation, _ = splits["validation"]
    first_validation_position = series.index.get_loc(X_validation.index[0])
    assert first_validation_position == int(len(series) * 0.7)
    assert X_validation.iloc[0, 0] == series.iloc[first_validation_position - 672]
