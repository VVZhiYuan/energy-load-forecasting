import numpy as np
import pandas as pd
import pytest

from src.ml_models import (
    LightGBMCandidate,
    fit_direct_lightgbm,
)


SMOKE_CANDIDATE = LightGBMCandidate(
    name="smoke",
    num_leaves=3,
    learning_rate=0.1,
    n_estimators=3,
    min_child_samples=2,
    reg_lambda=0.0,
)


def make_frames(horizon: int):
    train_index = pd.date_range("2024-01-01", periods=48, freq="15min")
    val_index = pd.date_range("2024-01-02", periods=16, freq="15min")
    X_train = pd.DataFrame(
        {
            "load": np.linspace(1.0, 5.0, len(train_index)),
            "hour": train_index.hour.astype(float),
        },
        index=train_index,
    )
    X_val = pd.DataFrame(
        {
            "load": np.linspace(5.1, 6.0, len(val_index)),
            "hour": val_index.hour.astype(float),
        },
        index=val_index,
    )
    offsets = np.arange(1, horizon + 1, dtype=float)
    y_train = pd.DataFrame(
        X_train["load"].to_numpy()[:, None] + offsets[None, :],
        index=train_index,
        columns=[f"target_step_{step}" for step in range(1, horizon + 1)],
    )
    y_val = pd.DataFrame(
        X_val["load"].to_numpy()[:, None] + offsets[None, :],
        index=val_index,
        columns=y_train.columns,
    )
    return X_train, y_train, X_val, y_val


@pytest.mark.parametrize("horizon", [4, 96])
def test_direct_lightgbm_returns_complete_trajectory(horizon):
    X_train, y_train, X_val, y_val = make_frames(horizon)
    forecaster = fit_direct_lightgbm(
        X_train,
        y_train,
        X_val,
        y_val,
        SMOKE_CANDIDATE,
        parallel_jobs=1,
    )

    assert forecaster.predict(X_val).shape == (len(X_val), horizon)
    assert len(forecaster.models) == horizon


def test_direct_lightgbm_rejects_misaligned_targets():
    X_train, y_train, X_val, y_val = make_frames(4)
    y_train.index = y_train.index.shift(1, freq="15min")

    with pytest.raises(ValueError, match="indexes"):
        fit_direct_lightgbm(
            X_train, y_train, X_val, y_val, SMOKE_CANDIDATE, parallel_jobs=1
        )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_direct_lightgbm_rejects_non_finite_features(bad_value):
    X_train, y_train, X_val, y_val = make_frames(4)
    X_train.iloc[0, 0] = bad_value

    with pytest.raises(ValueError, match="finite"):
        fit_direct_lightgbm(
            X_train, y_train, X_val, y_val, SMOKE_CANDIDATE, parallel_jobs=1
        )


def test_direct_lightgbm_rejects_changed_prediction_columns():
    X_train, y_train, X_val, y_val = make_frames(4)
    forecaster = fit_direct_lightgbm(
        X_train, y_train, X_val, y_val, SMOKE_CANDIDATE, parallel_jobs=1
    )

    with pytest.raises(ValueError, match="columns"):
        forecaster.predict(X_val.rename(columns={"hour": "weekday"}))


def test_direct_lightgbm_is_reproducible():
    X_train, y_train, X_val, y_val = make_frames(4)
    first = fit_direct_lightgbm(
        X_train, y_train, X_val, y_val, SMOKE_CANDIDATE, parallel_jobs=1
    )
    second = fit_direct_lightgbm(
        X_train, y_train, X_val, y_val, SMOKE_CANDIDATE, parallel_jobs=1
    )

    np.testing.assert_allclose(first.predict(X_val), second.predict(X_val))
