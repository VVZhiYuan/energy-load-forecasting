import numpy as np
import pandas as pd
import pytest

from src import ml_models
from src.ml_models import (
    DirectLightGBMForecaster,
    aggregate_gain_importance,
    LightGBMCandidate,
    fit_direct_quantile_lightgbm,
    fit_direct_lightgbm,
    refit_direct_lightgbm,
    select_lightgbm_candidate,
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


def test_quantile_lightgbm_returns_requested_forecasters():
    X_train, y_train, X_val, y_val = make_frames(4)
    fitted = fit_direct_quantile_lightgbm(
        X_train,
        y_train,
        X_val,
        y_val,
        candidate=SMOKE_CANDIDATE,
        quantiles=(0.1, 0.5, 0.9),
        parallel_jobs=1,
    )

    assert tuple(fitted) == (0.1, 0.5, 0.9)
    assert all(model.predict(X_val).shape == (len(X_val), 4) for model in fitted.values())
    assert fitted[0.1].objective == "quantile"
    assert fitted[0.1].alpha == 0.1


def test_refit_uses_every_supplied_origin():
    X_train, y_train, X_val, y_val = make_frames(4)
    fitted = fit_direct_lightgbm(
        X_train, y_train, X_val, y_val, SMOKE_CANDIDATE, parallel_jobs=1
    )
    X_all = pd.concat([X_train, X_val])
    y_all = pd.concat([y_train, y_val])

    refitted = refit_direct_lightgbm(X_all, y_all, fitted, parallel_jobs=1)

    assert refitted.predict(X_all.iloc[[-1]]).shape == (1, 4)
    assert all(model.n_features_in_ == X_all.shape[1] for model in refitted.models)


@pytest.mark.parametrize("quantiles", [(0.5, 0.1), (0.0, 0.5, 0.9), (0.1, 1.0)])
def test_quantile_lightgbm_rejects_invalid_quantiles(quantiles):
    X_train, y_train, X_val, y_val = make_frames(4)

    with pytest.raises(ValueError, match="quantiles"):
        fit_direct_quantile_lightgbm(
            X_train,
            y_train,
            X_val,
            y_val,
            SMOKE_CANDIDATE,
            quantiles=quantiles,
            parallel_jobs=1,
        )


class FakeForecaster:
    def __init__(self, candidate, prediction):
        self.candidate = candidate
        self._prediction = prediction

    def predict(self, features):
        return np.full((len(features), 2), self._prediction, dtype=float)


def test_candidate_selection_uses_validation_mae(monkeypatch):
    index = pd.date_range("2024-01-01", periods=8, freq="15min")
    X = pd.DataFrame({"load": np.arange(8, dtype=float)}, index=index)
    y = pd.DataFrame(np.ones((8, 2)), index=index)
    candidates = (
        LightGBMCandidate("small", 3, 0.1, 3, 2, 0.0),
        LightGBMCandidate("medium", 5, 0.1, 3, 2, 0.0),
    )

    def fake_fit(X_train, y_train, X_val, y_val, candidate, parallel_jobs):
        prediction = 3.0 if candidate.name == "small" else 1.0
        return FakeForecaster(candidate, prediction)

    monkeypatch.setattr(ml_models, "fit_direct_lightgbm", fake_fit)
    selected, results = select_lightgbm_candidate(
        X, y, X, y, candidates=candidates, parallel_jobs=1
    )

    assert selected.candidate.name == "medium"
    assert results.loc[results["selected"], "candidate"].item() == "medium"


def test_candidate_selection_breaks_ties_by_declared_order(monkeypatch):
    index = pd.date_range("2024-01-01", periods=8, freq="15min")
    X = pd.DataFrame({"load": np.arange(8, dtype=float)}, index=index)
    y = pd.DataFrame(np.ones((8, 2)), index=index)
    candidates = (
        LightGBMCandidate("first", 3, 0.1, 3, 2, 0.0),
        LightGBMCandidate("second", 5, 0.1, 3, 2, 0.0),
    )

    def fake_fit(X_train, y_train, X_val, y_val, candidate, parallel_jobs):
        return FakeForecaster(candidate, 1.0)

    monkeypatch.setattr(ml_models, "fit_direct_lightgbm", fake_fit)
    selected, _ = select_lightgbm_candidate(
        X, y, X, y, candidates=candidates, parallel_jobs=1
    )

    assert selected.candidate.name == "first"


class FakeBooster:
    def __init__(self, gains):
        self._gains = np.asarray(gains, dtype=float)

    def feature_importance(self, importance_type):
        assert importance_type == "gain"
        return self._gains


class FakeModel:
    def __init__(self, gains):
        self.booster_ = FakeBooster(gains)


def test_gain_importance_contains_and_normalizes_every_feature():
    candidate = LightGBMCandidate("smoke", 3, 0.1, 3, 2, 0.0)
    forecaster = DirectLightGBMForecaster(
        models=(FakeModel([3.0, 1.0]), FakeModel([1.0, 1.0])),
        feature_names=("load", "hour"),
        candidate=candidate,
    )

    importance = aggregate_gain_importance(forecaster)

    assert set(importance["feature"]) == {"load", "hour"}
    assert importance["normalized_gain"].sum() == pytest.approx(1.0)
    assert importance.iloc[0]["rank"] == 1
