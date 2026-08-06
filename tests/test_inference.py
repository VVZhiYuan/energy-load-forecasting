import json

import numpy as np
import pandas as pd
import pytest

from src.data_loader import LoadedLoadSeries
from src.forecasting import make_multistep_targets
from src.inference import _select_row, ordered_quantiles, run_latest_forecast
from src.ml_models import LightGBMCandidate


def make_loaded(length=3000):
    index = pd.date_range("2024-01-01", periods=length, freq="15min")
    values = 100 + 10 * np.sin(np.arange(length) * 2 * np.pi / 96)
    return LoadedLoadSeries(
        series=pd.Series(values, index=index, name="load"),
        input_format="long",
        source_label="fixture",
        meter=None,
        negative_load_count=0,
    )


class FakeDirectForecaster:
    def __init__(self, horizon, candidate=None):
        self.candidate = candidate or LightGBMCandidate("small", 3, 0.1, 3, 2, 0.0)
        self.horizon = horizon

    def predict(self, features):
        return np.zeros((len(features), self.horizon), dtype=float)


@pytest.fixture
def patch_fast_lightgbm(monkeypatch):
    def fake_fit(X_train, y_train, X_val, y_val, candidate, parallel_jobs=-1):
        return FakeDirectForecaster(y_train.shape[1], candidate)

    monkeypatch.setattr("src.inference.fit_direct_lightgbm", fake_fit)


def test_ordered_quantiles_corrects_crossing():
    values = np.array([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
    corrected, count = ordered_quantiles(values)
    np.testing.assert_allclose(corrected, [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    assert count == 1


def test_latest_forecast_has_four_future_rows(patch_fast_lightgbm):
    loaded = make_loaded()
    run = run_latest_forecast(
        loaded,
        horizon_label="1h",
        holiday_country=None,
        search=False,
        parallel_jobs=1,
    )
    assert len(run.forecast) == 4
    assert run.forecast.index[0] == loaded.series.index[-1] + pd.Timedelta(minutes=15)
    assert run.forecast.index[-1] == loaded.series.index[-1] + pd.Timedelta(hours=1)
    assert {"prediction", "p10", "p50", "p90"}.issubset(run.forecast.columns)


def test_winner_is_lowest_validation_mae_not_lowest_test_mae():
    table = pd.DataFrame(
        [
            {"model": "Naive", "validation_mae": 1.0, "test_mae": 10.0},
            {"model": "LightGBM", "validation_mae": 2.0, "test_mae": 0.1},
        ]
    )
    selected = _select_row(table)
    assert selected.loc[selected["selected"], "model"].item() == "Naive"


def test_search_passes_all_declared_lightgbm_candidates(monkeypatch):
    captured = {}

    def fake_select(X_train, y_train, X_val, y_val, candidates, parallel_jobs=-1):
        candidates = tuple(candidates)
        captured["names"] = [candidate.name for candidate in candidates]
        search = pd.DataFrame(
            {
                "candidate": captured["names"],
                "validation_MAE": [3.0, 2.0, 1.0],
                "selected": [False, False, True],
            }
        )
        return FakeDirectForecaster(y_train.shape[1], candidates[-1]), search

    monkeypatch.setattr("src.inference.select_lightgbm_candidate", fake_select)
    run = run_latest_forecast(
        make_loaded(),
        horizon_label="1h",
        holiday_country=None,
        search=True,
        parallel_jobs=1,
    )
    assert captured["names"] == ["small", "medium", "large"]
    assert len(run.summary["lightgbm_search"]) == 3


@pytest.mark.parametrize("label,rows", [("1h", 4), ("24h", 96)])
def test_latest_forecast_row_count_matches_horizon(
    label, rows, patch_fast_lightgbm
):
    run = run_latest_forecast(
        make_loaded(),
        horizon_label=label,
        holiday_country=None,
        search=False,
        parallel_jobs=1,
    )
    assert len(run.forecast) == rows
    assert run.forecast["step"].tolist() == list(range(1, rows + 1))


def test_run_result_has_stable_report_contract(patch_fast_lightgbm):
    run = run_latest_forecast(
        make_loaded(),
        horizon_label="1h",
        holiday_country=None,
        search=False,
        parallel_jobs=1,
    )

    assert run.model_comparison["model"].tolist() == [
        "Naive",
        "Seasonal Naive",
        "Ridge",
        "LightGBM",
    ]
    assert run.model_comparison["selected"].sum() == 1
    assert run.summary["forecast_origin"] == run.observed.index[-1].isoformat()
    assert run.summary["split_rows"]["all_labeled"] > sum(
        run.summary["split_rows"][name]
        for name in ("train", "validation", "test")
    )
    json.dumps(run.summary)


def test_lightgbm_winner_refits_point_and_quantiles_on_all_labeled_origins(
    monkeypatch,
):
    loaded = make_loaded()
    horizon = 4
    targets = make_multistep_targets(loaded.series, horizon)
    calls = {"quantile_fit": 0, "refit_rows": []}

    class ExactValidationForecaster:
        def __init__(self, candidate, alpha=None):
            self.candidate = candidate
            self.alpha = alpha

        def predict(self, features):
            return targets.loc[features.index].to_numpy()

    class LatestForecaster:
        def __init__(self, value):
            self.value = value

        def predict(self, features):
            return np.full((len(features), horizon), self.value, dtype=float)

    def fake_fit(X_train, y_train, X_val, y_val, candidate, parallel_jobs=-1):
        return ExactValidationForecaster(candidate)

    def fake_fit_quantiles(
        X_train,
        y_train,
        X_val,
        y_val,
        candidate,
        quantiles=(0.1, 0.5, 0.9),
        parallel_jobs=-1,
    ):
        calls["quantile_fit"] += 1
        return {
            quantile: ExactValidationForecaster(candidate, alpha=quantile)
            for quantile in quantiles
        }

    def fake_refit(X_all, y_all, fitted, parallel_jobs=-1):
        assert X_all.index.equals(y_all.index)
        calls["refit_rows"].append(len(X_all))
        values = {None: 2.0, 0.1: 3.0, 0.5: 2.0, 0.9: 1.0}
        return LatestForecaster(values[fitted.alpha])

    monkeypatch.setattr("src.inference.fit_direct_lightgbm", fake_fit)
    monkeypatch.setattr(
        "src.inference.fit_direct_quantile_lightgbm", fake_fit_quantiles
    )
    monkeypatch.setattr("src.inference.refit_direct_lightgbm", fake_refit)

    run = run_latest_forecast(
        loaded,
        horizon_label="1h",
        holiday_country=None,
        search=False,
        parallel_jobs=1,
    )

    assert run.summary["selected_model"] == "LightGBM"
    assert calls["quantile_fit"] == 1
    assert calls["refit_rows"] == [run.summary["split_rows"]["all_labeled"]] * 4
    assert run.summary["quantile_correction_count"] == horizon
    np.testing.assert_allclose(
        run.forecast[["p10", "p50", "p90"]],
        np.tile([1.0, 2.0, 3.0], (horizon, 1)),
    )
