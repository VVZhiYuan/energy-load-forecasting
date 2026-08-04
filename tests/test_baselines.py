import numpy as np
import pandas as pd

from src.baselines import naive_forecast, seasonal_naive_forecast, select_ridge_alpha


def test_naive_forecast_repeats_current_load():
    prediction = naive_forecast(np.array([10.0, 20.0]), horizon=4)
    np.testing.assert_allclose(
        prediction,
        np.array([[10.0, 10.0, 10.0, 10.0], [20.0, 20.0, 20.0, 20.0]]),
    )


def test_seasonal_naive_uses_previous_day_matching_steps():
    index = pd.date_range("2024-01-01", periods=300, freq="15min")
    series = pd.Series(np.arange(300, dtype=float), index=index)
    origins = index[[150, 200]]

    prediction = seasonal_naive_forecast(series, origins, horizon=4)

    np.testing.assert_allclose(prediction[0], [55.0, 56.0, 57.0, 58.0])
    np.testing.assert_allclose(prediction[1], [105.0, 106.0, 107.0, 108.0])


def test_seasonal_naive_96_step_forecast_never_uses_future_data():
    index = pd.date_range("2024-01-01", periods=300, freq="15min")
    series = pd.Series(np.arange(300, dtype=float), index=index)
    origin = index[[150]]

    prediction = seasonal_naive_forecast(series, origin, horizon=96)

    assert prediction.shape == (1, 96)
    assert prediction[0, -1] == series.loc[origin[0]]


def test_ridge_selection_returns_multioutput_model_and_smallest_tied_alpha():
    X_train = pd.DataFrame({"constant": np.zeros(20)})
    y_train = pd.DataFrame(np.ones((20, 4)))
    X_val = pd.DataFrame({"constant": np.zeros(5)})
    y_val = pd.DataFrame(np.ones((5, 4)))

    model, results = select_ridge_alpha(
        X_train,
        y_train,
        X_val,
        y_val,
        alphas=(0.1, 1.0, 10.0),
    )

    assert model.named_steps["ridge"].alpha == 0.1
    assert list(results.columns) == ["alpha", "validation_MAE"]
    assert model.predict(X_val).shape == (5, 4)
