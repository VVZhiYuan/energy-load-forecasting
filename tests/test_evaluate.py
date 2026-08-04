import numpy as np
import pytest

from src.evaluate import evaluate_multistep, per_step_metrics


def test_multistep_summary_matches_hand_calculation():
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = np.array([[2.0, 2.0], [1.0, 5.0]])

    metrics = evaluate_multistep(y_true, y_pred)

    assert metrics["MAE"] == pytest.approx(1.0)
    assert metrics["RMSE"] == pytest.approx(np.sqrt(1.5))
    assert metrics["Endpoint_MAE"] == pytest.approx(0.5)


def test_per_step_metrics_returns_one_row_per_step():
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = np.array([[2.0, 2.0], [1.0, 5.0]])

    result = per_step_metrics(y_true, y_pred)

    assert result["forecast_step"].tolist() == [1, 2]
    assert result["lead_minutes"].tolist() == [15, 30]
    assert result["MAE"].tolist() == pytest.approx([1.5, 0.5])


def test_multistep_metrics_reject_shape_mismatch():
    with pytest.raises(ValueError, match="identical shapes"):
        evaluate_multistep(np.ones((2, 4)), np.ones((2, 3)))
