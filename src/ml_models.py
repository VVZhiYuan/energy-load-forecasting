"""Direct per-step LightGBM models for multi-step load forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from src.evaluate import mae


@dataclass(frozen=True)
class LightGBMCandidate:
    """One bounded LightGBM configuration used during validation search."""

    name: str
    num_leaves: int
    learning_rate: float
    n_estimators: int
    min_child_samples: int
    reg_lambda: float


DEFAULT_LIGHTGBM_CANDIDATES = (
    LightGBMCandidate("small", 15, 0.05, 300, 40, 1.0),
    LightGBMCandidate("medium", 31, 0.05, 400, 20, 0.1),
    LightGBMCandidate("large", 63, 0.03, 500, 20, 0.1),
)


@dataclass(frozen=True)
class DirectLightGBMForecaster:
    """Ordered collection of one fitted LightGBM model per forecast step."""

    models: tuple[LGBMRegressor, ...]
    feature_names: tuple[str, ...]
    candidate: LightGBMCandidate

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        _validate_frame("features", features)
        if tuple(features.columns) != self.feature_names:
            raise ValueError("prediction feature columns must match training columns.")
        return np.column_stack([model.predict(features) for model in self.models])


def _validate_frame(name: str, frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{name} must be a non-empty DataFrame.")
    if not frame.index.is_unique:
        raise ValueError(f"{name} index must be unique.")
    try:
        values = frame.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values.")


def _validate_supervised_frames(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
) -> None:
    for name, frame in (
        ("X_train", X_train),
        ("y_train", y_train),
        ("X_val", X_val),
        ("y_val", y_val),
    ):
        _validate_frame(name, frame)
    if not X_train.index.equals(y_train.index) or not X_val.index.equals(y_val.index):
        raise ValueError("feature and target indexes must match within each partition.")
    if tuple(X_train.columns) != tuple(X_val.columns):
        raise ValueError("training and validation feature columns must match.")
    if tuple(y_train.columns) != tuple(y_val.columns):
        raise ValueError("training and validation target columns must match.")


def _fit_one_step(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    candidate: LightGBMCandidate,
) -> LGBMRegressor:
    model = LGBMRegressor(
        objective="regression_l1",
        random_state=42,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        n_jobs=1,
        verbosity=-1,
        num_leaves=candidate.num_leaves,
        learning_rate=candidate.learning_rate,
        n_estimators=candidate.n_estimators,
        min_child_samples=candidate.min_child_samples,
        reg_lambda=candidate.reg_lambda,
    )
    model.fit(
        X_train,
        y_train,
        eval_X=X_val,
        eval_y=y_val,
        eval_metric="l1",
        callbacks=[early_stopping(30, verbose=False), log_evaluation(0)],
    )
    return model


def fit_direct_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    candidate: LightGBMCandidate,
    parallel_jobs: int = -1,
) -> DirectLightGBMForecaster:
    """Fit one deterministic LightGBM estimator for every target step."""

    _validate_supervised_frames(X_train, y_train, X_val, y_val)
    if not isinstance(candidate, LightGBMCandidate):
        raise ValueError("candidate must be a LightGBMCandidate.")
    models = Parallel(n_jobs=parallel_jobs, prefer="threads")(
        delayed(_fit_one_step)(
            X_train,
            y_train.iloc[:, step],
            X_val,
            y_val.iloc[:, step],
            candidate,
        )
        for step in range(y_train.shape[1])
    )
    return DirectLightGBMForecaster(
        models=tuple(models),
        feature_names=tuple(X_train.columns),
        candidate=candidate,
    )


def select_lightgbm_candidate(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    candidates=DEFAULT_LIGHTGBM_CANDIDATES,
    parallel_jobs: int = -1,
) -> tuple[DirectLightGBMForecaster, pd.DataFrame]:
    """Select the declared LightGBM candidate using validation MAE only."""

    candidates = tuple(candidates)
    if not candidates:
        raise ValueError("candidates must not be empty.")

    best_forecaster = None
    best_score = np.inf
    rows = []
    for candidate in candidates:
        forecaster = fit_direct_lightgbm(
            X_train,
            y_train,
            X_val,
            y_val,
            candidate,
            parallel_jobs=parallel_jobs,
        )
        score = mae(y_val, forecaster.predict(X_val))
        rows.append(
            {
                "candidate": candidate.name,
                "num_leaves": candidate.num_leaves,
                "learning_rate": candidate.learning_rate,
                "n_estimators": candidate.n_estimators,
                "min_child_samples": candidate.min_child_samples,
                "reg_lambda": candidate.reg_lambda,
                "validation_MAE": score,
            }
        )
        if score < best_score:
            best_score = score
            best_forecaster = forecaster

    results = pd.DataFrame(rows)
    results["selected"] = results["candidate"].eq(best_forecaster.candidate.name)
    return best_forecaster, results


def aggregate_gain_importance(
    forecaster: DirectLightGBMForecaster,
) -> pd.DataFrame:
    """Aggregate and normalize gain importance across forecast-step models."""

    gains = np.vstack(
        [
            model.booster_.feature_importance(importance_type="gain")
            for model in forecaster.models
        ]
    )
    raw_gain = gains.sum(axis=0)
    total = raw_gain.sum()
    normalized = raw_gain / total if total > 0 else np.zeros_like(raw_gain)
    result = pd.DataFrame(
        {
            "feature": forecaster.feature_names,
            "raw_gain": raw_gain,
            "normalized_gain": normalized,
        }
    ).sort_values(["normalized_gain", "feature"], ascending=[False, True])
    result = result.reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    return result
