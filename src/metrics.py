"""
Evaluation metrics for RUL prediction.

All functions accept array-like y_true and y_pred (1-D, same length).
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred))


def phm_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """PHM 2008 Challenge asymmetric scoring function.

    d = y_pred - y_true (positive = late prediction, negative = early).

    Penalty:
      d < 0  (early / optimistic): exp(-d / 13) - 1   — lighter
      d >= 0 (late  / pessimistic): exp(d / 10) - 1   — heavier

    WHY asymmetric: a late prediction means the system is closer to failure
    than the model thinks — a missed failure is far more costly than an
    unnecessary maintenance stop.  RMSE cannot capture this asymmetry because
    it squares both directions equally.  The PHM score penalises late
    predictions ~30 % more steeply than early ones of the same magnitude.
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    scores = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(scores.sum())


def compute_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "phm_score": phm_score(y_true, y_pred),
    }
