"""Tests for metric implementations."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from metrics import compute_all, mae, phm_score, r2, rmse


def test_perfect_prediction_rmse():
    y = np.array([10.0, 20.0, 30.0])
    assert rmse(y, y) == pytest.approx(0.0)


def test_perfect_prediction_mae():
    y = np.array([10.0, 20.0, 30.0])
    assert mae(y, y) == pytest.approx(0.0)


def test_perfect_prediction_r2():
    y = np.array([10.0, 20.0, 30.0])
    assert r2(y, y) == pytest.approx(1.0)


def test_perfect_prediction_phm():
    y = np.array([10.0, 20.0, 30.0])
    assert phm_score(y, y) == pytest.approx(0.0)


def test_phm_asymmetry_late_worse_than_early():
    """PHM score penalises late predictions (d > 0) more than early (d < 0)
    at the same absolute error magnitude.

    At |d| = 10:
      early (d = -10): exp(10/13) - 1 ≈ 1.14
      late  (d = +10): exp(10/10) - 1 = e - 1 ≈ 1.72
    """
    y_true = np.array([50.0])
    error = 10.0

    late_score = phm_score(y_true, y_true + error)    # d = +10
    early_score = phm_score(y_true, y_true - error)   # d = -10

    assert late_score > early_score, (
        f"Late prediction (d=+{error}) should score worse than early (d=-{error}): "
        f"{late_score:.4f} vs {early_score:.4f}"
    )


def test_phm_score_positive_for_nonzero_error():
    y_true = np.array([100.0, 50.0])
    y_pred = np.array([110.0, 40.0])
    assert phm_score(y_true, y_pred) > 0


def test_compute_all_keys():
    y = np.array([10.0, 20.0, 30.0])
    result = compute_all(y, y)
    assert set(result.keys()) == {"rmse", "mae", "r2", "phm_score"}
