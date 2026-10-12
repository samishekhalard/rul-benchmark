"""Метрики качества прогноза остаточного ресурса."""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot)


def phm_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Асимметричная метрика конкурса PHM 2008.

    Запоздалый прогноз (предсказали больше ресурса, чем есть) штрафуется
    сильнее раннего: в эксплуатации пропущенный отказ дороже лишнего
    техобслуживания. RMSE этой разницы не видит, поэтому метрику считаем
    отдельно — именно она ближе к бизнес-смыслу задачи.
    """
    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(score))


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": round(rmse(y_true, y_pred), 3),
        "mae": round(mae(y_true, y_pred), 3),
        "r2": round(r2(y_true, y_pred), 4),
        "phm_score": round(phm_score(y_true, y_pred), 1),
    }
