"""
Tree-based baseline models: XGBoost and Random Forest.

These operate on flattened windows (window * n_features) rather than sequences.
They follow the same data protocol (normalisation, unit-based split, test
windows) so results are directly comparable with neural models.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False


def _flatten(X: np.ndarray) -> np.ndarray:
    """(n, window, features) → (n, window * features)."""
    return X.reshape(len(X), -1)


def build_baseline(name: str, seed: int = 42):
    """Instantiate a baseline model by name ('xgboost' or 'random_forest')."""
    name = name.lower()
    if name in ("xgboost", "xgb"):
        if not _XGBOOST_AVAILABLE:
            raise ImportError("xgboost is not installed. Run: pip install xgboost")
        return XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
        )
    elif name in ("random_forest", "rf"):
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=5,
            random_state=seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown baseline '{name}'. Choose 'xgboost' or 'random_forest'.")


def train_baseline(model, X_train: np.ndarray, y_train: np.ndarray) -> None:
    """Fit a baseline model on flattened windows."""
    model.fit(_flatten(X_train), y_train)


def predict_baseline(model, X: np.ndarray) -> np.ndarray:
    """Predict RUL for flattened windows."""
    return model.predict(_flatten(X)).astype(np.float32)
