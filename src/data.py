"""
C-MAPSS data loading, preprocessing, and windowing.

All protocol decisions are implemented here and documented with WHY comments.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Protocol §1: RUL cap.
# Turbofan degradation is not sensor-observable at the start of a run — sensors
# read healthy values whether the engine has 200 or 400 cycles left.  Capping
# RUL at 125 creates a piecewise-linear target that models only the observable
# degradation regime, preventing the model from fitting random noise in the
# "healthy" portion and improving convergence.
DEFAULT_RUL_CAP = 125


def load_raw(data_dir: str | Path, subset: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Return (train_df, test_df, rul_array) for a given subset (e.g. 'FD001')."""
    data_dir = Path(data_dir)
    train_df = pd.read_csv(
        data_dir / f"train_{subset}.txt",
        sep=r"\s+",
        header=None,
        names=COLUMNS,
    )
    test_df = pd.read_csv(
        data_dir / f"test_{subset}.txt",
        sep=r"\s+",
        header=None,
        names=COLUMNS,
    )
    rul_array = pd.read_csv(
        data_dir / f"RUL_{subset}.txt",
        header=None,
    ).values.ravel().astype(float)
    return train_df, test_df, rul_array


def add_rul(df: pd.DataFrame, rul_cap: int = DEFAULT_RUL_CAP) -> pd.DataFrame:
    """Append a 'rul' column to a training DataFrame.

    Protocol §1: RUL = max_cycle_of_unit - current_cycle, clipped to rul_cap.
    The clip prevents fitting the sensor-silent healthy phase as distinct signal.
    """
    df = df.copy()
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["rul"] = (max_cycle - df["cycle"]).clip(upper=rul_cap)
    return df


def drop_zero_variance(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Remove features whose std is zero (constant sensors carry no information).

    Protocol §4: FD001 has 7 constant sensors. Including them inflates the
    feature vector with noise and can destabilise normalisation (division by 0).
    The std is measured on the provided df — caller must pass the training split.
    """
    stds = df[feature_cols].std()
    active = stds[stds > 0].index.tolist()
    return df[["unit", "cycle", "rul"] + active].copy(), active


def normalise(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    val_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, pd.Series, pd.Series]:
    """Z-score normalisation.

    Protocol §3: mean and std are computed ONLY on the training split, then
    applied to validation and test.  Computing stats on val/test would leak
    future information and inflate metrics — a subtle but common mistake.
    """
    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std().replace(0, 1)  # guard against zero-std

    def _norm(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[feature_cols] = (df[feature_cols] - mean) / std
        return out

    normed_train = _norm(train_df)
    normed_val = _norm(val_df) if val_df is not None else None
    normed_test = _norm(test_df) if test_df is not None else None
    return normed_train, normed_val, normed_test, mean, std


def split_by_units(df: pd.DataFrame, val_fraction: float = 0.2, seed: int = 42):
    """Split a training DataFrame into train / val by unit id.

    Protocol §2: split on units, NOT on windows.  Adjacent windows from the
    same engine are nearly identical (Pearson r > 0.99 with a 1-cycle shift),
    so a random window split leaks information and inflates val metrics.
    This is the most common methodological error in C-MAPSS literature.
    See scripts/leakage_experiment.sh to measure the actual magnitude.
    """
    rng = np.random.default_rng(seed)
    units = df["unit"].unique()
    n_val = max(1, int(len(units) * val_fraction))
    val_units = rng.choice(units, size=n_val, replace=False)
    train_mask = ~df["unit"].isin(val_units)
    return df[train_mask].copy(), df[~train_mask].copy()


# METHODOLOGICALLY INCORRECT SPLIT — FOR LEAKAGE DEMONSTRATION ONLY.
# Do not use in real evaluations.
def split_windows_randomly(
    X: np.ndarray,
    y: np.ndarray,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Randomly split pre-created window arrays — WRONG, leakage demo only.

    Windows are constructed from consecutive cycles (the correct step), but
    then assigned to train/val at random without regard to engine boundaries.
    Adjacent windows from the same engine differ by exactly one cycle
    (Pearson r > 0.99), so near-duplicate windows end up on both sides of
    the split.  The model validates on samples that are almost identical to
    training samples, so val RMSE is artificially low.

    This matches the error seen in the literature: authors create all sliding
    windows, shuffle the window array, and split 80/20 — or equivalently
    split the raw sequences randomly before windowing.  The result is that
    val RMSE appears far lower than the honest test RMSE.

    Crucially, each individual window still spans consecutive cycles; the
    temporal structure within a window is intact.  What breaks is the
    boundary between train and val: engines appear in both.

    DO NOT use for real evaluation.  Run scripts/leakage_experiment.sh
    to measure the gap between inflated val RMSE and honest test RMSE.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def make_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window over each unit and collect (X, y) pairs.

    X shape: (n_windows, window, n_features)
    y shape: (n_windows,)  — RUL at the last cycle of the window

    Protocol §5 note: short engines (fewer cycles than window) are handled by
    padding in make_test_windows; here we only emit windows when there are
    enough cycles.
    """
    X_list, y_list = [], []
    for _, group in df.groupby("unit", sort=True):
        feats = group[feature_cols].values  # (n_cycles, n_features)
        labels = group["rul"].values        # (n_cycles,)
        n = len(feats)
        if n < window:
            # Not enough history — skip during training (covered by test padding)
            continue
        for start in range(n - window + 1):
            X_list.append(feats[start : start + window])
            y_list.append(labels[start + window - 1])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def make_windows_with_cycles(
    df: pd.DataFrame,
    feature_cols: list[str],
    window: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Like make_windows but also returns the cycle numbers and unit IDs per window.

    Returns (X, y, cycles, unit_ids) where:
      cycles   shape: (n_windows, window) — cycle index for each position in the window
      unit_ids shape: (n_windows,)        — which engine the window came from

    Used in tests to verify that each window spans consecutive cycles (no gaps)
    and to track which units appear in train vs val after a window-level split.
    """
    X_list, y_list, c_list, u_list = [], [], [], []
    for unit_id, group in df.groupby("unit", sort=True):
        feats = group[feature_cols].values
        labels = group["rul"].values
        cycles = group["cycle"].values
        n = len(feats)
        if n < window:
            continue
        for start in range(n - window + 1):
            X_list.append(feats[start : start + window])
            y_list.append(labels[start + window - 1])
            c_list.append(cycles[start : start + window])
            u_list.append(unit_id)
    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
        np.array(c_list, dtype=np.int64),
        np.array(u_list),
    )


def make_test_windows(
    test_df: pd.DataFrame,
    feature_cols: list[str],
    rul_array: np.ndarray,
    window: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Build exactly one test window per unit — the most recent `window` cycles.

    Protocol §5: if a unit has fewer than `window` cycles, pad by repeating
    the first available cycle at the front.  This preserves the temporal order
    and avoids introducing artificial trends from zero-padding.
    """
    X_list, y_list = [], []
    for unit_id, group in test_df.groupby("unit", sort=True):
        feats = group[feature_cols].values  # (n_cycles, n_features)
        n = len(feats)
        if n >= window:
            window_feats = feats[-window:]
        else:
            # Pad by repeating the first cycle
            pad = np.repeat(feats[[0]], window - n, axis=0)
            window_feats = np.vstack([pad, feats])
        X_list.append(window_feats)

    # rul_array is indexed by unit order (0-based)
    y_list = rul_array.copy()
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def prepare_dataset(
    data_dir: str | Path,
    subset: str = "FD001",
    window: int = 30,
    rul_cap: int = DEFAULT_RUL_CAP,
    val_fraction: float = 0.2,
    seed: int = 42,
    split_by: str = "unit",
):
    """End-to-end data preparation pipeline.

    split_by controls the train/val split strategy:
      "unit"   — Protocol §2: split on engine units (correct, default).
      "window" — METHODOLOGICALLY INCORRECT: normalise all training data,
                 create all windows from consecutive cycles, then randomly
                 assign windows to train/val.  Near-duplicate adjacent
                 windows from the same engine end up on both sides, so
                 val RMSE is inflated.  Use only with
                 scripts/leakage_experiment.sh to measure the gap.

    Returns a dict with keys:
      X_train, y_train, X_val, y_val, X_test, y_test,
      feature_cols, mean, std
    """
    if split_by not in ("unit", "window"):
        raise ValueError(f"split_by must be 'unit' or 'window', got '{split_by}'")

    train_raw, test_raw, rul_array = load_raw(data_dir, subset)
    train_raw = add_rul(train_raw, rul_cap)

    # Identify sensor + op_setting columns before dropping zero-variance ones
    all_feature_cols = [c for c in COLUMNS if c not in ("unit", "cycle")]
    # add_rul added 'rul' column; feature_cols should not include it
    # drop_zero_variance needs 'rul' present so it keeps it
    train_with_rul = train_raw[["unit", "cycle", "rul"] + all_feature_cols]

    # Protocol §4: drop zero-variance features using train statistics only
    _, active_features = drop_zero_variance(train_with_rul, all_feature_cols)

    # Protocol §2 (or its violation): choose split strategy
    if split_by == "unit":
        # CORRECT path: split by engine unit, then normalise and window each subset.
        train_df, val_df = split_by_units(train_raw, val_fraction=val_fraction, seed=seed)

        # Protocol §3: normalise using train-unit stats only
        train_norm, val_norm, test_norm, mean, std = normalise(
            train_df, active_features, val_df=val_df, test_df=test_raw
        )
        # Add RUL column to normalised val (normalise doesn't touch 'rul')
        val_norm["rul"] = val_df["rul"].values

        X_train, y_train = make_windows(train_norm, active_features, window)
        X_val, y_val = make_windows(val_norm, active_features, window)

    else:
        # WRONG path — leakage demo only; see split_windows_randomly docstring.
        #
        # Step 1: normalise the FULL training set (stats from all training data).
        # Step 2: slide windows over all units — each window spans consecutive cycles.
        # Step 3: randomly assign windows to train/val, ignoring engine boundaries.
        #
        # The leakage arises in step 3: adjacent windows from the same engine
        # (offset by one cycle, nearly identical) end up on both sides of the
        # split, so val RMSE is not an honest estimate of generalisation.
        all_norm, _, test_norm, mean, std = normalise(
            train_raw, active_features, test_df=test_raw
        )
        X_all, y_all = make_windows(all_norm, active_features, window)
        X_train, y_train, X_val, y_val = split_windows_randomly(
            X_all, y_all, val_fraction=val_fraction, seed=seed
        )

    X_test, y_test = make_test_windows(test_norm, active_features, rul_array, window)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "feature_cols": active_features,
        "mean": mean,
        "std": std,
    }
