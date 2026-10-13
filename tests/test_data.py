"""Tests for data loading, preprocessing, and windowing protocol."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data import (
    COLUMNS,
    DEFAULT_RUL_CAP,
    add_rul,
    drop_zero_variance,
    load_raw,
    make_windows,
    make_windows_with_cycles,
    normalise,
    prepare_dataset,
    split_by_units,
)

SUBSET = "FD001"
ALL_FEATURE_COLS = [c for c in COLUMNS if c not in ("unit", "cycle")]
WINDOW = 30


# ---------------------------------------------------------------------------
# RUL labelling
# ---------------------------------------------------------------------------

def test_rul_decreases_within_unit(cmapss_dir):
    train_df, _, _ = load_raw(cmapss_dir, SUBSET)
    train_df = add_rul(train_df)
    for _, group in train_df.groupby("unit"):
        rul = group["rul"].values
        # RUL must be non-increasing (monotone within cap, then constant at cap)
        assert np.all(np.diff(rul) <= 0), "RUL must not increase within a unit"


def test_rul_last_cycle_is_zero(cmapss_dir):
    train_df, _, _ = load_raw(cmapss_dir, SUBSET)
    train_df = add_rul(train_df)
    for _, group in train_df.groupby("unit"):
        assert group["rul"].iloc[-1] == 0, "Last cycle must have RUL = 0"


def test_rul_cap_respected(cmapss_dir):
    train_df, _, _ = load_raw(cmapss_dir, SUBSET)
    train_df = add_rul(train_df, rul_cap=DEFAULT_RUL_CAP)
    assert (train_df["rul"] <= DEFAULT_RUL_CAP).all(), "RUL must not exceed the cap"


def test_rul_cap_custom(cmapss_dir):
    train_df, _, _ = load_raw(cmapss_dir, SUBSET)
    cap = 50
    train_df = add_rul(train_df, rul_cap=cap)
    assert (train_df["rul"] <= cap).all()


# ---------------------------------------------------------------------------
# Zero-variance feature removal (Protocol §4)
# ---------------------------------------------------------------------------

def test_constant_sensors_removed(cmapss_dir):
    train_df, _, _ = load_raw(cmapss_dir, SUBSET)
    train_df = add_rul(train_df)
    # Our synthetic data has exactly 7 constant sensors
    _, active = drop_zero_variance(train_df, ALL_FEATURE_COLS)
    n_dropped = len(ALL_FEATURE_COLS) - len(active)
    assert n_dropped == 7, f"Expected 7 constant sensors dropped, got {n_dropped}"


# ---------------------------------------------------------------------------
# Train/val unit split (Protocol §2 — regression test for data leakage)
# ---------------------------------------------------------------------------

def test_no_unit_overlap_in_split(cmapss_dir):
    train_df, _, _ = load_raw(cmapss_dir, SUBSET)
    train_df = add_rul(train_df)
    tr, val = split_by_units(train_df, val_fraction=0.2, seed=42)
    train_units = set(tr["unit"].unique())
    val_units = set(val["unit"].unique())
    assert train_units.isdisjoint(val_units), (
        "Unit leakage detected: same unit appears in both train and validation"
    )


def test_val_size_approximately_correct(cmapss_dir):
    train_df, _, _ = load_raw(cmapss_dir, SUBSET)
    train_df = add_rul(train_df)
    _, val = split_by_units(train_df, val_fraction=0.2, seed=42)
    n_total = train_df["unit"].nunique()
    n_val = val["unit"].nunique()
    # Allow ±1 unit rounding
    assert abs(n_val / n_total - 0.2) <= 1 / n_total + 0.01


# ---------------------------------------------------------------------------
# Normalisation (Protocol §3 — no leakage from val/test into train stats)
# ---------------------------------------------------------------------------

def test_normalisation_uses_train_stats(cmapss_dir):
    train_df, test_raw, _ = load_raw(cmapss_dir, SUBSET)
    train_df = add_rul(train_df)
    _, active = drop_zero_variance(train_df, ALL_FEATURE_COLS)
    tr, val = split_by_units(train_df, val_fraction=0.2, seed=42)

    tr_norm, _val_norm, _test_norm, _mean, _std = normalise(tr, active, val_df=val, test_df=test_raw)

    # Train statistics after normalisation should be ~0 mean, ~1 std
    tr_mean = tr_norm[active].mean()
    tr_std = tr_norm[active].std()
    np.testing.assert_allclose(tr_mean.values, 0.0, atol=1e-5,
                               err_msg="Normalised train mean must be ~0")
    np.testing.assert_allclose(tr_std.values, 1.0, atol=1e-2,
                               err_msg="Normalised train std must be ~1")


def test_normalisation_does_not_alter_train_on_test_application(cmapss_dir):
    """Applying normalisation to test must not change train statistics."""
    train_df, test_raw, _ = load_raw(cmapss_dir, SUBSET)
    train_df = add_rul(train_df)
    _, active = drop_zero_variance(train_df, ALL_FEATURE_COLS)
    tr, _ = split_by_units(train_df, val_fraction=0.2, seed=42)

    _tr_norm_before, _, _, mean_before, std_before = normalise(tr, active)
    _tr_norm_after, _, _, mean_after, std_after = normalise(tr, active, test_df=test_raw)

    np.testing.assert_array_equal(mean_before.values, mean_after.values)
    np.testing.assert_array_equal(std_before.values, std_after.values)


# ---------------------------------------------------------------------------
# Window shapes
# ---------------------------------------------------------------------------

def test_window_shapes(cmapss_dir):
    data = prepare_dataset(cmapss_dir, SUBSET, window=WINDOW)
    n_features = len(data["feature_cols"])
    assert data["X_train"].ndim == 3
    assert data["X_train"].shape[1] == WINDOW
    assert data["X_train"].shape[2] == n_features
    assert data["y_train"].ndim == 1
    assert len(data["X_train"]) == len(data["y_train"])


def test_window_label_is_last_cycle_rul(cmapss_dir):
    """Label of each window must equal the RUL at the last cycle in the window."""
    train_df, _, _ = load_raw(cmapss_dir, SUBSET)
    train_df = add_rul(train_df)
    _, active = drop_zero_variance(train_df, ALL_FEATURE_COLS)
    tr, _ = split_by_units(train_df, val_fraction=0.2, seed=42)
    tr_norm, _, _, _, _ = normalise(tr, active)
    tr_norm["rul"] = tr["rul"].values

    _X, y = make_windows(tr_norm, active, window=WINDOW)

    # Reconstruct expected labels: for each unit, expected label at position i
    # is rul[i + window - 1]
    expected = []
    for _, group in tr_norm.groupby("unit", sort=True):
        rul = group["rul"].values
        n = len(rul)
        if n < WINDOW:
            continue
        for start in range(n - WINDOW + 1):
            expected.append(rul[start + WINDOW - 1])

    np.testing.assert_allclose(y, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# Test windows (Protocol §5)
# ---------------------------------------------------------------------------

def test_one_test_window_per_unit(cmapss_dir):
    data = prepare_dataset(cmapss_dir, SUBSET, window=WINDOW)
    n_test_units = 20  # as defined in conftest
    assert data["X_test"].shape[0] == n_test_units, (
        f"Expected {n_test_units} test windows, got {data['X_test'].shape[0]}"
    )
    assert data["X_test"].shape[1] == WINDOW


def test_short_test_units_padded(cmapss_dir):
    """Every test window must be exactly `window` long (short ones padded)."""
    data = prepare_dataset(cmapss_dir, SUBSET, window=WINDOW)
    assert (data["X_test"].shape[1] == WINDOW), "All test windows must have length == window"


# ---------------------------------------------------------------------------
# split_by parameter: unit (correct) vs window (leakage demo)
# ---------------------------------------------------------------------------

def test_unit_split_units_disjoint(cmapss_dir):
    """Regression test for Protocol §2: unit split must produce disjoint unit sets."""
    train_df, _, _ = load_raw(cmapss_dir, SUBSET)
    train_df = add_rul(train_df)
    tr, val = split_by_units(train_df, val_fraction=0.2, seed=42)
    assert set(tr["unit"].unique()).isdisjoint(set(val["unit"].unique())), (
        "Unit split must produce disjoint train and val unit sets"
    )


def test_window_split_units_overlap(cmapss_dir):
    """Window split (WRONG) must put the same engine in both train and val windows.

    split_windows_randomly shuffles the pre-built window array randomly.
    Adjacent windows from the same engine differ by one cycle, so they will
    land on both sides of the split — that is the leakage being demonstrated.
    We reproduce the same permutation here to verify the unit overlap directly.
    """
    train_df, _, _ = load_raw(cmapss_dir, SUBSET)
    train_df = add_rul(train_df)
    _, active = drop_zero_variance(train_df, ALL_FEATURE_COLS)
    all_norm, _, _, _, _ = normalise(train_df, active)
    all_norm["rul"] = train_df["rul"].values

    # Build all windows and keep their unit IDs
    X_all, _y_all, _, unit_ids = make_windows_with_cycles(all_norm, active, WINDOW)

    # Reproduce split_windows_randomly(seed=42) to map windows → unit sets
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X_all))
    n_val = max(1, int(len(X_all) * 0.2))
    train_units = set(unit_ids[idx[n_val:]])
    val_units   = set(unit_ids[idx[:n_val]])

    assert train_units & val_units, (
        "Window split must put the same engine unit in both train and val — "
        "this overlap is the leakage mechanism"
    )


def test_windows_have_consecutive_cycles(cmapss_dir):
    """Every window must span consecutive cycles with no gaps — in both split modes.

    unit mode:   windows come from each engine's training portion (sorted by cycle).
    window mode: all windows are built before the split; split_windows_randomly only
                 reassigns existing window objects and cannot create gaps within them.
    Both cases are verified here by inspecting the raw cycle arrays.
    """
    train_df, _, _ = load_raw(cmapss_dir, SUBSET)
    train_df = add_rul(train_df)
    _, active = drop_zero_variance(train_df, ALL_FEATURE_COLS)

    # --- unit mode: window a train-only subset ---
    tr, _ = split_by_units(train_df, val_fraction=0.2, seed=42)
    tr_norm, _, _, _, _ = normalise(tr, active)
    tr_norm["rul"] = tr["rul"].values
    _, _, cycles_unit, _ = make_windows_with_cycles(tr_norm, active, WINDOW)

    for i, wc in enumerate(cycles_unit):
        assert np.all(np.diff(wc) == 1), (
            f"unit-mode window {i}: non-consecutive cycles {wc}"
        )

    # --- window mode: window the full training set (before random split) ---
    all_norm, _, _, _, _ = normalise(train_df, active)
    all_norm["rul"] = train_df["rul"].values
    _, _, cycles_win, _ = make_windows_with_cycles(all_norm, active, WINDOW)

    for i, wc in enumerate(cycles_win):
        assert np.all(np.diff(wc) == 1), (
            f"window-mode pre-split window {i}: non-consecutive cycles {wc}"
        )


def test_prepare_dataset_split_by_window_accepted(cmapss_dir):
    """prepare_dataset must accept split_by='window' without raising."""
    data = prepare_dataset(cmapss_dir, SUBSET, window=WINDOW, split_by="window")
    assert data["X_train"].ndim == 3
    assert data["X_val"].ndim == 3


def test_prepare_dataset_invalid_split_by_raises(cmapss_dir):
    import pytest
    with pytest.raises(ValueError, match="split_by"):
        prepare_dataset(cmapss_dir, SUBSET, window=WINDOW, split_by="random")
