"""
Pytest fixtures: synthetic C-MAPSS data in a temporary directory.

The synthetic dataset mirrors the real C-MAPSS format exactly:
  - 26 space-separated columns, no header
  - 40 training units, 20 test units
  - 7 constant sensors (simulating FD001's zero-variance features)
  - Remaining sensors have a linear degradation trend

No real data download is required for any test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

N_TRAIN_UNITS = 40
N_TEST_UNITS = 20
MIN_CYCLES = 80
MAX_CYCLES = 200
N_SENSORS = 21
N_OP_SETTINGS = 3
N_CONSTANT_SENSORS = 7  # mirrors FD001
SUBSET = "FD001"


def _make_unit(unit_id: int, n_cycles: int, rng: np.random.Generator) -> np.ndarray:
    """Generate one engine's time series."""
    rows = []
    for cycle in range(1, n_cycles + 1):
        degradation = cycle / n_cycles  # 0 → 1 over the unit's life
        op = rng.uniform(0, 1, size=N_OP_SETTINGS)
        sensors = np.zeros(N_SENSORS)
        # First 7 sensors are constant (zero-variance)
        sensors[:N_CONSTANT_SENSORS] = 42.0
        # Remaining sensors carry a degradation trend + small noise
        sensors[N_CONSTANT_SENSORS:] = (
            0.5 + degradation * 0.5 + rng.normal(0, 0.05, size=N_SENSORS - N_CONSTANT_SENSORS)
        )
        rows.append([unit_id, cycle] + list(op) + list(sensors))
    return np.array(rows)


@pytest.fixture(scope="session")
def cmapss_dir(tmp_path_factory) -> Path:
    """Write synthetic C-MAPSS files to a temporary directory."""
    rng = np.random.default_rng(0)
    tmp = tmp_path_factory.mktemp("cmapss")

    # --- Training data ---
    train_rows = []
    for uid in range(1, N_TRAIN_UNITS + 1):
        n_cycles = int(rng.integers(MIN_CYCLES, MAX_CYCLES + 1))
        train_rows.append(_make_unit(uid, n_cycles, rng))
    train_df = np.vstack(train_rows)
    np.savetxt(tmp / f"train_{SUBSET}.txt", train_df, fmt="%.6f")

    # --- Test data (last 50 cycles of a random-length unit) ---
    test_rows = []
    rul_values = []
    for uid in range(1, N_TEST_UNITS + 1):
        n_cycles_full = int(rng.integers(MIN_CYCLES, MAX_CYCLES + 1))
        observe = int(rng.integers(20, n_cycles_full))
        true_rul = n_cycles_full - observe
        unit_data = _make_unit(uid, observe, rng)
        test_rows.append(unit_data)
        rul_values.append(true_rul)
    test_df = np.vstack(test_rows)
    np.savetxt(tmp / f"test_{SUBSET}.txt", test_df, fmt="%.6f")

    rul_array = np.array(rul_values, dtype=float)
    np.savetxt(tmp / f"RUL_{SUBSET}.txt", rul_array, fmt="%d")

    return tmp
