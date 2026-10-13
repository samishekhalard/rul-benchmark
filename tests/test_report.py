"""Tests for report.py result aggregation."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from report import aggregate, load_results, render_table


def _make_run(subset, model, seed, rmse, mae, r2, phm):
    return {
        "model": model,
        "subset": subset,
        "seed": seed,
        "protocol": {"window": 30, "rul_cap": 125, "val_fraction": 0.2},
        "n_params": 66000,
        "train_time_s": 12.5,
        "val_rmse": rmse + 2,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "phm_score": phm,
    }


@pytest.fixture
def json_results_dir(tmp_path):
    runs = [
        _make_run("FD001", "lstm", 42,   15.0, 12.0, 0.90, 500.0),
        _make_run("FD001", "lstm", 1337, 17.0, 14.0, 0.88, 600.0),
        _make_run("FD001", "lstm", 2024, 16.0, 13.0, 0.89, 550.0),
        _make_run("FD001", "gru",  42,   14.0, 11.0, 0.91, 450.0),
        _make_run("FD001", "gru",  1337, 16.0, 13.0, 0.89, 500.0),
    ]
    for r in runs:
        fname = f"{r['subset']}_{r['model']}_seed{r['seed']}.json"
        (tmp_path / fname).write_text(json.dumps(r))
    return tmp_path


def test_load_results(json_results_dir):
    records = load_results(json_results_dir)
    assert len(records) == 5


def test_aggregate_groups_by_subset_model(json_results_dir):
    records = load_results(json_results_dir)
    rows = aggregate(records)
    pairs = {(r["subset"], r["model"]) for r in rows}
    assert ("FD001", "lstm") in pairs
    assert ("FD001", "gru") in pairs


def test_aggregate_mean_rmse(json_results_dir):
    records = load_results(json_results_dir)
    rows = aggregate(records)
    lstm_row = next(r for r in rows if r["model"] == "lstm")
    expected_mean = (15.0 + 17.0 + 16.0) / 3
    assert abs(lstm_row["rmse"] - expected_mean) < 1e-6


def test_aggregate_seed_count(json_results_dir):
    records = load_results(json_results_dir)
    rows = aggregate(records)
    lstm_row = next(r for r in rows if r["model"] == "lstm")
    assert lstm_row["n_seeds"] == 3
    gru_row = next(r for r in rows if r["model"] == "gru")
    assert gru_row["n_seeds"] == 2


def test_sorted_by_rmse(json_results_dir):
    records = load_results(json_results_dir)
    rows = aggregate(records)
    fd001_rows = [r for r in rows if r["subset"] == "FD001"]
    rmses = [r["rmse"] for r in fd001_rows]
    assert rmses == sorted(rmses), "Rows must be sorted by ascending RMSE within subset"


def test_render_table_contains_headers(json_results_dir):
    records = load_results(json_results_dir)
    rows = aggregate(records)
    table = render_table(rows)
    assert "RMSE" in table
    assert "PHM Score" in table
    assert "lstm" in table
    assert "gru" in table
