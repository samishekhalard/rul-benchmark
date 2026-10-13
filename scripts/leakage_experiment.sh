#!/usr/bin/env bash
# Leakage experiment: quantify how much window-based train/val splitting
# inflates val RMSE vs the honest test RMSE.
#
# Trains LSTM on FD001 with two split strategies × three seeds, then prints
# a side-by-side comparison.
#
# Key insight: the test set is always isolated (one last window per unit,
# completely separate from training data) regardless of split_by.  So test
# RMSE is honest in both conditions.  Val RMSE inflated by window split
# vs val RMSE with unit split is the measure of self-deception.
#
# Usage: bash scripts/leakage_experiment.sh [data_dir] [epochs]

set -euo pipefail

DATA_DIR="${1:-data}"
EPOCHS="${2:-100}"
MODEL="lstm"
SUBSET="FD001"
SEEDS=(42 1337 2024)
RESULTS_DIR="results/leakage_experiment"

mkdir -p "${RESULTS_DIR}"

echo "=== Leakage Experiment: unit split vs window split ==="
echo "Model  : ${MODEL} on ${SUBSET}"
echo "Epochs : ${EPOCHS}"
echo "Data   : ${DATA_DIR}"
echo ""

for split in unit window; do
    echo "--- split_by=${split} ---"
    for seed in "${SEEDS[@]}"; do
        python src/train.py \
            --model "${MODEL}" \
            --subset "${SUBSET}" \
            --data-dir "${DATA_DIR}" \
            --epochs "${EPOCHS}" \
            --patience 15 \
            --seed "${seed}" \
            --split-by "${split}" \
            --results-dir "${RESULTS_DIR}"
    done
    echo ""
done

echo "=== Results ==="
python - <<'PYEOF'
import json, glob, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

results_dir = Path("results/leakage_experiment")
records = defaultdict(list)
for f in sorted(results_dir.glob("*.json")):
    r = json.loads(f.read_text())
    records[r["split_by"]].append(r)

print(f"{'Split':>8}  {'Val RMSE (mean±std)':>22}  {'Test RMSE (mean±std)':>22}  {'Val-Test gap':>12}")
print("-" * 74)
for split in ("unit", "window"):
    runs = records.get(split, [])
    if not runs:
        print(f"{split:>8}  no results found")
        continue
    val_rmses  = [r["val_rmse"] for r in runs]
    test_rmses = [r["rmse"]     for r in runs]
    val_m,  val_s  = np.mean(val_rmses),  np.std(val_rmses)
    test_m, test_s = np.mean(test_rmses), np.std(test_rmses)
    gap = test_m - val_m
    print(f"{split:>8}  {val_m:>8.2f} ± {val_s:>5.2f}  "
          f"{test_m:>8.2f} ± {test_s:>5.2f}  {gap:>+12.2f}")

print()
print("Interpretation:")
print("  unit   — val RMSE is an honest estimate (no overlap between train/val units)")
print("  window — val RMSE is inflated (leakage); test RMSE stays honest")
print("  gap    = test_rmse - val_rmse; positive gap with window split")
print("           is the direct measure of how much the split inflates reported val perf.")
PYEOF
