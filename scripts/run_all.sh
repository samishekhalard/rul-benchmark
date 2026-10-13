#!/usr/bin/env bash
# Run the full benchmark: all models × FD001, seeds 42/1337/2024.
# Each model is trained three times (once per seed) via train.py's default
# seed loop.  Adjust --epochs and --patience for a full vs. smoke run.

set -euo pipefail

DATA_DIR="${1:-data}"
EPOCHS="${2:-100}"
SUBSET="FD001"

MODELS_NEURAL=(lstm gru cnn rnn)
MODELS_BASELINE=(xgboost random_forest)

echo "=== RUL Benchmark — subset ${SUBSET} ==="
echo "Data dir : ${DATA_DIR}"
echo "Epochs   : ${EPOCHS}"
echo ""

for model in "${MODELS_NEURAL[@]}"; do
    echo "--- Neural: ${model} ---"
    python src/train.py \
        --model "${model}" \
        --subset "${SUBSET}" \
        --data-dir "${DATA_DIR}" \
        --epochs "${EPOCHS}" \
        --patience 15
done

for model in "${MODELS_BASELINE[@]}"; do
    echo "--- Baseline: ${model} ---"
    python src/train.py \
        --model "${model}" \
        --subset "${SUBSET}" \
        --data-dir "${DATA_DIR}"
done

echo ""
echo "=== Generating report ==="
python src/report.py --results-dir results/raw --output results/RESULTS.md
echo "Done. See results/RESULTS.md"
