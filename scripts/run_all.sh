#!/usr/bin/env bash
# Полный прогон бенчмарка: 4 нейросетевые архитектуры и 2 базовые модели
# на трёх сидах. На CPU занимает несколько часов, на GPU — около получаса.
set -euo pipefail

SUBSET="${1:-FD001}"

for seed in 42 1337 2024; do
  for model in lstm gru cnn1d rnn; do
    echo "=== $model | $SUBSET | seed $seed ==="
    python -m src.train --model "$model" --subset "$SUBSET" --seed "$seed"
  done
  for model in xgboost random_forest; do
    echo "=== $model | $SUBSET | seed $seed ==="
    python -m src.baselines --model "$model" --subset "$SUBSET" --seed "$seed"
  done
done

python -m src.report --results-dir results/raw --out results/RESULTS.md
echo "Готово. Таблица: results/RESULTS.md"
