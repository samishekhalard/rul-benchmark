"""
Сборка итоговой таблицы из отдельных прогонов.

Если для модели есть несколько сидов, считается среднее и стандартное
отклонение. Одиночный прогон без разброса — слабое доказательство:
разница в полтора пункта RMSE между архитектурами может целиком
объясняться инициализацией.

Запуск:
    python -m src.report --results-dir results/raw --out results/RESULTS.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

MODEL_LABELS = {
    "lstm": "LSTM",
    "gru": "GRU",
    "cnn1d": "1D-CNN",
    "rnn": "RNN",
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
}


def collect(results_dir: Path) -> dict[tuple[str, str], list[dict]]:
    grouped = defaultdict(list)
    for path in sorted(results_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        grouped[(payload["subset"], payload["model"])].append(payload)
    return grouped


def aggregate(runs: list[dict]) -> dict[str, str]:
    row = {}
    for metric in ("rmse", "mae", "r2", "phm_score"):
        values = np.array([run["test"][metric] for run in runs], dtype=float)
        precision = 4 if metric == "r2" else 2
        if len(values) == 1:
            row[metric] = f"{values[0]:.{precision}f}"
        else:
            row[metric] = f"{values.mean():.{precision}f} ± {values.std(ddof=0):.{precision}f}"
    row["seeds"] = str(len(runs))
    params = [run.get("parameters") for run in runs if run.get("parameters")]
    row["parameters"] = f"{int(np.mean(params)):,}".replace(",", " ") if params else "—"
    seconds = [run.get("train_seconds") for run in runs if run.get("train_seconds")]
    row["train_seconds"] = f"{np.mean(seconds):.0f}" if seconds else "—"
    return row


def render(grouped: dict[tuple[str, str], list[dict]]) -> str:
    lines = ["# Результаты", ""]
    subsets = sorted({subset for subset, _ in grouped})

    for subset in subsets:
        lines.append(f"## {subset}")
        lines.append("")
        lines.append(
            "| Модель | RMSE ↓ | MAE ↓ | R² ↑ | PHM score ↓ | Параметры | Обучение, с | Сидов |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")

        rows = []
        for (current_subset, model), runs in grouped.items():
            if current_subset != subset:
                continue
            stats = aggregate(runs)
            sort_key = float(stats["rmse"].split(" ")[0])
            rows.append(
                (
                    sort_key,
                    f"| {MODEL_LABELS.get(model, model)} | {stats['rmse']} | {stats['mae']} | "
                    f"{stats['r2']} | {stats['phm_score']} | {stats['parameters']} | "
                    f"{stats['train_seconds']} | {stats['seeds']} |",
                )
            )

        for _, line in sorted(rows):
            lines.append(line)
        lines.append("")

    lines.append("Стрелки показывают направление улучшения. PHM score — асимметричная метрика")
    lines.append("конкурса PHM 2008: запоздалый прогноз штрафуется сильнее раннего.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Сборка таблицы результатов")
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument("--out", default="results/RESULTS.md")
    args = parser.parse_args()

    grouped = collect(Path(args.results_dir))
    if not grouped:
        raise SystemExit(f"В {args.results_dir} нет файлов результатов. Сначала запустите обучение.")

    text = render(grouped)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
