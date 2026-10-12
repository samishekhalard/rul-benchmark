"""
Базовые модели: XGBoost и Random Forest.

Градиентный бустинг не умеет работать с последовательностями напрямую,
поэтому окно разворачивается в плоский вектор признаков. Это честный
способ сравнения: данные те же, протокол тот же, отличается только
способ представления.

Запуск:
    python -m src.baselines --model xgboost --subset FD001
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data import load_dataset
from src.metrics import evaluate_all


def flatten_windows(x: np.ndarray) -> np.ndarray:
    """(n, window, features) -> (n, window * features)"""
    return x.reshape(x.shape[0], -1)


def build_model(name: str, seed: int):
    if name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
            early_stopping_rounds=30,
        )
    if name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Неизвестная базовая модель: {name}")


def run(args: argparse.Namespace) -> dict:
    dataset = load_dataset(
        data_dir=args.data_dir,
        subset=args.subset,
        window=args.window,
        rul_cap=args.rul_cap,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    x_train = flatten_windows(dataset.x_train)
    x_val = flatten_windows(dataset.x_val)
    x_test = flatten_windows(dataset.x_test)

    model = build_model(args.model, args.seed)

    if args.model == "xgboost":
        model.fit(x_train, dataset.y_train, eval_set=[(x_val, dataset.y_val)], verbose=False)
    else:
        model.fit(x_train, dataset.y_train)

    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)

    result = {
        "model": args.model,
        "subset": args.subset,
        "seed": args.seed,
        "window": args.window,
        "rul_cap": args.rul_cap,
        "n_features_flat": int(x_train.shape[1]),
        "val_rmse": round(float(np.sqrt(np.mean((dataset.y_val - val_pred) ** 2))), 3),
        "test": evaluate_all(dataset.y_test, test_pred),
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{args.subset}_{args.model}_seed{args.seed}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    np.save(results_dir / f"{args.subset}_{args.model}_seed{args.seed}_pred.npy", test_pred)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Базовые модели для прогноза RUL")
    parser.add_argument("--model", default="xgboost", choices=["xgboost", "random_forest"])
    parser.add_argument("--subset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--data-dir", default="data/CMAPSS")
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--rul-cap", type=int, default=125)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
