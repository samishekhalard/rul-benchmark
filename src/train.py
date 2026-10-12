"""
Обучение и оценка одной модели по единому протоколу.

Запуск:
    python -m src.train --model lstm --subset FD001
    python -m src.train --model cnn1d --subset FD001 --epochs 80
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import load_dataset
from src.metrics import evaluate_all
from src.models import build_model, count_parameters


def set_seed(seed: int) -> None:
    """Фиксация всех источников случайности — иначе сравнение не воспроизводится."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    outputs = []
    for batch_x, _ in loader:
        outputs.append(model(batch_x.to(device)).cpu().numpy())
    return np.concatenate(outputs)


def run(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_dataset(
        data_dir=args.data_dir,
        subset=args.subset,
        window=args.window,
        rul_cap=args.rul_cap,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    train_loader = make_loader(dataset.x_train, dataset.y_train, args.batch_size, shuffle=True)
    val_loader = make_loader(dataset.x_val, dataset.y_val, args.batch_size, shuffle=False)
    test_loader = make_loader(dataset.x_test, dataset.y_test, args.batch_size, shuffle=False)

    model = build_model(args.model, n_features=dataset.n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    epochs_without_improvement = 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * len(batch_x)

        train_loss = running / len(train_loader.dataset)
        val_pred = predict(model, val_loader, device)
        val_rmse = float(np.sqrt(np.mean((dataset.y_val - val_pred) ** 2)))
        scheduler.step(val_rmse)

        if val_rmse < best_val - 1e-4:
            best_val = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"epoch {epoch:3d} | train mse {train_loss:8.2f} | "
                f"val rmse {val_rmse:6.2f} | best {best_val:6.2f}"
            )

        if epochs_without_improvement >= args.patience:
            print(f"Ранняя остановка на эпохе {epoch}: валидация не улучшается {args.patience} эпох")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed = time.time() - started
    test_pred = predict(model, test_loader, device)
    metrics = evaluate_all(dataset.y_test, test_pred)

    result = {
        "model": args.model,
        "subset": args.subset,
        "seed": args.seed,
        "window": args.window,
        "rul_cap": args.rul_cap,
        "parameters": count_parameters(model),
        "train_seconds": round(elapsed, 1),
        "val_rmse": round(best_val, 3),
        "test": metrics,
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{args.subset}_{args.model}_seed{args.seed}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    np.save(results_dir / f"{args.subset}_{args.model}_seed{args.seed}_pred.npy", test_pred)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение модели прогноза RUL на C-MAPSS")
    parser.add_argument("--model", default="lstm", choices=["lstm", "gru", "cnn1d", "rnn"])
    parser.add_argument("--subset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--data-dir", default="data/CMAPSS")
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--rul-cap", type=int, default=125)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
