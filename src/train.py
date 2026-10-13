"""
Training script for neural RUL models.

Usage:
    python src/train.py --model lstm --subset FD001 --data-dir data/ --epochs 100
    python src/train.py --model xgboost --subset FD001 --data-dir data/
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from baselines import build_baseline, predict_baseline, train_baseline
from data import prepare_dataset
from metrics import compute_all
from models import build_model, count_parameters

NEURAL_MODELS = {"lstm", "gru", "cnn", "rnn"}
BASELINE_MODELS = {"xgboost", "xgb", "random_forest", "rf"}


def fix_seed(seed: int) -> None:
    """Fix all sources of randomness for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN ops (may slow training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_neural(args, data: dict, seed: int, results_dir: Path) -> dict:
    fix_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_features = data["X_train"].shape[2]
    model = build_model(args.model, n_features=n_features).to(device)
    n_params = count_parameters(model)

    X_tr = torch.from_numpy(data["X_train"])
    y_tr = torch.from_numpy(data["y_train"])
    X_val = torch.from_numpy(data["X_val"]).to(device)
    X_test = torch.from_numpy(data["X_test"]).to(device)

    loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=args.batch_size,
        shuffle=True,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    best_val_rmse = float("inf")
    best_state = None
    patience_counter = 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val).cpu().numpy()
        val_rmse = float(np.sqrt(((val_pred - data["y_val"]) ** 2).mean()))
        scheduler.step(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    elapsed = time.time() - t_start

    # Restore best weights and predict on test set
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test).cpu().numpy()

    metrics = compute_all(data["y_test"], test_pred)

    result = {
        "model": args.model,
        "subset": args.subset,
        "seed": seed,
        "split_by": args.split_by,
        "protocol": {
            "window": args.window,
            "rul_cap": args.rul_cap,
            "val_fraction": 0.2,
            "split_by": args.split_by,
        },
        "n_params": n_params,
        "train_time_s": round(elapsed, 2),
        "val_rmse": round(best_val_rmse, 4),
        **{k: round(v, 4) for k, v in metrics.items()},
    }

    tag = f"{args.subset}_{args.model}_{args.split_by}_seed{seed}"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{tag}.json", "w") as f:
        json.dump(result, f, indent=2)
    np.save(results_dir / f"{tag}_preds.npy", test_pred)

    return result


def train_baseline_model(args, data: dict, seed: int, results_dir: Path) -> dict:
    fix_seed(seed)
    t_start = time.time()

    model = build_baseline(args.model, seed=seed)
    train_baseline(model, data["X_train"], data["y_train"])
    elapsed = time.time() - t_start

    val_pred = predict_baseline(model, data["X_val"])
    val_rmse = float(np.sqrt(((val_pred - data["y_val"]) ** 2).mean()))

    test_pred = predict_baseline(model, data["X_test"])
    metrics = compute_all(data["y_test"], test_pred)

    # Tree models don't have trainable parameters in the PyTorch sense
    n_params = getattr(model, "n_estimators", 0) or 0

    result = {
        "model": args.model,
        "subset": args.subset,
        "seed": seed,
        "split_by": args.split_by,
        "protocol": {
            "window": args.window,
            "rul_cap": args.rul_cap,
            "val_fraction": 0.2,
            "split_by": args.split_by,
        },
        "n_params": n_params,
        "train_time_s": round(elapsed, 2),
        "val_rmse": round(val_rmse, 4),
        **{k: round(v, 4) for k, v in metrics.items()},
    }

    tag = f"{args.subset}_{args.model}_{args.split_by}_seed{seed}"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{tag}.json", "w") as f:
        json.dump(result, f, indent=2)
    np.save(results_dir / f"{tag}_preds.npy", test_pred)

    return result


def parse_args():
    p = argparse.ArgumentParser(description="Train a RUL prediction model on C-MAPSS")
    p.add_argument("--model", required=True, help="lstm | gru | cnn | rnn | xgboost | random_forest")
    p.add_argument("--subset", default="FD001", help="C-MAPSS subset (FD001–FD004)")
    p.add_argument("--data-dir", default="data", help="Directory with C-MAPSS .txt files")
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--rul-cap", type=int, default=125)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=None,
                   help="Single seed. Omit to run all three protocol seeds.")
    p.add_argument(
        "--split-by",
        default="unit",
        choices=["unit", "window"],
        help=(
            "Train/val split strategy. "
            "'unit' (default) is correct (Protocol §2). "
            "'window' is METHODOLOGICALLY INCORRECT and exists only for the "
            "leakage experiment (scripts/leakage_experiment.sh)."
        ),
    )
    p.add_argument("--results-dir", default="results/raw")
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [args.seed] if args.seed is not None else [42, 1337, 2024]
    results_dir = Path(args.results_dir)

    data = prepare_dataset(
        data_dir=args.data_dir,
        subset=args.subset,
        window=args.window,
        rul_cap=args.rul_cap,
        seed=seeds[0],  # split seed; kept constant across model seeds
        split_by=args.split_by,
    )

    if args.split_by == "window":
        print(
            "WARNING: --split-by window is METHODOLOGICALLY INCORRECT. "
            "Val RMSE will be artificially low due to leakage. "
            "Use only for the leakage experiment."
        )

    model_name = args.model.lower()
    for seed in seeds:
        print(f"\n[{args.model}] subset={args.subset} seed={seed}")
        if model_name in NEURAL_MODELS:
            result = train_neural(args, data, seed, results_dir)
        elif model_name in BASELINE_MODELS:
            result = train_baseline_model(args, data, seed, results_dir)
        else:
            raise ValueError(f"Unknown model '{args.model}'")
        print(
            f"  RMSE={result['rmse']:.2f}  MAE={result['mae']:.2f}"
            f"  R²={result['r2']:.3f}  PHM={result['phm_score']:.1f}"
            f"  time={result['train_time_s']:.1f}s"
        )


if __name__ == "__main__":
    main()
