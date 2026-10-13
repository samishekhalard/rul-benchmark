"""
Neural network models for RUL prediction.

Input shape: (batch, window, n_features)
Output shape: (batch,)  — single scalar RUL estimate

All architectures are sized so that trainable parameter counts are comparable
(target ~50 k–80 k parameters each), enabling a fair architecture comparison
without confounding model capacity with architecture choice.
"""

from __future__ import annotations

import torch
from torch import nn

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type] = {}


def register(name: str):
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def build_model(name: str, n_features: int, **kwargs) -> nn.Module:
    """Factory: instantiate a registered model by name."""
    name_lower = name.lower()
    if name_lower not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name_lower](n_features=n_features, **kwargs)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Shared head: flatten recurrent output → scalar
# ---------------------------------------------------------------------------

class _RegressionHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

@register("lstm")
class LSTMModel(nn.Module):
    """Stacked LSTM with a two-layer regression head.

    hidden_size=64, num_layers=2 → ~66 k parameters with 14 input features.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = _RegressionHead(hidden_size, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])  # last time step


# ---------------------------------------------------------------------------
# GRU
# ---------------------------------------------------------------------------

@register("gru")
class GRUModel(nn.Module):
    """Stacked GRU — fewer gates than LSTM, comparable parameter count.

    hidden_size=72 gives ~64 k parameters with 14 input features.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 72,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = _RegressionHead(hidden_size, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


# ---------------------------------------------------------------------------
# 1-D CNN
# ---------------------------------------------------------------------------

@register("cnn")
class CNN1DModel(nn.Module):
    """Two-block 1-D convolutional network.

    Each block: Conv1d → BatchNorm → ReLU → MaxPool.
    Temporal dimension is collapsed by global average pooling before the head.
    ~68 k parameters with 14 input features, window=30.
    """

    def __init__(
        self,
        n_features: int,
        num_filters: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv1d(n_features, num_filters, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv1d(num_filters, num_filters * 2, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(num_filters * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # global average pooling → (batch, 128, 1)
        )
        self.head = _RegressionHead(num_filters * 2, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, features) → (batch, features, window) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.encoder(x).squeeze(-1)  # (batch, num_filters*2)
        return self.head(x)


# ---------------------------------------------------------------------------
# Simple RNN
# ---------------------------------------------------------------------------

@register("rnn")
class SimpleRNNModel(nn.Module):
    """Vanilla (Elman) RNN — baseline to show why gated units matter.

    hidden_size=96 to match parameter count (~58 k with 14 input features).
    Expect worse performance on long sequences due to vanishing gradients.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 96,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            nonlinearity="tanh",
        )
        self.head = _RegressionHead(hidden_size, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])
