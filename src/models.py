"""
Нейросетевые архитектуры для прогнозирования остаточного ресурса.

Все модели принимают вход формы (batch, window, features) и возвращают
одно число на последовательность. Размеры подобраны так, чтобы число
обучаемых параметров было сопоставимым — иначе сравнение архитектур
превращается в сравнение ёмкости.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :]).squeeze(-1)


class GRURegressor(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        return self.head(output[:, -1, :]).squeeze(-1)


class CNN1DRegressor(nn.Module):
    """
    Свёртки по временной оси. В отличие от рекуррентных моделей
    не хранит состояние, зато обучается заметно быстрее и лучше
    ловит локальные паттерны деградации.
    """

    def __init__(
        self,
        n_features: int,
        channels: int = 64,
        kernel_size: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.features = nn.Sequential(
            nn.Conv1d(n_features, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, window, features) -> (batch, features, window)
        x = x.transpose(1, 2)
        x = self.features(x).squeeze(-1)
        return self.head(x).squeeze(-1)


class RNNRegressor(nn.Module):
    """Простая рекуррентная сеть — нижняя граница для сравнения."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(x)
        return self.head(output[:, -1, :]).squeeze(-1)


MODEL_REGISTRY = {
    "lstm": LSTMRegressor,
    "gru": GRURegressor,
    "cnn1d": CNN1DRegressor,
    "rnn": RNNRegressor,
}


def build_model(name: str, n_features: int, **kwargs) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Неизвестная модель: {name}. Доступны: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](n_features=n_features, **kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
