"""
Загрузка и подготовка данных NASA C-MAPSS.

Формат исходных файлов (train_FDxxx.txt / test_FDxxx.txt):
    26 колонок без заголовка, разделитель — пробел:
    unit, cycle, op_setting_1..3, sensor_1..21

RUL_FDxxx.txt содержит истинный остаточный ресурс для последнего цикла
каждого юнита из тестовой выборки.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

INDEX_COLUMNS = ["unit", "cycle"]
SETTING_COLUMNS = [f"op_setting_{i}" for i in range(1, 4)]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLUMNS = INDEX_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS

# Сенсоры с нулевой или почти нулевой дисперсией на FD001/FD003.
# Исключаются, чтобы не вносить шум в нормализацию.
CONSTANT_SENSORS_FD001 = [
    "sensor_1", "sensor_5", "sensor_6", "sensor_10",
    "sensor_16", "sensor_18", "sensor_19",
]


@dataclass
class Dataset:
    """Подготовленные окна для обучения и теста."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]

    @property
    def n_features(self) -> int:
        return self.x_train.shape[2]

    @property
    def window_size(self) -> int:
        return self.x_train.shape[1]


def read_raw(path: Path) -> pd.DataFrame:
    """Читает файл C-MAPSS в DataFrame с именованными колонками."""
    frame = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    frame = frame.dropna(axis=1, how="all")
    frame.columns = ALL_COLUMNS[: frame.shape[1]]
    return frame


def add_rul(frame: pd.DataFrame, cap: int | None = 125) -> pd.DataFrame:
    """
    Размечает остаточный ресурс для обучающей выборки.

    RUL = (последний цикл юнита) - (текущий цикл).

    cap ограничивает RUL сверху. Это стандартный приём для C-MAPSS:
    пока деградация не началась, точное значение остаточного ресурса
    по датчикам не восстановимо, и модель, обученная без ограничения,
    тратит ёмкость на предсказание неинформативного участка.
    Значение 125 взято из литературы и зафиксировано для всех моделей,
    чтобы сравнение оставалось честным.
    """
    last_cycle = frame.groupby("unit")["cycle"].transform("max")
    frame = frame.copy()
    frame["rul"] = last_cycle - frame["cycle"]
    if cap is not None:
        frame["rul"] = frame["rul"].clip(upper=cap)
    return frame


def select_features(frame: pd.DataFrame, drop_constant: bool = True) -> list[str]:
    """Возвращает список признаков, исключая вырожденные сенсоры."""
    features = SETTING_COLUMNS + SENSOR_COLUMNS
    if not drop_constant:
        return features

    kept = []
    for column in features:
        if column not in frame.columns:
            continue
        if frame[column].std() < 1e-6:
            continue
        kept.append(column)
    return kept


def fit_scaler(frame: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Считает параметры нормализации ТОЛЬКО по обучающей выборке.

    Это ключевой момент протокола: статистики теста в обучение не попадают,
    иначе результаты завышаются за счёт утечки.
    """
    values = frame[features].to_numpy(dtype=np.float32)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


def apply_scaler(
    frame: pd.DataFrame,
    features: list[str],
    mean: np.ndarray,
    std: np.ndarray,
) -> pd.DataFrame:
    frame = frame.copy()
    frame[features] = (frame[features].to_numpy(dtype=np.float32) - mean) / std
    return frame


def make_train_windows(
    frame: pd.DataFrame,
    features: list[str],
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Нарезает скользящие окна внутри каждого юнита.

    Метка окна — RUL последнего цикла в окне.
    Юниты с историей короче окна пропускаются.
    """
    sequences, targets = [], []
    for _, unit_frame in frame.groupby("unit", sort=False):
        values = unit_frame[features].to_numpy(dtype=np.float32)
        labels = unit_frame["rul"].to_numpy(dtype=np.float32)
        if len(values) < window:
            continue
        for end in range(window, len(values) + 1):
            sequences.append(values[end - window : end])
            targets.append(labels[end - 1])
    return np.asarray(sequences), np.asarray(targets)


def make_test_windows(
    frame: pd.DataFrame,
    features: list[str],
    window: int,
) -> tuple[np.ndarray, list[int]]:
    """
    Для теста берётся ОДНО последнее окно на юнит — так устроена
    постановка задачи в C-MAPSS: предсказать остаточный ресурс
    на момент обрыва записи.

    Короткие истории дополняются повтором первого цикла в начало.
    """
    sequences, units = [], []
    for unit, unit_frame in frame.groupby("unit", sort=False):
        values = unit_frame[features].to_numpy(dtype=np.float32)
        if len(values) < window:
            padding = np.repeat(values[:1], window - len(values), axis=0)
            values = np.concatenate([padding, values], axis=0)
        sequences.append(values[-window:])
        units.append(int(unit))
    return np.asarray(sequences), units


def split_units(frame: pd.DataFrame, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """
    Делит обучающую выборку на train и val ПО ЮНИТАМ, а не по окнам.

    Разбиение по окнам дало бы утечку: соседние окна одного двигателя
    почти идентичны, и часть валидации оказалась бы в обучении.
    """
    units = np.array(sorted(frame["unit"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(units)
    n_val = max(1, int(round(len(units) * val_fraction)))
    return units[n_val:].tolist(), units[:n_val].tolist()


def load_dataset(
    data_dir: str | Path,
    subset: str = "FD001",
    window: int = 30,
    rul_cap: int | None = 125,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Dataset:
    """Полный пайплайн подготовки данных для одного подмножества C-MAPSS."""
    data_dir = Path(data_dir)

    train_raw = read_raw(data_dir / f"train_{subset}.txt")
    test_raw = read_raw(data_dir / f"test_{subset}.txt")
    rul_true = read_raw(data_dir / f"RUL_{subset}.txt").iloc[:, 0].to_numpy(dtype=np.float32)

    train_raw = add_rul(train_raw, cap=rul_cap)
    features = select_features(train_raw)

    train_units, val_units = split_units(train_raw, val_fraction, seed)
    train_part = train_raw[train_raw["unit"].isin(train_units)]
    val_part = train_raw[train_raw["unit"].isin(val_units)]

    mean, std = fit_scaler(train_part, features)
    train_part = apply_scaler(train_part, features, mean, std)
    val_part = apply_scaler(val_part, features, mean, std)
    test_scaled = apply_scaler(test_raw, features, mean, std)

    x_train, y_train = make_train_windows(train_part, features, window)
    x_val, y_val = make_train_windows(val_part, features, window)
    x_test, test_units = make_test_windows(test_scaled, features, window)

    y_test = rul_true[np.array(test_units) - 1]
    if rul_cap is not None:
        y_test = np.clip(y_test, None, rul_cap)

    return Dataset(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        feature_names=features,
    )
