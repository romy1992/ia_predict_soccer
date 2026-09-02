from __future__ import annotations

from typing import Optional

import pandas as pd


def ensure_temporal_order(frame: pd.DataFrame, time_col: str) -> pd.DataFrame:
    if time_col not in frame.columns:
        raise ValueError(f"Colonna temporale non trovata: {time_col}")

    ordered = frame.copy()
    ordered[time_col] = pd.to_datetime(ordered[time_col], utc=True, errors="coerce")
    if ordered[time_col].isna().any():
        raise ValueError(f"Valori data non validi in {time_col}")

    ordered = ordered.sort_values(by=[time_col]).reset_index(drop=True)
    return ordered


def expanding_window_splits(
    frame: pd.DataFrame,
    time_col: str,
    n_splits: int = 5,
    min_train_size: int = 120,
    min_valid_size: int = 30,
) -> list[tuple[list[int], list[int]]]:
    ordered = ensure_temporal_order(frame, time_col)
    total_rows = len(ordered)
    if total_rows < (min_train_size + min_valid_size):
        return []

    max_splits = max(1, (total_rows - min_train_size) // min_valid_size)
    n_splits = min(n_splits, max_splits)

    splits: list[tuple[list[int], list[int]]] = []
    for idx in range(n_splits):
        train_end = min_train_size + idx * min_valid_size
        valid_start = train_end
        valid_end = min(valid_start + min_valid_size, total_rows)
        if valid_end <= valid_start:
            break

        train_idx = list(range(0, train_end))
        valid_idx = list(range(valid_start, valid_end))
        splits.append((train_idx, valid_idx))

    return splits


def rolling_window_splits(
    frame: pd.DataFrame,
    time_col: str,
    train_window_size: int,
    valid_window_size: int,
    step_size: Optional[int] = None,
) -> list[tuple[list[int], list[int]]]:
    if train_window_size <= 0 or valid_window_size <= 0:
        raise ValueError("train_window_size e valid_window_size devono essere > 0")

    ordered = ensure_temporal_order(frame, time_col)
    total_rows = len(ordered)
    step = step_size or valid_window_size
    if total_rows < (train_window_size + valid_window_size):
        return []

    splits: list[tuple[list[int], list[int]]] = []
    start = 0
    while True:
        train_start = start
        train_end = train_start + train_window_size
        valid_start = train_end
        valid_end = valid_start + valid_window_size

        if valid_end > total_rows:
            break

        train_idx = list(range(train_start, train_end))
        valid_idx = list(range(valid_start, valid_end))
        splits.append((train_idx, valid_idx))

        start += step

    return splits


def final_holdout_split(
    frame: pd.DataFrame,
    time_col: str,
    holdout_ratio: float = 0.2,
) -> tuple[list[int], list[int]]:
    if holdout_ratio <= 0 or holdout_ratio >= 1:
        raise ValueError("holdout_ratio deve essere tra 0 e 1")

    ordered = ensure_temporal_order(frame, time_col)
    total_rows = len(ordered)
    holdout_size = max(1, int(total_rows * holdout_ratio))

    split_index = total_rows - holdout_size
    train_idx = list(range(0, split_index))
    holdout_idx = list(range(split_index, total_rows))
    return train_idx, holdout_idx

