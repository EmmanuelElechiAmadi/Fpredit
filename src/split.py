"""
Chronological train/validation/test split utilities.

Matches are time-ordered, so standard k-fold or random splits are invalid
(they leak future information into training). This module provides:

    train_val_test_split(df, ratios)
        Chronological three-way split.
        
    walk_forward_dates(df, n_windows, min_train, val_frac)
        Returns split points for walk-forward time-series cross-validation.

Usage:
    from src.split import train_val_test_split
    train, val, test = train_val_test_split(df, ratios=[0.6, 0.15, 0.25])
"""

from typing import List, Optional

import pandas as pd


def train_val_test_split(
    df: pd.DataFrame,
    ratios: Optional[List[float]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train/val/test split based on date quantiles.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a 'date' column and be sorted (or will be sorted).
    ratios : list of float, optional
        Three ratios for train/val/test, by default [0.6, 0.15, 0.25].
        Must sum to 1.0.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        train, val, test splits in chronological order.
    """
    if ratios is None:
        ratios = [0.6, 0.15, 0.25]
    assert len(ratios) == 3, "Must provide exactly 3 ratios"
    assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1.0"

    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    train_end = int(n * ratios[0])
    val_end = train_end + int(n * ratios[1])

    train = df.iloc[:train_end].reset_index(drop=True)
    val = df.iloc[train_end:val_end].reset_index(drop=True)
    test = df.iloc[val_end:].reset_index(drop=True)
    return train, val, test


def walk_forward_windows(
    df: pd.DataFrame,
    n_windows: int = 5,
    min_train_frac: float = 0.4,
    val_frac: float = 0.15,
) -> list[tuple[int, int]]:
    """Generate chronological, non-overlapping walk-forward train/val split indices.

    Each window expands the training set:
        train_idx = [0, train_end)
        val_idx   = [train_end, train_end + val_size)

    Later windows include MORE training data (never less), so each val set
    is a strict future hold-out relative to its train set.

    Parameters
    ----------
    df : pd.DataFrame
        Data (sorted by date).
    n_windows : int
        Number of walk-forward windows.
    min_train_frac : float
        Minimum fraction of data in the first training window.
    val_frac : float
        Fraction of data to reserve for validation in each window.

    Returns
    -------
    list of (train_end, val_end) tuples
        Each tuple corresponds to one window's index boundaries.
    """
    n = len(df)
    min_train = int(n * min_train_frac)
    val_size = int(n * val_frac)
    if val_size < 1:
        return []

    remaining = n - min_train - val_size
    step = max(1, remaining // n_windows) if n_windows > 0 else remaining

    windows = []
    train_end = min_train
    for _ in range(n_windows):
        if train_end + val_size > n:
            break
        val_end = train_end + val_size
        windows.append((train_end, val_end))
        train_end = train_end + step  # advance the starting point, NOT compound

    return windows
