"""
Tests for the chronological train/val/test split utilities.

Split logic is simple but critical: if future data leaks into training,
all evaluation metrics are invalidated. These tests ensure the split
preserves temporal ordering.
"""

import pandas as pd
import pytest

from src.split import train_val_test_split, walk_forward_windows


@pytest.fixture
def sample_df():
    """Small time-ordered synthetic dataset."""
    dates = pd.date_range("2020-01-01", periods=100, freq="7D")
    return pd.DataFrame(
        {
            "date": dates,
            "home_team": ["TeamA"] * 100,
            "away_team": ["TeamB"] * 100,
            "home_goals": [1] * 100,
            "away_goals": [0] * 100,
            "result": ["H"] * 100,
        }
    )


class TestTrainValTestSplit:
    def test_returns_three_dataframes(self, sample_df):
        train, val, test = train_val_test_split(sample_df)
        assert isinstance(train, pd.DataFrame)
        assert isinstance(val, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)

    def test_preserves_chronological_order(self, sample_df):
        train, val, test = train_val_test_split(sample_df, ratios=[0.6, 0.15, 0.25])
        assert train["date"].max() < val["date"].min()
        assert val["date"].max() < test["date"].min()

    def test_total_rows_preserved(self, sample_df):
        train, val, test = train_val_test_split(sample_df, ratios=[0.5, 0.25, 0.25])
        assert len(train) + len(val) + len(test) == len(sample_df)

    def test_custom_ratios(self, sample_df):
        train, val, test = train_val_test_split(sample_df, ratios=[0.7, 0.1, 0.2])
        n = len(sample_df)
        assert len(train) == int(n * 0.7)
        assert len(val) == int(n * 0.1)
        assert len(test) == int(n * 0.2)

    def test_invalid_ratios_raises(self, sample_df):
        with pytest.raises(AssertionError):
            train_val_test_split(sample_df, ratios=[0.5, 0.5, 0.3])  # sums to 1.3
        with pytest.raises(AssertionError):
            train_val_test_split(sample_df, ratios=[0.5, 0.3])  # only 2 ratios


class TestWalkForwardWindows:
    def test_returns_list_of_tuples(self, sample_df):
        windows = walk_forward_windows(sample_df, n_windows=3)
        assert isinstance(windows, list)
        assert all(isinstance(w, tuple) for w in windows)

    def test_windows_are_sequential(self, sample_df):
        windows = walk_forward_windows(sample_df, n_windows=5)
        # Each window should have train_end > 0, val_end > train_end
        for start, end in windows:
            assert 0 < start < end <= len(sample_df)
        # Val sets should not go backwards
        for i in range(len(windows) - 1):
            assert windows[i][1] < windows[i + 1][1]

    def test_windows_count(self, sample_df):
        windows = walk_forward_windows(sample_df, n_windows=3)
        assert len(windows) <= 3

    def test_min_train_respected(self, sample_df):
        n = len(sample_df)
        windows = walk_forward_windows(
            sample_df, n_windows=2, min_train_frac=0.5, val_frac=0.1
        )
        min_train = int(n * 0.5)
        assert windows[0][0] >= min_train
