"""
Unit tests for walk-forward backtesting.
"""

import numpy as np
import pandas as pd
import pytest

from backtest import evaluate, walk_forward_backtest
from src.data_loader import generate_synthetic_league


@pytest.fixture(scope="module")
def synthetic_2_seasons():
    # Small dataset so the walk-forward folds fit quickly.
    return generate_synthetic_league(n_teams=8, n_seasons=2)


@pytest.fixture(scope="module")
def backtest_results(synthetic_2_seasons):
    df = synthetic_2_seasons
    return walk_forward_backtest(df, min_train_matches=40, step_matches=20)


class TestWalkForwardBacktest:
    def test_returns_dataframe(self, backtest_results):
        assert isinstance(backtest_results, pd.DataFrame)

    def test_has_expected_columns(self, backtest_results):
        expected = {
            "date",
            "home_team",
            "away_team",
            "actual",
            "pred_home_win",
            "pred_draw",
            "pred_away_win",
            "pred_over_2_5",
            "pred_btts_yes",
            "expected_home_goals",
            "expected_away_goals",
        }
        assert expected.issubset(backtest_results.columns)

    def test_produces_predictions(self, backtest_results):
        """Walk-forward should generate at least some predictions."""
        assert len(backtest_results) > 0

    def test_probabilities_sum_to_one(self, backtest_results):
        """Each prediction's H/D/A probs should sum to 1."""
        sums = (
            backtest_results["pred_home_win"]
            + backtest_results["pred_draw"]
            + backtest_results["pred_away_win"]
        )
        assert np.allclose(sums, 1.0, atol=1e-6)

    def test_actual_is_valid_result(self, backtest_results):
        """All actual results should be H, D, or A."""
        assert set(backtest_results["actual"]).issubset({"H", "D", "A"})

    def test_no_leakage_in_dates(self, synthetic_2_seasons, backtest_results):
        """Every predicted row's date must be >= the first test block's cutoff,
        i.e. predictions are only ever made on matches after the training window."""
        df = synthetic_2_seasons.sort_values("date").reset_index(drop=True)
        # With min_train_matches=40, the first 40 matches are the initial training set.
        train_cutoff_date = df.iloc[39]["date"]
        assert (backtest_results["date"] >= train_cutoff_date).all()

    def test_empty_results_with_too_little_data(self):
        """With insufficient data, backtest should produce an empty frame (not crash)."""
        small_df = generate_synthetic_league(n_seasons=1, n_teams=6)
        result = walk_forward_backtest(small_df)
        assert result.empty


class TestEvaluate:
    def test_evaluate_returns_scores(self, backtest_results):
        metrics = evaluate(backtest_results)
        expected_keys = {"log_loss", "brier", "accuracy", "baseline_log_loss"}
        assert expected_keys.issubset(metrics.keys())

    def test_metrics_are_finite(self, backtest_results):
        metrics = evaluate(backtest_results)
        for key, value in metrics.items():
            assert np.isfinite(value), f"{key} should be finite, got {value}"

    def test_accuracy_between_zero_and_one(self, backtest_results):
        metrics = evaluate(backtest_results)
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_log_loss_positive(self, backtest_results):
        metrics = evaluate(backtest_results)
        assert metrics["log_loss"] > 0
