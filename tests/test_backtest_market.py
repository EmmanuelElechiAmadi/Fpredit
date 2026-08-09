"""
Regression tests for the backtest market path: with odds present, evaluate()
must produce market comparison, residual log loss, edge correlation and the
Kelly staking report without crashing (this path was previously broken — the
backtest emitted `actual` while the market layer expected `result`).
"""

import numpy as np
import pytest

from backtest import evaluate, walk_forward_backtest
from src.data_loader import generate_synthetic_league
from src.xg_loader import generate_synthetic_xg


@pytest.fixture(scope="module")
def market_backtest_results():
    df = generate_synthetic_league(n_teams=6, n_seasons=2).reset_index(drop=True)
    rng = np.random.default_rng(7)
    df["B365H"] = rng.uniform(1.5, 3.0, size=len(df))
    df["B365D"] = rng.uniform(3.0, 4.5, size=len(df))
    df["B365A"] = rng.uniform(2.2, 5.0, size=len(df))
    xg = generate_synthetic_xg(df)
    return walk_forward_backtest(df, min_train_matches=40, step_matches=20, xg_df=xg)


class TestMarketBacktestPath:
    def test_does_not_crash_with_odds(self, market_backtest_results):
        assert len(market_backtest_results) > 0

    def test_evaluate_returns_market_metrics(self, market_backtest_results):
        metrics = evaluate(market_backtest_results)
        assert "market" in metrics
        assert metrics["market"]["n"] > 0
        assert "log_loss_model" in metrics["market"]
        assert "residual_log_loss" in metrics["market"]

    def test_returns_staking_and_edge_corr(self, market_backtest_results):
        metrics = evaluate(market_backtest_results)
        assert "edge_corr" in metrics
        assert "staking" in metrics
        assert metrics["staking"]["n"] >= 0

    def test_core_metrics_still_present(self, market_backtest_results):
        metrics = evaluate(market_backtest_results)
        for key in ("log_loss", "brier", "accuracy", "baseline_log_loss"):
            assert key in metrics
            assert np.isfinite(metrics[key])

    def test_evaluate_without_odds_skips_market(self):
        df = generate_synthetic_league(n_teams=6, n_seasons=2).reset_index(drop=True)
        rdf = walk_forward_backtest(df, min_train_matches=40, step_matches=20)
        metrics = evaluate(rdf, with_odds=True)
        assert metrics.get("market", {}).get("n", 0) == 0
        assert "staking" not in metrics
