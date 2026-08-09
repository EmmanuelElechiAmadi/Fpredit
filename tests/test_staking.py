"""
Tests for covariance-adjusted fractional Kelly staking and portfolio risk
metrics (Sharpe, max drawdown, CVaR).
"""

import pandas as pd
import pytest

from src.staking import (
    covariance_adjusted_stakes,
    fractional_kelly_stake,
    kelly_fraction,
    portfolio_report,
)


class TestKellyFraction:
    def test_positive_edge_gives_positive_kelly(self):
        # 60% chance at 2.0 odds -> f* = (0.6*2 - 1)/1 = 0.2
        assert kelly_fraction(0.6, 2.0) == pytest.approx(0.2)

    def test_fair_odds_gives_zero(self):
        assert kelly_fraction(0.5, 2.0) == pytest.approx(0.0)

    def test_negative_edge_clamped_to_zero(self):
        assert kelly_fraction(0.4, 2.0) == 0.0

    def test_fractional_kelly_capped(self):
        # huge edge at short odds -> full Kelly > cap -> returns the cap
        assert fractional_kelly_stake(
            0.95, 1.1, fraction=0.25, max_stake=0.1
        ) == pytest.approx(0.1)
        # modest edge -> scaled
        assert fractional_kelly_stake(
            0.6, 2.0, fraction=0.5, max_stake=0.1
        ) == pytest.approx(0.1)


class TestCovarianceAdjustedStakes:
    def test_single_bet_matches_scalar_kelly(self):
        bets = pd.DataFrame({"model_prob": [0.6], "odds": [2.0]})
        stakes = covariance_adjusted_stakes(bets, kelly_fraction=0.25, max_stake=0.1)
        assert stakes.iloc[0] == pytest.approx(0.05)

    def test_negative_edge_bets_get_zero(self):
        bets = pd.DataFrame({"model_prob": [0.3, 0.6], "odds": [2.0, 2.0]})
        stakes = covariance_adjusted_stakes(bets, kelly_fraction=0.25, max_stake=0.1)
        assert stakes.iloc[0] == 0.0  # negative edge (0.3*2 - 1 = -0.4)
        assert stakes.iloc[1] > 0.0

    def test_stakes_respect_cap(self):
        bets = pd.DataFrame({"model_prob": [0.8, 0.85], "odds": [1.5, 1.4]})
        stakes = covariance_adjusted_stakes(bets, kelly_fraction=0.25, max_stake=0.1)
        assert (stakes <= 0.1 + 1e-12).all()
        assert (stakes >= 0).all()

    def test_correlation_concentrates_weight_on_best_edge(self):
        """Positive cross-match correlation makes the high-edge bet absorb
        weight from the mediocre one (they are partially redundant): the
        weight ratio shifts toward the best edge."""
        bets = pd.DataFrame({"model_prob": [0.7, 0.55], "odds": [2.0, 2.0]})
        s_indep = covariance_adjusted_stakes(
            bets, corr=0.0, kelly_fraction=0.25, max_stake=0.5
        )
        s_corr = covariance_adjusted_stakes(
            bets, corr=0.2, kelly_fraction=0.25, max_stake=0.5
        )
        eps = 1e-12
        ratio_indep = s_indep.iloc[0] / (s_indep.iloc[1] + eps)
        ratio_corr = s_corr.iloc[0] / (s_corr.iloc[1] + eps)
        assert s_corr.iloc[1] > 0  # both bets still get a stake
        assert ratio_corr > ratio_indep

    def test_empty_bets_returns_empty(self):
        stakes = covariance_adjusted_stakes(pd.DataFrame())
        assert len(stakes) == 0


class TestPortfolioReport:
    def test_empty_returns_empty_stats(self):
        r = portfolio_report(pd.Series(dtype=float), pd.Series(dtype=float))
        assert r["n"] == 0
        assert r["sharpe"] is None

    def test_all_wins_positive_stats(self):
        pnl = pd.Series([0.2, 0.3, 0.25])
        stakes = pd.Series([0.2, 0.3, 0.25])
        r = portfolio_report(pnl, stakes)
        assert r["n"] == 3
        assert r["roi"] > 0
        assert r["max_drawdown"] == 0.0

    def test_cvar_negative_with_losses(self):
        pnl = pd.Series(
            [0.5, -0.4, 0.3, -0.5, -0.6, 0.1, -0.2, 0.4, -0.1, 0.2, -0.3, 0.6]
        )
        stakes = pd.Series([1.0] * 12)
        r = portfolio_report(pnl, stakes)
        assert r["cvar95"] < 0  # expected shortfall of the worst tail is negative
        assert r["max_drawdown"] < 0

    def test_sharpe_positive_when_mean_positive(self):
        pnl = pd.Series([0.2, 0.15, 0.1, 0.25, 0.18])
        stakes = pd.Series([1.0] * 5)
        r = portfolio_report(pnl, stakes)
        assert r["sharpe"] is not None and r["sharpe"] > 0
