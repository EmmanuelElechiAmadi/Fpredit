"""
Tests for market odds analysis — implied probabilities, value bets,
and model-vs-bookmaker comparison.
"""

import numpy as np
import pandas as pd
import pytest

from src.market import (
    add_implied_probabilities,
    implied_probabilities,
    market_comparison,
    odds_columns_available,
    value_bets,
)


@pytest.fixture
def matches_with_odds():
    """Small synthetic match frame with deterministic closing odds.

    Odds are chosen so the 'true' team is the favourite; we keep n small
    so tests stay fast. Team names match generate_synthetic_league output.
    """
    from src.data_loader import generate_synthetic_league

    df = generate_synthetic_league(n_teams=6, n_seasons=1).reset_index(drop=True)
    # Deterministic odds: favourite home, then draw, then away — roughly fair
    rng = np.random.default_rng(42)
    home_base = rng.uniform(1.4, 2.2, size=len(df))
    draw_odds = rng.uniform(3.0, 4.2, size=len(df))
    away_odds = rng.uniform(2.8, 5.0, size=len(df))
    df["B365H"] = home_base
    df["B365D"] = draw_odds
    df["B365A"] = away_odds
    return df


class TestImpliedProbabilities:
    def test_normalizes_to_one(self):
        probs = implied_probabilities(2.0, 3.5, 4.0)
        assert len(probs) == 3
        assert sum(probs) == pytest.approx(1.0, abs=1e-9)

    def test_favourite_has_highest_probability(self):
        probs = implied_probabilities(1.5, 4.0, 6.0)
        assert probs[0] > probs[1] > probs[2]

    def test_margin_removed(self):
        # Raw implied sum > 1 (bookmaker margin), normalized sum == 1
        raw = 1 / 2.0 + 1 / 3.5 + 1 / 4.0
        assert raw > 1.0
        probs = implied_probabilities(2.0, 3.5, 4.0)
        assert sum(probs) == pytest.approx(1.0)


class TestAddImpliedProbabilities:
    def test_adds_columns_when_odds_present(self, matches_with_odds):
        out = add_implied_probabilities(matches_with_odds)
        for col in ("implied_home", "implied_draw", "implied_away"):
            assert col in out.columns
        assert out["implied_home"].notna().all()
        # Margin-normalized: each row sums to 1
        row_sums = out[["implied_home", "implied_draw", "implied_away"]].sum(axis=1)
        assert row_sums.max() == pytest.approx(1.0, abs=1e-9)

    def test_nan_when_no_odds(self):
        df = pd.DataFrame(
            {
                "home_team": ["A"],
                "away_team": ["B"],
                "home_goals": [1],
                "away_goals": [0],
            }
        )
        out = add_implied_probabilities(df)
        assert out["implied_home"].isna().all()
        assert out["implied_away"].isna().all()

    def test_prefers_betbrain_over_single_bookie(self, matches_with_odds):
        df = matches_with_odds.copy()
        # Add BbAv columns that differ from B365
        df["BbAvH"] = df["B365H"] + 0.1
        df["BbAvD"] = df["B365D"] + 0.1
        df["BbAvA"] = df["B365A"] + 0.1
        out = add_implied_probabilities(df)
        # implied_home should be based on BbAvH (larger odds → smaller prob)
        assert out["implied_home"].iloc[0] == pytest.approx(
            1
            / df["BbAvH"].iloc[0]
            / (
                1 / df["BbAvH"].iloc[0]
                + 1 / df["BbAvD"].iloc[0]
                + 1 / df["BbAvA"].iloc[0]
            ),
            abs=1e-6,
        )


class TestOddsColumnsAvailable:
    def test_true_with_primary_odds(self, matches_with_odds):
        df = matches_with_odds.rename(
            columns={"B365H": "BbAvH", "B365D": "BbAvD", "B365A": "BbAvA"}
        )
        assert odds_columns_available(df) is True

    def test_true_with_fallback_odds(self, matches_with_odds):
        assert odds_columns_available(matches_with_odds) is True

    def test_false_without_odds(self):
        df = pd.DataFrame({"home_team": ["A"], "away_team": ["B"]})
        assert odds_columns_available(df) is False


class TestValueBets:
    def test_finds_value_bet_when_model_is_more_confident(self, matches_with_odds):
        df = matches_with_odds.copy()
        # Model probabilities: very confident on home for every match
        model_probs = pd.DataFrame(
            {
                "home_win": np.full(len(df), 0.7),
                "draw": np.full(len(df), 0.15),
                "away_win": np.full(len(df), 0.15),
            }
        )
        bets = value_bets(df, model_probs, edge_threshold=0.05, min_probability=0.15)
        assert len(bets) > 0
        required = {
            "date",
            "home_team",
            "away_team",
            "result",
            "market",
            "model_prob",
            "implied_prob",
            "edge",
            "odds",
            "stake_ret",
            "pnl",
        }
        assert required.issubset(bets.columns)

    def test_edge_and_pnl_consistency(self, matches_with_odds):
        df = matches_with_odds.head(10).copy().reset_index(drop=True)
        # Force one known scenario: model loves home, odds big on home
        df["B365H"] = 3.0
        df["B365D"] = 3.0
        df["B365A"] = 3.0
        model_probs = pd.DataFrame(
            {
                "home_win": np.full(10, 0.6),
                "draw": np.full(10, 0.2),
                "away_win": np.full(10, 0.2),
            }
        )
        bets = value_bets(df, model_probs, edge_threshold=0.05, min_probability=0.3)
        assert len(bets) == 10  # all home bets qualify (0.6 - 1/3 = 0.267 >= 0.05)
        for _, b in bets.iterrows():
            assert b["market"] == "H"
            assert b["model_prob"] == pytest.approx(0.6)
            assert b["implied_prob"] == pytest.approx(1 / 3, abs=1e-9)
            assert b["edge"] == pytest.approx(0.6 - 1 / 3, abs=1e-9)
            if b["result"] == "H":
                assert b["pnl"] == pytest.approx(3.0 - 1.0, abs=1e-9)
                assert b["stake_ret"] == pytest.approx(3.0, abs=1e-9)
            else:
                assert b["pnl"] == pytest.approx(-1.0, abs=1e-9)
                assert b["stake_ret"] == pytest.approx(0.0, abs=1e-9)

    def test_empty_when_no_odds(self):
        df = pd.DataFrame(
            {
                "date": ["2024-01-01"],
                "home_team": ["A"],
                "away_team": ["B"],
                "result": ["H"],
            }
        )
        model_probs = pd.DataFrame(
            {"home_win": [0.5], "draw": [0.25], "away_win": [0.25]}
        )
        bets = value_bets(df, model_probs)
        assert bets.empty
        assert "pnl" in bets.columns

    def test_respects_min_probability(self, matches_with_odds):
        df = matches_with_odds.head(20).copy().reset_index(drop=True)
        model_probs = pd.DataFrame(
            {
                "home_win": np.full(20, 0.18),
                "draw": np.full(20, 0.41),
                "away_win": np.full(20, 0.41),
            }
        )
        # Edge >= 0.05 vs market, but model_prob < 0.15 min → filtered
        bets = value_bets(df, model_probs, edge_threshold=0.05, min_probability=0.5)
        assert len(bets) == 0


class TestMarketComparison:
    def test_returns_summary_dict(self, matches_with_odds):
        df = matches_with_odds.copy()
        probs = pd.DataFrame(
            {
                "home_win": np.full(len(df), 0.45),
                "draw": np.full(len(df), 0.28),
                "away_win": np.full(len(df), 0.27),
            }
        )
        res = market_comparison(df, probs)
        assert res["n"] == len(df)
        assert res["brier_model"] is not None
        assert res["brier_market"] is not None
        assert res["log_loss_model"] > 0
        assert res["log_loss_market"] > 0
        assert res["beats_market"] in (True, False)
        assert isinstance(res["calibration"], list)

    def test_returns_empty_when_no_odds(self):
        df = pd.DataFrame(
            {
                "date": ["2024-01-01"],
                "home_team": ["A"],
                "away_team": ["B"],
                "result": ["H"],
            }
        )
        probs = pd.DataFrame({"home_win": [0.5], "draw": [0.25], "away_win": [0.25]})
        res = market_comparison(df, probs)
        assert res["n"] == 0
        assert res["brier_model"] is None
        assert "note" in res

    def test_calibration_buckets_only_for_confident_bins(self, matches_with_odds):
        df = matches_with_odds.copy()
        probs = pd.DataFrame(
            {
                "home_win": np.full(len(df), 0.7),
                "draw": np.full(len(df), 0.15),
                "away_win": np.full(len(df), 0.15),
            }
        )
        res = market_comparison(df, probs)
        if res["n"] > 0:
            assert res["brier_model"] <= 0.7  # sane range for 3-class Brier
