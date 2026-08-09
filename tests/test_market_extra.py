"""
Tests for the market-residual feature layer: implied probabilities as meta
features and opening->closing line movement.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import generate_synthetic_league
from src.market import add_market_features, opening_odds_columns_available


@pytest.fixture
def matches_with_odds():
    df = generate_synthetic_league(n_teams=6, n_seasons=1).reset_index(drop=True)
    rng = np.random.default_rng(42)
    df["B365H"] = rng.uniform(1.4, 2.2, size=len(df))
    df["B365D"] = rng.uniform(3.0, 4.2, size=len(df))
    df["B365A"] = rng.uniform(2.8, 5.0, size=len(df))
    return df


class TestOpeningOddsColumnsAvailable:
    def test_true_when_both_present(self, matches_with_odds):
        df = matches_with_odds.copy()
        df["PH"], df["PD"], df["PA"] = 2.0, 3.5, 4.0
        df["PSH"], df["PSD"], df["PSA"] = 1.9, 3.6, 4.2
        assert opening_odds_columns_available(df)

    def test_false_without_opening(self, matches_with_odds):
        assert not opening_odds_columns_available(matches_with_odds)


class TestAddMarketFeatures:
    def test_adds_market_probability_columns(self, matches_with_odds):
        out = add_market_features(matches_with_odds)
        for col in ("market_home", "market_draw", "market_away"):
            assert col in out.columns
        # Margin-normalized: each row sums to 1
        row_sums = out[["market_home", "market_draw", "market_away"]].sum(axis=1)
        assert row_sums.max() == pytest.approx(1.0, abs=1e-9)

    def test_line_movement_nan_without_opening(self, matches_with_odds):
        out = add_market_features(matches_with_odds)
        assert out["line_mv_home"].isna().all()

    def test_line_movement_computed_when_opening_present(self, matches_with_odds):
        df = matches_with_odds.copy()
        # Opening: slight favourite-shift vs closing
        df["PH"] = df["B365H"] - 0.05
        df["PD"] = df["B365D"] + 0.1
        df["PA"] = df["B365A"] + 0.2
        df["PSH"] = df["B365H"]
        df["PSD"] = df["B365D"]
        df["PSA"] = df["B365A"]
        out = add_market_features(df)
        assert out["line_mv_home"].notna().all()
        assert out["line_mv_abs"].notna().all()
        # Opening favourite (lower odds) -> implied prob higher at open, so the
        # closing implied prob is lower -> movement is negative for the favourite
        assert (out["line_mv_home"] < 0).all()
        assert (out["line_mv_abs"] > 0).all()

    def test_market_probs_match_closing_odds(self, matches_with_odds):
        out = add_market_features(matches_with_odds)
        row = out.iloc[0]
        inv = 1 / row["B365H"] + 1 / row["B365D"] + 1 / row["B365A"]
        assert row["market_home"] == pytest.approx(1 / row["B365H"] / inv, abs=1e-9)

    def test_nan_market_when_no_odds(self):
        df = pd.DataFrame(
            {
                "date": ["2024-01-01"],
                "home_team": ["A"],
                "away_team": ["B"],
                "home_goals": [1],
                "away_goals": [0],
                "result": ["H"],
            }
        )
        out = add_market_features(df)
        assert out["market_home"].isna().all()
        assert out["line_mv_abs"].isna().all()
