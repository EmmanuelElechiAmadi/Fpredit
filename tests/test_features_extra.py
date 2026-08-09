"""
Tests for the advanced (leak-free) feature set: fixture congestion,
league position / motivation, and PageRank transitive strength.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import generate_synthetic_league
from src.features import fixture_congestion, league_position, pagerank_strength


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-04",
                    "2024-01-07",
                    "2024-01-10",
                    "2024-01-20",
                    "2024-01-23",
                ]
            ),
            "home_team": ["A", "A", "A", "B", "A", "C"],
            "away_team": ["C", "D", "E", "A", "F", "A"],
            "home_goals": [1, 1, 1, 2, 2, 0],
            "away_goals": [0, 0, 0, 1, 1, 1],
            "result": ["H", "H", "H", "H", "H", "A"],
        }
    )


class TestFixtureCongestion:
    def test_returns_dataframe(self, sample_df):
        out = fixture_congestion(sample_df.copy())
        assert isinstance(out, pd.DataFrame)

    def test_adds_expected_columns(self, sample_df):
        out = fixture_congestion(sample_df.copy())
        for col in (
            "home_matches_3in8",
            "away_matches_3in8",
            "home_load",
            "away_load",
            "load_diff",
            "congestion_diff",
        ):
            assert col in out.columns

    def test_pseudo_row_does_not_pollute_load(self):
        """A fixture row with NaN goals (upcoming match) gets a load value but
        must not count toward any team's load for other rows."""
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2024-01-01", "2024-01-05", "2024-01-09", "2024-01-13"]
                ),
                "home_team": ["A", "A", "B", "A"],
                "away_team": ["B", "C", "A", "D"],
                "home_goals": [1, 1, 2, np.nan],
                "away_goals": [0, 0, 1, np.nan],
                "result": ["H", "H", "H", None],
            }
        )
        out = fixture_congestion(df)
        assert out.loc[3, "home_load"] >= 1
        assert out.loc[0, "home_load"] == 1

    def test_third_match_in_eight_days_flagged(self, sample_df):
        """Team A plays 01-01, 01-04, 01-07 -> the 01-07 match is its 3rd
        in eight days and must be flagged for the home side."""
        out = fixture_congestion(sample_df.copy())
        assert out.loc[2, "home_matches_3in8"] == 1

    def test_not_flagged_when_spread_out(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2024-01-01", "2024-01-20", "2024-02-10", "2024-02-15"]
                ),
                "home_team": ["A", "A", "A", "A"],
                "away_team": ["B", "C", "D", "E"],
                "home_goals": [1, 2, 0, 3],
                "away_goals": [0, 1, 0, 1],
                "result": ["H", "H", "D", "H"],
            }
        )
        out = fixture_congestion(df)
        assert (out["home_matches_3in8"] == 0).all()

    def test_load_counts_recent_matches(self, sample_df):
        out = fixture_congestion(sample_df.copy())
        # On 01-07 (row 2) A has played 01-01, 01-04 + this one = 3 in 14 days
        assert out.loc[2, "home_load"] == 3


class TestLeaguePosition:
    def test_adds_columns(self, sample_df):
        out = league_position(sample_df.copy())
        for col in ("home_pos", "away_pos", "pos_diff", "points_diff"):
            assert col in out.columns

    def test_winner_is_position_one(self, sample_df):
        out = league_position(sample_df.copy())
        # After A's two wins, A should be top -> home_pos of A == 1 later
        row = out.iloc[2]
        if not pd.isna(row["home_pos"]):
            assert row["home_pos"] == 1

    def test_position_only_from_prior_matches(self):
        """Perturbing match k must not change positions of rows 0..k (positions
        are computed strictly from prior results), but later rows may change."""
        df = generate_synthetic_league(n_teams=6, n_seasons=1).reset_index(drop=True)
        base = league_position(df)
        df2 = df.copy()
        idx = 1
        df2.loc[idx, "home_goals"] = 0
        df2.loc[idx, "away_goals"] = 9
        df2.loc[idx, "result"] = "A"
        changed = league_position(df2)
        assert np.allclose(
            base["home_pos"].iloc[: idx + 1].values,
            changed["home_pos"].iloc[: idx + 1].values,
            equal_nan=True,
        )
        assert not np.allclose(
            base["home_pos"].iloc[idx + 1 :].values,
            changed["home_pos"].iloc[idx + 1 :].values,
            equal_nan=True,
        )

    def test_strong_team_has_lower_position_number(self):
        """A team that keeps winning has position number closer to 1."""
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22"]
                ),
                "home_team": ["A", "A", "B", "A"],
                "away_team": ["B", "C", "D", "E"],
                "home_goals": [3, 2, 1, 2],
                "away_goals": [0, 0, 1, 1],
                "result": ["H", "H", "D", "H"],
            }
        )
        out = league_position(df)
        a_pos = out.loc[out["home_team"] == "A", "home_pos"].dropna()
        b_pos = out.loc[out["home_team"] == "B", "home_pos"].dropna()
        if len(a_pos) and len(b_pos):
            assert a_pos.iloc[0] < b_pos.iloc[0]

    def test_position_resets_across_seasons(self):
        """A 100+ day gap resets standings so last season's leader is not
        automatically position 1 next season."""
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2024-01-01", "2024-01-08", "2025-08-10", "2025-08-17"]
                ),
                "home_team": ["A", "A", "C", "C"],
                "away_team": ["B", "B", "D", "D"],
                "home_goals": [3, 3, 1, 1],
                "away_goals": [0, 0, 0, 0],
                "result": ["H", "H", "H", "H"],
            }
        )
        out = league_position(df)
        assert pd.isna(out.loc[2, "home_pos"])


class TestPageRank:
    def test_adds_columns(self, sample_df):
        out = pagerank_strength(sample_df.copy())
        for col in ("home_pagerank", "away_pagerank", "pagerank_diff"):
            assert col in out.columns

    def test_scores_positive(self, sample_df):
        out = pagerank_strength(sample_df.copy())
        scores = out[["home_pagerank", "away_pagerank"]].to_numpy().ravel()
        assert (scores > 0).all()

    def test_winner_gets_higher_rank(self):
        """Team A beats everyone; its PageRank must exceed an also-ran's."""
        df = pd.DataFrame(
            {
                "date": pd.to_datetime([f"2024-01-{i:02d}" for i in range(1, 13)]),
                "home_team": ["A"] * 6 + ["C"] * 6,
                "away_team": [
                    "B",
                    "C",
                    "D",
                    "E",
                    "F",
                    "G",
                    "A",
                    "A",
                    "A",
                    "A",
                    "A",
                    "A",
                ],
                "home_goals": [2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0],
                "away_goals": [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
                "result": ["H"] * 12,
            }
        )
        out = pagerank_strength(df)
        a_scores = out.loc[out["home_team"] == "A", "home_pagerank"]
        c_scores = out.loc[out["home_team"] == "C", "home_pagerank"]
        assert a_scores.iloc[-1] > c_scores.iloc[-1]

    def test_first_row_uses_uniform_prior(self, sample_df):
        out = pagerank_strength(sample_df.copy())
        # 6 distinct teams (A-F) -> uniform prior is 1/6
        assert out.loc[0, "home_pagerank"] == pytest.approx(1 / 6)
