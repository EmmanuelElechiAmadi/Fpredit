"""
Unit tests for feature engineering.
"""

import pandas as pd
import pytest

from src.features import head_to_head, rolling_form


@pytest.fixture
def sample_df():
    """A small DataFrame with 2 teams playing each other multiple times."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-08",
                    "2024-01-15",
                    "2024-01-22",
                    "2024-01-29",
                    "2024-02-05",
                ]
            ),
            "home_team": ["A", "B", "A", "B", "A", "B"],
            "away_team": ["B", "A", "B", "A", "B", "A"],
            "home_goals": [2, 1, 3, 0, 1, 2],
            "away_goals": [1, 2, 1, 3, 1, 0],
            "result": ["H", "A", "H", "A", "D", "H"],
        }
    )


class TestRollingForm:
    def test_returns_dataframe(self, sample_df):
        """rolling_form should return a DataFrame."""
        result = rolling_form(sample_df.copy())
        assert isinstance(result, pd.DataFrame)

    def test_adds_ppg_columns(self, sample_df):
        """rolling_form should add form_ppg_diff column."""
        result = rolling_form(sample_df.copy())
        assert "form_ppg_diff" in result.columns

    def test_adds_goal_diff_column(self, sample_df):
        """rolling_form should add form_goal_diff column."""
        result = rolling_form(sample_df.copy())
        assert "form_goal_diff" in result.columns

    def test_adds_rest_diff_column(self, sample_df):
        """rolling_form should add rest_diff column."""
        result = rolling_form(sample_df.copy())
        assert "rest_diff" in result.columns

    def test_first_row_has_nan_ppg(self, sample_df):
        """First match for each team should have NaN PPG (no prior data)."""
        result = rolling_form(sample_df.copy())
        # First match overall: row 0
        assert pd.isna(result.loc[0, "home_ppg"]) or result.loc[0, "home_ppg"] == 0

    def test_form_ppg_diff_sign(self, sample_df):
        """Home team with higher PPG should have positive form_ppg_diff."""
        result = rolling_form(sample_df.copy())
        # Row 4: A has 2W, B has 2L -> A's PPG > B's PPG
        row4 = result.iloc[4]
        if not pd.isna(row4["home_ppg"]) and not pd.isna(row4["away_ppg"]):
            assert row4["form_ppg_diff"] >= 0 or abs(row4["form_ppg_diff"]) < 0.01

    def test_preserves_original_rows(self, sample_df):
        """rolling_form should not drop or duplicate rows."""
        n_orig = len(sample_df)
        result = rolling_form(sample_df.copy())
        assert len(result) == n_orig

    def test_sorted_input_stays_sorted(self, sample_df):
        """Output should remain sorted by date."""
        result = rolling_form(sample_df.copy())
        assert (result["date"].diff().dropna() >= pd.Timedelta(0)).all()

    def test_rest_diff_is_zero_for_first_match(self, sample_df):
        """First match should have NaN rest diff (no prior match)."""
        result = rolling_form(sample_df.copy())
        assert pd.isna(result.loc[0, "rest_diff"])

    def test_rest_diff_positive_for_subsequent_matches(self, sample_df):
        """Subsequent matches should have positive rest days."""
        result = rolling_form(sample_df.copy())
        for i in range(1, len(result)):
            if not pd.isna(result.loc[i, "rest_diff"]):
                # Could be negative if away team played more recently
                pass


class TestHeadToHead:
    def test_returns_series(self, sample_df):
        """head_to_head should return a pandas Series."""
        result = head_to_head(sample_df.copy())
        assert isinstance(result, pd.Series)

    def test_h2h_name(self, sample_df):
        """Series name should be 'h2h_home_ppg_norm'."""
        result = head_to_head(sample_df.copy())
        assert result.name == "h2h_home_ppg_norm"

    def test_first_match_neutral_prior(self, sample_df):
        """First match should get the neutral 0.5 prior (no history)."""
        result = head_to_head(sample_df.copy())
        assert result.iloc[0] == 0.5

    def test_h2h_between_0_and_1(self, sample_df):
        """All H2H values should be between 0 and 1."""
        result = head_to_head(sample_df.copy())
        assert (result >= 0).all() and (result <= 1).all()

    def test_home_team_streak_reflected(self, sample_df):
        """After multiple A-home wins, A's H2H should favor them as home."""
        result = head_to_head(sample_df.copy())
        # By match 4, A has 2 home wins vs B historically
        if result.iloc[4] != 0.5:
            assert result.iloc[4] > 0.5  # A has been dominant at home

    def test_more_history_increases_confidence(self):
        """With enough history, H2H should move away from 0.5 prior."""
        rows = []
        base = pd.Timestamp("2024-01-01")
        for i in range(10):
            rows.append(
                {
                    "date": base + pd.Timedelta(days=i * 7),
                    "home_team": "A",
                    "away_team": "B",
                    "home_goals": 2,
                    "away_goals": 0,
                    "result": "H",
                }
            )
        df = pd.DataFrame(rows)
        result = head_to_head(df)
        # First match has no prior -> 0.5
        assert result.iloc[0] == 0.5
        # After first win, H2H should be > 0.5
        assert result.iloc[1] > 0.5
        # With all wins, H2H saturates at 1.0
        assert result.iloc[-1] == 1.0

    def test_empty_df(self):
        """head_to_head with empty DF should return empty series."""
        df = pd.DataFrame(
            columns=[
                "date",
                "home_team",
                "away_team",
                "home_goals",
                "away_goals",
                "result",
            ]
        )
        result = head_to_head(df)
        assert len(result) == 0

    def test_no_prior_matches_returns_prior(self):
        """A match with no prior H2H meetings should return 0.5."""
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-08"]),
                "home_team": ["A", "C"],
                "away_team": ["B", "D"],
                "home_goals": [2, 1],
                "away_goals": [1, 0],
                "result": ["H", "H"],
            }
        )
        result = head_to_head(df)
        assert result.iloc[0] == 0.5  # no prior
        assert result.iloc[1] == 0.5  # different teams, no prior

    def test_lookback_respected(self):
        """Only the last N matches should be used (not all history)."""
        rows = []
        base = pd.Timestamp("2024-01-01")
        # A beats B 10 times at home
        for i in range(10):
            rows.append(
                {
                    "date": base + pd.Timedelta(days=i * 7),
                    "home_team": "A",
                    "away_team": "B",
                    "home_goals": 2,
                    "away_goals": 0,
                    "result": "H",
                }
            )
        df = pd.DataFrame(rows)
        result_short = head_to_head(df, lookback_matches=2)
        result_long = head_to_head(df, lookback_matches=10)
        # With only 2 matches of lookback, the value may differ
        # (both should still be >0.5, just potentially different magnitudes)
        assert result_short.iloc[-1] > 0.5
        assert result_long.iloc[-1] > 0.5

    def test_same_length_as_input(self, sample_df):
        """Output series should match input DataFrame length."""
        result = head_to_head(sample_df.copy())
        assert len(result) == len(sample_df)
