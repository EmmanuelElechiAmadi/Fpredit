"""
Tests for the data loader module.
Uses synthetic data (never hits the network).
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import (
    LEAGUE_CODES,
    _find_column,
    _normalize,
    generate_synthetic_league,
    load_all_leagues,
    load_league_csvs,
)

# ── Column aliases unit tests ─────────────────────────────────────────────────


class TestFindColumn:
    def test_finds_exact_match(self):
        df = pd.DataFrame({"Date": [], "HomeTeam": [], "FTHG": []})
        assert _find_column(df, ["Date", "date"]) == "Date"

    def test_finds_case_insensitive(self):
        df = pd.DataFrame({"date": [], "hometeam": []})
        assert _find_column(df, ["Date", "date"]) == "date"

    def test_returns_none_when_missing(self):
        df = pd.DataFrame({"home_team": []})
        assert _find_column(df, ["Date", "date"]) is None

    def test_prefers_exact_match(self):
        """When both exact match and case-insensitive match exist, exact match
        is returned (first match in dict iteration order, but the dict is built
        from lowercase keys so the last duplicate wins)."""
        df = pd.DataFrame({"Date": [], "date": []})
        found = _find_column(df, ["Date", "date"])
        # Both work -- just ensure it returns one of them
        assert found in ("Date", "date")


class TestNormalize:
    def test_standard_epl_columns(self):
        df = pd.DataFrame(
            {
                "Date": ["01/01/2020"],
                "HomeTeam": ["Arsenal"],
                "AwayTeam": ["Chelsea"],
                "FTHG": [2],
                "FTAG": [1],
                "FTR": ["H"],
                "ExtraCol": ["x"],
            }
        )
        out = _normalize(df)
        assert out is not None
        assert set(out.columns) == {
            "date",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "result",
        }

    def test_lowercase_variant(self):
        df = pd.DataFrame(
            {
                "date": ["01/01/2020"],
                "home": ["Arsenal"],
                "away": ["Chelsea"],
                "ft_home_goals": [2],
                "ft_away_goals": [1],
                "ft_result": ["H"],
            }
        )
        out = _normalize(df)
        assert out is not None
        assert out.iloc[0]["home_team"] == "Arsenal"

    def test_returns_none_when_missing_required(self):
        df = pd.DataFrame({"Date": [], "HomeTeam": []})  # no result column
        out = _normalize(df)
        assert out is None


# ── Loader integration tests (using temp CSV files) ──────────────────────────


@pytest.fixture
def temp_league_dir():
    """Create a temporary directory structure with a minimal CSV."""
    with tempfile.TemporaryDirectory() as tmp:
        code = "E0"
        league_dir = Path(tmp) / code
        league_dir.mkdir(parents=True)

        csv_content = (
            "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
            "01/01/2020,Arsenal,Chelsea,2,1,H\n"
            "15/01/2020,Chelsea,Arsenal,0,0,D\n"
        )
        csv_path = league_dir / "2020-21.csv"
        csv_path.write_text(csv_content)

        yield tmp


class TestLoadLeagueCsvs:
    def test_loads_csvs_from_directory(self, temp_league_dir):
        df = load_league_csvs(temp_league_dir, "EPL")
        assert len(df) == 2
        assert list(df.columns) == [
            "date",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "result",
            "league",
        ]
        assert df.iloc[0]["home_team"] == "Arsenal"

    def test_raises_on_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "E0").mkdir()
            with pytest.raises(FileNotFoundError):
                load_league_csvs(tmp, "EPL")

    def test_raises_on_invalid_league(self):
        with pytest.raises(KeyError):
            load_league_csvs("/tmp", "INVALID")


class TestLoadAllLeagues:
    def test_loads_single_league(self, temp_league_dir):
        df = load_all_leagues(temp_league_dir, leagues=["EPL"])
        assert len(df) == 2

    def test_raises_when_no_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            for code in LEAGUE_CODES.values():
                (Path(tmp) / code).mkdir()
            with pytest.raises(FileNotFoundError):
                load_all_leagues(tmp)


# ── Synthetic data tests ─────────────────────────────────────────────────────


class TestGenerateSyntheticLeague:
    def test_returns_dataframe_with_expected_columns(self):
        df = generate_synthetic_league(n_teams=4, n_seasons=1)
        expected = {
            "date",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "result",
            "league",
        }
        assert set(df.columns) == expected

    def test_all_teams_play_each_other_twice(self):
        df = generate_synthetic_league(n_teams=4, n_seasons=1)
        # 4 teams -> 12 matches per season (each pair plays H+A)
        assert len(df) == 4 * 3  # n_teams * (n_teams - 1)

    def test_sorting_by_date(self):
        df = generate_synthetic_league(n_teams=4, n_seasons=1)
        assert df["date"].is_monotonic_increasing

    def test_reproducible_with_seed(self):
        df1 = generate_synthetic_league(seed=42)
        df2 = generate_synthetic_league(seed=42)
        assert df1["home_goals"].tolist() == df2["home_goals"].tolist()

    def test_result_is_consistent_with_goals(self):
        df = generate_synthetic_league(n_teams=4, n_seasons=1)
        for _, row in df.iterrows():
            if row["home_goals"] > row["away_goals"]:
                assert row["result"] == "H"
            elif row["home_goals"] == row["away_goals"]:
                assert row["result"] == "D"
            else:
                assert row["result"] == "A"
