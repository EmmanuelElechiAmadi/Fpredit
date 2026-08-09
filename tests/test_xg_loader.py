"""
Tests for the xG loader: Understat JSON parsing, name normalization, caching,
joining onto results frames, and the synthetic generator. Network-dependent
scraping is NOT exercised here (tests stay offline-safe).
"""

import json

import pandas as pd
import pytest

from src.data_loader import generate_synthetic_league
from src.xg_loader import (
    _parse_matches_json,
    fetch_league_xg,
    generate_synthetic_xg,
    join_xg,
    load_league_xg,
)


def _sample_matches_json():
    return json.dumps(
        [
            {
                "id": "1",
                "isResult": True,
                "datetime": "2023-08-11 20:00:00Z",
                "h": {"id": "1", "title": "Manchester City"},
                "a": {"id": "2", "title": "Arsenal"},
                "goals": {"h": "2", "a": "1"},
                "xG": {"h": "2.3", "a": "1.1"},
            },
            {
                "id": "2",
                "isResult": True,
                "datetime": "2023-08-12 17:30:00Z",
                "h": {"id": "3", "title": "Chelsea"},
                "a": {"id": "4", "title": "Tottenham"},
                "goals": {"h": "1", "a": "1"},
                "xG": {"h": "1.4", "a": "1.6"},
            },
        ]
    )


class TestParseMatchesJson:
    def test_parses_basic_structure(self):
        df = _parse_matches_json(_sample_matches_json())
        assert len(df) == 2
        assert {"home_goals", "away_goals", "home_xg", "away_xg", "result"}.issubset(
            df.columns
        )

    def test_normalizes_team_names(self):
        df = _parse_matches_json(_sample_matches_json())
        # Manchester City -> Man City, Tottenham -> Spurs (football-data.co.uk)
        assert df.loc[0, "home_team"] == "Man City"
        assert df.loc[1, "away_team"] == "Spurs"
        assert df.loc[0, "away_team"] == "Arsenal"

    def test_xg_values_float(self):
        df = _parse_matches_json(_sample_matches_json())
        assert df.loc[0, "home_xg"] == pytest.approx(2.3)
        assert df.loc[1, "away_xg"] == pytest.approx(1.6)

    def test_results_correct(self):
        df = _parse_matches_json(_sample_matches_json())
        assert df.loc[0, "result"] == "H"
        assert df.loc[1, "result"] == "D"


class TestJoinXG:
    def test_merges_xg_columns(self):
        results = generate_synthetic_league(n_teams=6, n_seasons=1).reset_index(
            drop=True
        )
        xg = generate_synthetic_xg(results)
        joined = join_xg(results, xg)
        assert "home_xg" in joined.columns
        assert joined["home_xg"].notna().all()

    def test_missing_matches_keep_nan(self):
        results = generate_synthetic_league(n_teams=6, n_seasons=1).reset_index(
            drop=True
        )
        xg = generate_synthetic_xg(results)
        # Drop the last row from the xG set -> that match has no xG
        xg_trimmed = xg.iloc[:-1]
        joined = join_xg(results, xg_trimmed)
        assert joined["home_xg"].isna().iloc[-1]
        assert joined["home_xg"].notna().iloc[:-1].all()

    def test_returns_original_when_xg_empty(self):
        results = generate_synthetic_league(n_teams=6, n_seasons=1).reset_index(
            drop=True
        )
        joined = join_xg(results, pd.DataFrame())
        assert "home_xg" not in joined.columns


class TestSyntheticXG:
    def test_generates_columns(self):
        results = generate_synthetic_league(n_teams=6, n_seasons=1).reset_index(
            drop=True
        )
        out = generate_synthetic_xg(results)
        assert "home_xg" in out.columns and "away_xg" in out.columns
        assert out["home_xg"].between(0.05, 6.0).all()
        assert out["home_xg"].notna().all()

    def test_deterministic(self):
        results = generate_synthetic_league(n_teams=6, n_seasons=1).reset_index(
            drop=True
        )
        a = generate_synthetic_xg(results, seed=3)
        b = generate_synthetic_xg(results, seed=3)
        assert a["home_xg"].equals(b["home_xg"])


class TestCaching:
    def test_fetch_uses_cache_without_network(self, tmp_path):
        """If cached CSVs exist, fetch_league_xg must not hit the network."""
        results = generate_synthetic_league(n_teams=6, n_seasons=1).reset_index(
            drop=True
        )
        xg = generate_synthetic_xg(results)
        out_dir = tmp_path / "EPL"
        out_dir.mkdir(parents=True)
        xg.to_csv(out_dir / "2023-24.csv", index=False)

        # _scrape_season would fail without network; we monkeypatch it to
        # prove the cache path is taken.
        import src.xg_loader as xg_loader

        def _boom(league, season, timeout=30):
            raise AssertionError("network hit!")

        xg_loader._scrape_season = _boom
        df = fetch_league_xg("EPL", seasons=["2023-24"], data_dir=str(tmp_path))
        assert len(df) == len(results)
        assert df["home_xg"].notna().all()

    def test_load_league_xg_roundtrip(self, tmp_path):
        results = generate_synthetic_league(n_teams=6, n_seasons=1).reset_index(
            drop=True
        )
        xg = generate_synthetic_xg(results)
        out_dir = tmp_path / "EPL"
        out_dir.mkdir(parents=True)
        xg.to_csv(out_dir / "2023-24.csv", index=False)
        loaded = load_league_xg(str(tmp_path), "EPL")
        assert len(loaded) == len(xg)

    def test_load_league_xg_missing_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_league_xg(str(tmp_path), "EPL")
