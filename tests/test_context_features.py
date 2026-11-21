"""Tests for the gated context features (Section C, items 1-7 scaffolding):
optional data/context/<LEAGUE>/<season>.csv files whose whitelisted pre-match
columns join onto matches and activate meta features only when present."""

import pandas as pd
import pytest

from src.context_features import (
    available_context_files,
    context_feature_columns,
    join_context,
    read_context_season,
)
from src.ensemble import FootballEnsemble


def _write_context(tmp_path, df, extra_cols=None, league="EPL", leaky=False):
    folder = tmp_path / league
    folder.mkdir(parents=True, exist_ok=True)
    ctx = df[["date", "home_team", "away_team"]].copy()
    ctx["Date"] = ctx["date"].dt.strftime("%d/%m/%Y")
    ctx = ctx.rename(columns={"home_team": "HomeTeam", "away_team": "AwayTeam"})
    ctx = ctx[["Date", "HomeTeam", "AwayTeam"]]
    for i, (col, val) in enumerate((extra_cols or {}).items()):
        ctx[col] = val
    if leaky:
        ctx["home_goals"] = 9  # realised result — must never become a feature
    ctx.to_csv(folder / "2026-27.csv", index=False)
    return folder


@pytest.fixture
def matches():
    from src.data_loader import generate_synthetic_league

    return generate_synthetic_league(n_teams=6, n_seasons=1)


class TestContextJoin:
    def test_whitelist_columns_join_and_leaky_column_dropped(self, tmp_path, matches):
        df = matches.copy()
        _write_context(
            tmp_path,
            df,
            extra_cols={
                "injuries_home": 1,
                "injuries_away": 0,
                "temp_c": 12,
                "not_a_feature": 7,
            },
            leaky=True,
        )
        joined, added = join_context(df, tmp_path, league="EPL")
        assert "injuries_home" in added and "temp_c" in added
        assert "not_a_feature" not in added  # not on the whitelist
        assert "home_goals" not in added  # realised result dropped
        assert added == ["injuries_home", "injuries_away", "temp_c"]  # registry order
        assert joined.loc[0, "injuries_home"] == 1
        assert "injuries_home" in context_feature_columns()

    def test_no_files_returns_unchanged(self, tmp_path, matches):
        joined, added = join_context(matches, tmp_path, league="EPL")
        assert added == []
        assert "injuries_home" not in joined.columns

    def test_available_files_discovers_seasons(self, tmp_path, matches):
        _write_context(tmp_path, matches, extra_cols={"rain_mm": 0})
        files = available_context_files(tmp_path)
        assert len(files) == 1

    def test_read_context_season_rejects_missing_keys(self, tmp_path, matches):
        p = tmp_path / "bad.csv"
        pd.DataFrame({"foo": [1]}).to_csv(p, index=False)
        df, used = read_context_season(p)
        assert df.empty and used == []


class TestEnsembleIntegration:
    def test_context_features_activate_when_files_exist(self, tmp_path):
        from src.data_loader import generate_synthetic_league

        df = generate_synthetic_league(n_teams=8, n_seasons=2)
        _write_context(
            tmp_path, df, extra_cols={"injuries_home": 2, "injuries_away": 1}
        )
        ens = FootballEnsemble(context_dir=str(tmp_path))
        ens.fit(df, league="EPL")
        assert "injuries_home" in ens.feature_cols
        assert "injuries_away" in ens.feature_cols
        out = ens.predict("Team A", "Team B")
        assert sum([out["home_win"], out["draw"], out["away_win"]]) == pytest.approx(
            1.0, abs=1e-6
        )

    def test_no_context_files_no_context_features(self, tmp_path):
        from src.data_loader import generate_synthetic_league

        df = generate_synthetic_league(n_teams=8, n_seasons=2)
        ens = FootballEnsemble(context_dir=str(tmp_path))
        ens.fit(df, league="EPL")
        assert not any(c.startswith("injuries_") for c in ens.feature_cols)
