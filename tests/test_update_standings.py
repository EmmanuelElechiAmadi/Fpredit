"""Tests for src.update.standings — league table computed from stored results."""

import sqlite3

import pandas as pd

from src.update import db as updb
from src.update.standings import compute_standings, write_standings


def _df():
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-08-14"),
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "home_goals": 2,
                "away_goals": 1,
                "result": "H",
                "season": "2026-27",
            },
            {
                "date": pd.Timestamp("2026-08-21"),
                "home_team": "Chelsea",
                "away_team": "Arsenal",
                "home_goals": 1,
                "away_goals": 1,
                "result": "D",
                "season": "2026-27",
            },
            {
                "date": pd.Timestamp("2026-08-28"),
                "home_team": "Arsenal",
                "away_team": "Coventry",
                "home_goals": 3,
                "away_goals": 0,
                "result": "H",
                "season": "2026-27",
            },
            {
                "date": pd.Timestamp("2026-09-04"),
                "home_team": "Coventry",
                "away_team": "Chelsea",
                "home_goals": 0,
                "away_goals": 2,
                "result": "A",
                "season": "2026-27",
            },
        ]
    )


def test_compute_standings_points_gd_gf_and_rank():
    rows = compute_standings(_df())
    by_team = {r["team"]: r for r in rows}
    assert by_team["Arsenal"]["played"] == 3
    assert by_team["Arsenal"]["won"] == 2
    assert by_team["Arsenal"]["draw"] == 1
    assert by_team["Arsenal"]["points"] == 7
    assert by_team["Arsenal"]["gf"] == 6
    assert by_team["Arsenal"]["ga"] == 2
    assert by_team["Arsenal"]["goal_diff"] == 4

    assert by_team["Chelsea"]["played"] == 3
    assert by_team["Chelsea"]["points"] == 4
    assert by_team["Chelsea"]["gf"] == 4
    assert by_team["Chelsea"]["ga"] == 3

    assert by_team["Coventry"]["played"] == 2
    assert by_team["Coventry"]["points"] == 0
    assert by_team["Coventry"]["goal_diff"] == -5

    # Sorted best-first, competition ranking.
    assert rows[0]["team"] == "Arsenal" and rows[0]["rank"] == 1
    assert rows[1]["team"] == "Chelsea" and rows[1]["rank"] == 2
    assert rows[2]["team"] == "Coventry" and rows[2]["rank"] == 3


def test_home_away_splits_and_form():
    rows = compute_standings(_df())
    by_team = {r["team"]: r for r in rows}
    arsenal = by_team["Arsenal"]
    assert arsenal["home_played"] == 2
    assert arsenal["home_won"] == 2
    assert arsenal["home_gf"] == 5
    assert arsenal["home_ga"] == 1
    assert arsenal["away_played"] == 1
    assert arsenal["away_draw"] == 1
    assert arsenal["form"] == "WDW"  # most recent last


def test_teams_are_pre_seeded_with_zero_matches():
    rows = compute_standings(_df(), teams=["Arsenal", "Chelsea", "Coventry", "Everton"])
    by_team = {r["team"]: r for r in rows}
    assert by_team["Everton"]["played"] == 0
    assert by_team["Everton"]["points"] == 0


def test_write_standings_is_idempotent():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    updb.init_db(conn)
    rows = compute_standings(_df())
    assert write_standings(conn, "EPL", "2026-27", rows) == 3
    assert write_standings(conn, "EPL", "2026-27", rows) == 3  # replaced, not appended
    assert conn.execute("SELECT COUNT(*) FROM standings").fetchone()[0] == 3
