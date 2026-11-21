"""Tests for fixtures.csv normalisation + storage (src.update.normalise/store)."""

import sqlite3

import pandas as pd

from src.update import db as updb
from src.update import store as upstore
from src.update.config import load_team_names
from src.update.normalise import normalise_fixtures_csv


def _raw():
    return pd.DataFrame(
        {
            "Div": ["E0", "E0", "SP1", "E2"],
            "Date": ["21/11/2026", "21/11/2026", "22/11/2026", "21/11/2026"],
            "Time": ["15:00", "17:30", "20:00", "15:00"],
            "HomeTeam": ["Chelsea", "Newcastle", "Barcelona", "NotInLeague"],
            "AwayTeam": ["Leeds", "Arsenal", "Villarreal", "AlsoNot"],
            "B365H": [1.80, 3.10, 1.40, 2.00],
            "B365D": [3.60, 3.40, 5.00, 3.50],
            "B365A": [4.50, 2.20, 8.00, 4.00],
            "AvgH": [1.75, 3.00, 1.35, 1.90],
            "AvgD": [3.50, 3.30, 5.20, 3.40],
            "AvgA": [4.60, 2.30, 8.50, 4.10],
        }
    )


def test_filters_and_canonicalises():
    fx, unmapped = normalise_fixtures_csv(_raw(), load_team_names())
    assert set(fx["league"]) == {"EPL", "LALIGA"}  # E2 dropped
    assert unmapped == set()
    chelsea = fx[fx["home"] == "Chelsea"].iloc[0]
    assert chelsea["away"] == "Leeds"
    assert chelsea["date"] == pd.Timestamp("2026-11-21")
    assert chelsea["kickoff_utc"] == pd.Timestamp("2026-11-21 15:00")
    assert chelsea["b365h"] == 1.80
    assert chelsea["avgh"] == 1.75


def test_reports_unmapped_team():
    raw = pd.DataFrame(
        {
            "Div": ["E0"],
            "Date": ["21/11/2026"],
            "Time": ["15:00"],
            "HomeTeam": ["Mystery FC"],
            "AwayTeam": ["Chelsea"],
        }
    )
    fx, unmapped = normalise_fixtures_csv(raw, load_team_names())
    assert "Mystery FC" in unmapped
    assert len(fx) == 1


def _conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    updb.init_db(conn)
    return conn


def test_upsert_fixtures_replaces_idempotently():
    conn = _conn()
    fx, _ = normalise_fixtures_csv(_raw(), load_team_names())
    assert upstore.upsert_fixtures(conn, fx) == 3
    assert upstore.upsert_fixtures(conn, fx) == 3  # replace, not append
    assert conn.execute("SELECT COUNT(*) FROM fixtures").fetchone()[0] == 3


def test_load_fixtures_for_prediction_shape():
    conn = _conn()
    fx, _ = normalise_fixtures_csv(_raw(), load_team_names())
    upstore.upsert_fixtures(conn, fx)
    df = upstore.load_fixtures_for_prediction(conn, league="EPL")
    assert list(df.columns[:4]) == ["date", "home_team", "away_team", "league"]
    assert "B365H" in df.columns and "AvgH" in df.columns
    assert len(df) == 2
    assert set(df["home_team"]) == {"Chelsea", "Newcastle"}
