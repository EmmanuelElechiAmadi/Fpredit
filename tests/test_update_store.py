"""Tests for src.update.store — upsert, correction, odds, row-count guard."""

import sqlite3

import pandas as pd
import pytest

from src.update import db as updb
from src.update import store as upstore


def _conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    updb.init_db(conn)
    return conn


def _row(date, home, away, fthg, ftag, ftr="H"):
    d = pd.Timestamp(date)
    return {
        "league": "EPL",
        "season": "2026-27",
        "date": d,
        "kickoff_utc": d,
        "home": home,
        "away": away,
        "fthg": fthg,
        "ftag": ftag,
        "ftr": ftr,
        "hthg": 0,
        "htag": 0,
        "htr": None,
        "hs": 10,
        "as": 8,
        "hst": 4,
        "ast": 3,
        "hc": 5,
        "ac": 4,
        "hf": 10,
        "af": 12,
        "hy": 1,
        "ay": 2,
        "hr": 0,
        "ar": 0,
        "referee": "Ref",
        "source": "test",
        "_key": f"{d.date().isoformat()}|{home}|{away}",
    }


def _odds(date, home, away):
    d = pd.Timestamp(date)
    return [
        {
            "_key": f"{d.date().isoformat()}|{home}|{away}",
            "bookmaker": "B365",
            "market": "1X2",
            "home": 1.6,
            "draw": 3.9,
            "away": 5.5,
            "is_closing": 1,
        }
    ]


def test_upsert_adds_then_idempotent():
    conn = _conn()
    s1 = upstore.upsert_matches(
        conn,
        pd.DataFrame([_row("2026-08-14", "Arsenal", "Chelsea", 2, 1)]),
        _odds("2026-08-14", "Arsenal", "Chelsea"),
    )
    assert s1["added"] == 1
    assert s1["odds_inserted"] == 1

    s2 = upstore.upsert_matches(
        conn,
        pd.DataFrame([_row("2026-08-14", "Arsenal", "Chelsea", 2, 1)]),
        _odds("2026-08-14", "Arsenal", "Chelsea"),
    )
    assert s2["added"] == 0
    assert s2["unchanged"] == 1
    assert s2["odds_inserted"] == 0

    assert conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM odds").fetchone()[0] == 1


def test_correction_updates_row_and_keeps_ingested_at():
    conn = _conn()
    upstore.upsert_matches(
        conn, pd.DataFrame([_row("2026-08-14", "Arsenal", "Chelsea", 2, 1)]), []
    )
    first = conn.execute("SELECT ingested_at FROM matches").fetchone()["ingested_at"]

    s = upstore.upsert_matches(
        conn, pd.DataFrame([_row("2026-08-14", "Arsenal", "Chelsea", 3, 1)]), []
    )
    assert s["corrected"] == 1
    row = conn.execute("SELECT fthg, ingested_at FROM matches").fetchone()
    assert row["fthg"] == 3
    assert row["ingested_at"] == first  # not overwritten


def test_row_count_never_decreases():
    conn = _conn()
    df = pd.DataFrame(
        [
            _row("2026-08-14", "Arsenal", "Chelsea", 2, 1),
            _row("2026-08-21", "Chelsea", "Arsenal", 1, 1, "D"),
        ]
    )
    upstore.upsert_matches(conn, df, [])
    assert upstore.check_row_count(conn, "EPL", "2026-27") == 2

    conn.execute("DELETE FROM matches WHERE home = 'Arsenal'")
    with pytest.raises(upstore.RowCountError):
        upstore.check_row_count(conn, "EPL", "2026-27")


def test_load_engine_df_translates_columns():
    conn = _conn()
    upstore.upsert_matches(
        conn, pd.DataFrame([_row("2026-08-14", "Arsenal", "Chelsea", 2, 1)]), []
    )
    df = upstore.load_engine_df(conn, league="EPL")
    assert list(df.columns[:6]) == [
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "result",
    ]
    assert df.loc[0, "home_team"] == "Arsenal"
    assert int(df.loc[0, "home_goals"]) == 2


def test_insert_quarantine_writes_rows():
    conn = _conn()
    n = upstore.insert_quarantine(
        conn,
        [
            {
                "league": "EPL",
                "season": "2026-27",
                "date": "2026-08-14",
                "home": "Arsenal",
                "away": "Chelsea",
                "reason": "result disagrees with goals",
                "raw": "{}",
                "ingested_at": "now",
            },
        ],
    )
    assert n == 1
    row = conn.execute("SELECT reason FROM quarantine").fetchone()
    assert row["reason"] == "result disagrees with goals"
