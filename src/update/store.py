"""Store — upsert matches/odds, write quarantine, guard row counts, merge xG.

Upserts are keyed on ``match_id`` (a hash of league + season + date + home +
away). A changed row is corrected (``updated_at`` bumps, ``ingested_at`` kept);
an unchanged row is a no-op, which makes re-running the update idempotent.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from src.update.db import MATCH_COLUMNS, UPDATABLE_COLUMNS, make_match_id, q

# Columns whose change counts as a "correction" (audit timestamps excluded).
CONTENT_COLUMNS = [
    c for c in UPDATABLE_COLUMNS if c not in ("ingested_at", "updated_at")
]


class RowCountError(RuntimeError):
    """A season's row count decreased between runs — the update is aborted."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm(value):
    """Normalise a value for storage / comparison (NaT/NaN/NA -> None)."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _date_str(row) -> str:
    d = row.get("date")
    if d is None or (hasattr(d, "date") and pd.isna(d)):
        return ""
    return d.date().isoformat() if hasattr(d, "date") else str(d)


def _row_values(row, match_id: str, ingested_at: str) -> dict:
    """Build the DB column -> value mapping for one validated row."""
    vals = {
        "match_id": match_id,
        "league": row.get("league"),
        "season": row.get("season"),
        "date": _date_str(row),
        "kickoff_utc": _norm(row.get("kickoff_utc")),
        "home": row.get("home"),
        "away": row.get("away"),
        "fthg": _norm(row.get("fthg")),
        "ftag": _norm(row.get("ftag")),
        "ftr": _norm(row.get("ftr")),
        "hthg": _norm(row.get("hthg")),
        "htag": _norm(row.get("htag")),
        "htr": _norm(row.get("htr")),
        "hs": _norm(row.get("hs")),
        "as": _norm(row.get("as")),
        "hst": _norm(row.get("hst")),
        "ast": _norm(row.get("ast")),
        "hc": _norm(row.get("hc")),
        "ac": _norm(row.get("ac")),
        "hf": _norm(row.get("hf")),
        "af": _norm(row.get("af")),
        "hy": _norm(row.get("hy")),
        "ay": _norm(row.get("ay")),
        "hr": _norm(row.get("hr")),
        "ar": _norm(row.get("ar")),
        "home_xg": None,
        "away_xg": None,
        "xg_status": "pending",
        "referee": _norm(row.get("referee")),
        "source": _norm(row.get("source")),
        "ingested_at": ingested_at,
        "updated_at": ingested_at,
    }
    return {c: vals[c] for c in MATCH_COLUMNS}


def _content_changed(old: dict, new: dict) -> bool:
    for c in CONTENT_COLUMNS:
        if _cmp(old.get(c)) != _cmp(new.get(c)):
            return True
    return False


def _cmp(v):
    if v is None:
        return None
    if isinstance(v, float) and v != v:  # NaN
        return None
    return v


def upsert_matches(
    conn: sqlite3.Connection,
    valid_df: pd.DataFrame,
    odds_records: list[dict],
    ingested_at: Optional[str] = None,
) -> dict:
    """Upsert valid rows and their odds. Returns a summary dict."""
    ingested_at = ingested_at or _now()
    odds_by_key: dict[str, list[dict]] = {}
    for rec in odds_records:
        odds_by_key.setdefault(rec.pop("_key", None), []).append(rec)

    league = valid_df.iloc[0]["league"] if len(valid_df) else None
    season = valid_df.iloc[0]["season"] if len(valid_df) else None

    existing: dict[str, dict] = {}
    if league and season:
        cur = conn.execute(
            "SELECT * FROM matches WHERE league = ? AND season = ?", (league, season)
        )
        for r in cur.fetchall():
            existing[r["match_id"]] = {c: r[c] for c in MATCH_COLUMNS}

    added, corrected, unchanged = 0, 0, 0
    changed_dates: list[str] = []
    odds_inserted = 0

    insert_cols = ", ".join(q(c) for c in MATCH_COLUMNS)
    insert_sql = f"INSERT INTO matches ({insert_cols}) VALUES ({','.join('?' * len(MATCH_COLUMNS))})"
    update_sql = (
        "UPDATE matches SET "
        + ", ".join(f"{q(c)} = ?" for c in UPDATABLE_COLUMNS)
        + " WHERE match_id = ?"
    )

    for _, row in valid_df.iterrows():
        date_str = _date_str(row)
        match_id = make_match_id(
            row["league"], row["season"], date_str, row["home"], row["away"]
        )
        values = _row_values(row, match_id, ingested_at)
        changed = False

        if match_id not in existing:
            conn.execute(insert_sql, [values[c] for c in MATCH_COLUMNS])
            added += 1
            changed = True
            if date_str:
                changed_dates.append(date_str)
        else:
            old = existing[match_id]
            if _content_changed(old, values):
                new_vals = dict(values)
                new_vals["ingested_at"] = old["ingested_at"]
                new_vals["updated_at"] = ingested_at
                conn.execute(
                    update_sql, [new_vals[c] for c in UPDATABLE_COLUMNS] + [match_id]
                )
                corrected += 1
                changed = True
                if date_str:
                    changed_dates.append(date_str)
            else:
                unchanged += 1

        if changed:
            for rec in odds_by_key.get(row.get("_key"), []):
                conn.execute(
                    "INSERT INTO odds (match_id, bookmaker, market, home, draw, away, over, under, line, odds_timestamp, is_closing) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        match_id,
                        rec.get("bookmaker"),
                        rec.get("market"),
                        rec.get("home"),
                        rec.get("draw"),
                        rec.get("away"),
                        rec.get("over"),
                        rec.get("under"),
                        rec.get("line"),
                        ingested_at,
                        rec.get("is_closing", 0),
                    ),
                )
                odds_inserted += 1

    return {
        "league": league,
        "season": season,
        "added": added,
        "corrected": corrected,
        "unchanged": unchanged,
        "odds_inserted": odds_inserted,
        "earliest_changed": min(changed_dates) if changed_dates else None,
    }


def insert_quarantine(conn: sqlite3.Connection, quarantined: list[dict]) -> int:
    for rec in quarantined:
        conn.execute(
            "INSERT INTO quarantine (league, season, date, home, away, reason, raw, ingested_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rec.get("league"),
                rec.get("season"),
                rec.get("date"),
                rec.get("home"),
                rec.get("away"),
                rec.get("reason"),
                rec.get("raw"),
                rec.get("ingested_at"),
            ),
        )
    return len(quarantined)


def check_row_count(conn: sqlite3.Connection, league: str, season: str) -> int:
    """Enforce the "row count never decreases" rule. Returns the new count."""
    count = conn.execute(
        "SELECT COUNT(*) AS n FROM matches WHERE league = ? AND season = ?",
        (league, season),
    ).fetchone()["n"]
    key = f"count:{league}:{season}"
    prev = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    if prev is not None and count < int(prev["value"]):
        raise RowCountError(
            f"{league} {season}: row count decreased from {prev['value']} to {count}"
        )
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, str(count)),
    )
    return count


def load_matches(
    conn: sqlite3.Connection,
    league: Optional[str] = None,
    season: Optional[str] = None,
) -> pd.DataFrame:
    """Load stored matches in DB schema."""
    sql = "SELECT * FROM matches"
    clauses: list[str] = []
    params: list = []
    if league is not None:
        clauses.append("league = ?")
        params.append(league)
    if season is not None:
        clauses.append("season = ?")
        params.append(season)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY date, kickoff_utc"
    return pd.read_sql_query(sql, conn, params=params)


def load_engine_df(
    conn: sqlite3.Connection,
    league: Optional[str] = None,
    season: Optional[str] = None,
) -> pd.DataFrame:
    """Load stored matches translated into the src/* engine schema."""
    df = load_matches(conn, league=league, season=season)
    columns = [
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "result",
        "league",
        "season",
        "match_id",
        "home_xg",
        "away_xg",
        "xg_status",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["kickoff_utc"], errors="coerce").fillna(
        pd.to_datetime(df["date"], errors="coerce")
    )
    out["home_team"] = df["home"]
    out["away_team"] = df["away"]
    out["home_goals"] = df["fthg"]
    out["away_goals"] = df["ftag"]
    out["result"] = df["ftr"]
    out["league"] = df["league"]
    out["season"] = df["season"]
    out["match_id"] = df["match_id"]
    out["home_xg"] = df["home_xg"]
    out["away_xg"] = df["away_xg"]
    out["xg_status"] = df["xg_status"]
    return out[columns]


def add_odds_columns(conn: sqlite3.Connection, df: pd.DataFrame) -> pd.DataFrame:
    """Pivot stored closing 1X2 odds onto an engine DataFrame (B365/BbAv)."""
    if df.empty:
        return df
    odds = pd.read_sql_query(
        "SELECT match_id, bookmaker, home, draw, away FROM odds "
        "WHERE market = '1X2' AND is_closing = 1",
        conn,
    )
    if odds.empty:
        return df
    for book in ("B365", "BbAv"):
        sub = odds[odds["bookmaker"] == book][["match_id", "home", "draw", "away"]]
        if sub.empty:
            continue
        sub = sub.drop_duplicates("match_id", keep="first").rename(
            columns={"home": f"{book}H", "draw": f"{book}D", "away": f"{book}A"}
        )
        df = df.merge(sub, on="match_id", how="left")
    return df


def update_xg(
    conn: sqlite3.Connection,
    match_id: str,
    home_xg: Optional[float],
    away_xg: Optional[float],
) -> None:
    """Write xG for a match and flip its status to complete."""
    conn.execute(
        "UPDATE matches SET home_xg = ?, away_xg = ?, xg_status = ? WHERE match_id = ?",
        (home_xg, away_xg, "complete", match_id),
    )


def _date_only(v) -> Optional[str]:
    v = _norm(v)
    if v is None:
        return None
    return v[:10]


_FIXTURE_INSERTSQL = (
    "INSERT INTO fixtures (league, season, date, kickoff_utc, home, away, referee, "
    "b365h, b365d, b365a, avgh, avgd, avga, maxh, maxd, maxa, source, ingested_at) "
    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
)


def upsert_fixtures(
    conn: sqlite3.Connection,
    fixtures_df: pd.DataFrame,
    ingested_at: Optional[str] = None,
) -> int:
    """Replace stored fixtures for each league+season with the fresh set."""
    ingested_at = ingested_at or _now()
    if fixtures_df is None or fixtures_df.empty:
        return 0
    total = 0
    for (league, season), grp in fixtures_df.groupby(["league", "season"]):
        conn.execute(
            "DELETE FROM fixtures WHERE league = ? AND season = ?", (league, season)
        )
        for _, r in grp.iterrows():
            conn.execute(
                _FIXTURE_INSERTSQL,
                (
                    league,
                    season,
                    _date_only(r.get("date")),
                    _norm(r.get("kickoff_utc")),
                    r.get("home"),
                    r.get("away"),
                    _norm(r.get("referee")),
                    _norm(r.get("b365h")),
                    _norm(r.get("b365d")),
                    _norm(r.get("b365a")),
                    _norm(r.get("avgh")),
                    _norm(r.get("avgd")),
                    _norm(r.get("avga")),
                    _norm(r.get("maxh")),
                    _norm(r.get("maxd")),
                    _norm(r.get("maxa")),
                    _norm(r.get("source")),
                    ingested_at,
                ),
            )
            total += 1
    return total


def load_fixtures_for_prediction(
    conn: sqlite3.Connection,
    league: Optional[str] = None,
    season: Optional[str] = None,
) -> pd.DataFrame:
    """Load stored fixtures in the shape ``src.update.predict`` expects, with
    pre-match B365/Avg 1X2 odds attached for market-edge comparison."""
    sql = (
        "SELECT league, season, date, kickoff_utc, home, away, "
        "b365h, b365d, b365a, avgh, avgd, avga, maxh, maxd, maxa FROM fixtures"
    )
    clauses, params = [], []
    if league is not None:
        clauses.append("league = ?")
        params.append(league)
    if season is not None:
        clauses.append("season = ?")
        params.append(season)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY league, kickoff_utc"
    df = pd.read_sql_query(sql, conn, params=params)

    columns = ["date", "home_team", "away_team", "league", "B365H", "B365D", "B365A"]
    if df.empty:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["kickoff_utc"], errors="coerce").fillna(
        pd.to_datetime(df["date"], errors="coerce")
    )
    out["home_team"] = df["home"]
    out["away_team"] = df["away"]
    out["league"] = df["league"]
    out["B365H"], out["B365D"], out["B365A"] = df["b365h"], df["b365d"], df["b365a"]
    out["AvgH"], out["AvgD"], out["AvgA"] = df["avgh"], df["avgd"], df["avga"]
    return out[columns + ["AvgH", "AvgD", "AvgA"]]
