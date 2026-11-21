"""SQLite schema, connection helpers and the single-run lock.

Schema follows the spec (Section 4) with the project's existing league keys
(EPL / LALIGA / SERIEA) and football-data.co.uk canonical team names.

Tables
------
matches       one row per (league, season, date, home, away); ``match_id`` is a
              hash of those five fields. Raw values are stored untouched.
odds          one row per (match, bookmaker, snapshot/market).
quarantine    rows that failed validation, with a reason (never in ``matches``).
team_ratings  latest Elo rating per (league, team).
meta          key/value bookkeeping (row-count guard, lock).
run_log       one row per pipeline run (status + report path).
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from pathlib import Path

# ``as`` is a SQL keyword, so every identifier is emitted double-quoted below.
MATCH_COLUMNS = [
    "match_id",
    "league",
    "season",
    "date",
    "kickoff_utc",
    "home",
    "away",
    "fthg",
    "ftag",
    "ftr",
    "hthg",
    "htag",
    "htr",
    "hs",
    "as",
    "hst",
    "ast",
    "hc",
    "ac",
    "hf",
    "af",
    "hy",
    "ay",
    "hr",
    "ar",
    "home_xg",
    "away_xg",
    "xg_status",
    "referee",
    "source",
    "ingested_at",
    "updated_at",
]

# Columns upserted on conflict (everything except the identity key match_id).
UPDATABLE_COLUMNS = [c for c in MATCH_COLUMNS if c != "match_id"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    match_id    TEXT PRIMARY KEY,
    league      TEXT NOT NULL,
    season      TEXT NOT NULL,
    date        TEXT NOT NULL,
    kickoff_utc TEXT,
    home        TEXT NOT NULL,
    away        TEXT NOT NULL,
    fthg        INTEGER,
    ftag        INTEGER,
    ftr         TEXT,
    hthg        INTEGER,
    htag        INTEGER,
    htr         TEXT,
    hs          INTEGER,
    "as"        INTEGER,
    hst         INTEGER,
    ast         INTEGER,
    hc          INTEGER,
    ac          INTEGER,
    hf          INTEGER,
    af          INTEGER,
    hy          INTEGER,
    ay          INTEGER,
    hr          INTEGER,
    ar          INTEGER,
    home_xg     REAL,
    away_xg     REAL,
    xg_status   TEXT,
    referee     TEXT,
    source      TEXT,
    ingested_at TEXT,
    updated_at  TEXT
);

CREATE INDEX IF NOT EXISTS idx_matches_league_season ON matches (league, season);
CREATE INDEX IF NOT EXISTS idx_matches_date ON matches (date);

CREATE TABLE IF NOT EXISTS odds (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id       TEXT NOT NULL,
    bookmaker      TEXT NOT NULL,
    market         TEXT NOT NULL,
    home           REAL,
    draw           REAL,
    away           REAL,
    over           REAL,
    under          REAL,
    line           REAL,
    odds_timestamp TEXT,
    is_closing     INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_odds_match ON odds (match_id);

CREATE TABLE IF NOT EXISTS quarantine (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    league      TEXT,
    season      TEXT,
    date        TEXT,
    home        TEXT,
    away        TEXT,
    reason      TEXT NOT NULL,
    raw         TEXT,
    ingested_at TEXT
);

CREATE TABLE IF NOT EXISTS team_ratings (
    league         TEXT NOT NULL,
    team           TEXT NOT NULL,
    rating         REAL NOT NULL,
    matches_played INTEGER,
    updated_at     TEXT,
    PRIMARY KEY (league, team)
);

CREATE TABLE IF NOT EXISTS standings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    league      TEXT NOT NULL,
    season      TEXT NOT NULL,
    as_of       TEXT NOT NULL,
    rank        INTEGER,
    team        TEXT NOT NULL,
    played      INTEGER,
    won         INTEGER,
    draw        INTEGER,
    lost        INTEGER,
    gf          INTEGER,
    ga          INTEGER,
    goal_diff   INTEGER,
    points      INTEGER,
    home_played INTEGER,
    home_won    INTEGER,
    home_draw   INTEGER,
    home_lost   INTEGER,
    home_gf     INTEGER,
    home_ga     INTEGER,
    home_points INTEGER,
    away_played INTEGER,
    away_won    INTEGER,
    away_draw   INTEGER,
    away_lost   INTEGER,
    away_gf     INTEGER,
    away_ga     INTEGER,
    away_points INTEGER,
    form        TEXT
);

CREATE INDEX IF NOT EXISTS idx_standings_league_season ON standings (league, season);

CREATE TABLE IF NOT EXISTS fixtures (
    league      TEXT NOT NULL,
    season      TEXT NOT NULL,
    date        TEXT NOT NULL,
    kickoff_utc TEXT,
    home        TEXT NOT NULL,
    away        TEXT NOT NULL,
    referee     TEXT,
    b365h       REAL,
    b365d       REAL,
    b365a       REAL,
    avgh        REAL,
    avgd        REAL,
    avga        REAL,
    maxh        REAL,
    maxd        REAL,
    maxa        REAL,
    source      TEXT,
    ingested_at TEXT,
    PRIMARY KEY (league, season, date, home, away)
);

CREATE INDEX IF NOT EXISTS idx_fixtures_league_season ON fixtures (league, season);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS run_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT,
    finished_at TEXT,
    status      TEXT,
    report_path TEXT
);
"""


def q(identifier: str) -> str:
    """Double-quote a SQL identifier (so ``as`` works as a column name)."""
    return '"' + identifier + '"'


def make_match_id(league: str, season: str, date_str: str, home: str, away: str) -> str:
    """Hash of the natural key, exactly as the spec defines it."""
    raw = f"{league}|{season}|{date_str}|{home}|{away}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open (creating parents if needed) a SQLite connection with Row rows."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create the schema if it does not already exist."""
    conn.executescript(SCHEMA)
    conn.commit()


class LockError(RuntimeError):
    """Raised when the single-run lock cannot be acquired."""


class RunLock:
    """A filesystem lock allowing one update run at a time.

    Stale locks (crashed runs) are broken after ``stale_after_seconds``.
    """

    def __init__(self, path: str | Path, stale_after_seconds: int = 1800):
        self.path = Path(path)
        self.stale_after_seconds = stale_after_seconds
        self._held = False

    def acquire(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {time.time()}".encode("utf-8"))
            os.close(fd)
            self._held = True
            return self
        except FileExistsError:
            if self._is_stale():
                self.path.unlink(missing_ok=True)
                return self.acquire()
            raise LockError(f"Another update run holds {self.path}") from None

    def _is_stale(self) -> bool:
        try:
            age = time.time() - self.path.stat().st_mtime
            return age > self.stale_after_seconds
        except FileNotFoundError:
            return True

    def release(self) -> None:
        if self._held:
            self.path.unlink(missing_ok=True)
            self._held = False

    def __enter__(self) -> "RunLock":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()
