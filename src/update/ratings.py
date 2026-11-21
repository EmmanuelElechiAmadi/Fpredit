"""Elo ratings — chronological updates with season-rollover handling.

Reuses ``src.elo.EloEngine``. Ratings are recomputed from all stored matches in
chronological order, which makes a rerun over the same data deterministic
(identical ratings), and a corrected historical score is handled by simply
re-running this step from the earliest changed date.

Season rollover (per spec Section 6):
- newly promoted teams start at ``promoted_team_start``,
- every returning team is regressed ``season_regression`` toward the league
  mean at the start of each new season.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pandas as pd

from src.elo import EloEngine


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fit_elo(
    matches: pd.DataFrame,
    elo_k: float = 20.0,
    home_advantage: float = 75.0,
    promoted_start: float = 1450.0,
    season_regression: float = 0.25,
) -> EloEngine:
    """Fit an Elo engine over completed matches in chronological order."""
    engine = EloEngine(k=elo_k, home_advantage=home_advantage)
    df = matches.copy()
    df = df.dropna(subset=["home_goals", "away_goals"]).sort_values("date")

    if "season" not in df.columns:
        df["season"] = ""

    first_season = True
    for season, group in df.groupby("season", sort=False):
        if not first_season:
            _regress(engine, season_regression)
        first_season = False

        teams = set(group["home_team"]) | set(group["away_team"])
        for team in teams:
            if team not in engine.ratings:
                engine.ratings[team] = promoted_start

        for _, m in group.iterrows():
            engine.update(
                m["date"],
                m["home_team"],
                m["away_team"],
                int(m["home_goals"]),
                int(m["away_goals"]),
            )
    return engine


def _regress(engine: EloEngine, fraction: float) -> None:
    if not engine.ratings:
        return
    mean = sum(engine.ratings.values()) / len(engine.ratings)
    for team in list(engine.ratings):
        engine.ratings[team] = mean + (1 - fraction) * (engine.ratings[team] - mean)


def load_ratings(conn: sqlite3.Connection, league: str) -> dict[str, float]:
    rows = conn.execute(
        "SELECT team, rating FROM team_ratings WHERE league = ?", (league,)
    ).fetchall()
    return {r["team"]: r["rating"] for r in rows}


def persist_ratings(
    conn: sqlite3.Connection,
    league: str,
    engine: EloEngine,
) -> list[dict]:
    """Write current Elo ratings and return the largest movers vs the stored ones."""
    prev = load_ratings(conn, league)
    movers = []
    for team, rating in engine.ratings.items():
        old = prev.get(team)
        delta = rating - old if old is not None else 0.0
        movers.append({"team": team, "old": old, "new": rating, "delta": delta})
        played = _matches_played(engine, team)
        conn.execute(
            "INSERT INTO team_ratings (league, team, rating, matches_played, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(league, team) DO UPDATE SET rating = excluded.rating, "
            "matches_played = excluded.matches_played, updated_at = excluded.updated_at",
            (league, team, rating, played, _now()),
        )
    movers.sort(key=lambda m: abs(m["delta"]), reverse=True)
    return movers


def _matches_played(engine: EloEngine, team: str) -> int:
    return sum(1 for _, h, a, _, _ in engine.history if h == team or a == team)
