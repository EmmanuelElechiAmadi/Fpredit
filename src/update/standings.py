"""Standings + form — computed from stored results (never scraped).

Derives a full league table from the ``matches`` rows: played, won/drawn/lost,
goals for/against, goal difference, points, home/away splits, and last-N form.
Ranks use competition ranking — clubs level on (points, goal_diff, goals_for)
share a rank.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

STANDING_FIELDS = [
    "played",
    "won",
    "draw",
    "lost",
    "gf",
    "ga",
    "points",
    "home_played",
    "home_won",
    "home_draw",
    "home_lost",
    "home_gf",
    "home_ga",
    "home_points",
    "away_played",
    "away_won",
    "away_draw",
    "away_lost",
    "away_gf",
    "away_ga",
    "away_points",
]


def _empty(team: str) -> dict:
    row: dict = {"team": team, "form": ""}
    for f in STANDING_FIELDS:
        row[f] = 0
    return row


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_standings(
    matches_df: pd.DataFrame,
    teams: Optional[list[str]] = None,
    form_n: int = 5,
) -> list[dict]:
    """Return a league table (list of rows, best first) from completed matches.

    ``matches_df`` uses the engine schema (date, home_team, away_team,
    home_goals, away_goals, result). ``teams`` optionally pre-seeds clubs with
    no matches yet (0 played) so a new season shows its full roster.
    """
    table: dict[str, dict] = {}
    for t in teams or []:
        table.setdefault(t, _empty(t))

    for _, r in matches_df.iterrows():
        if pd.isna(r.get("home_goals")) or pd.isna(r.get("away_goals")):
            continue
        home, away = r["home_team"], r["away_team"]
        hg, ag = int(r["home_goals"]), int(r["away_goals"])
        h = table.setdefault(home, _empty(home))
        a = table.setdefault(away, _empty(away))

        # Overall.
        h["played"] += 1
        a["played"] += 1
        h["gf"] += hg
        h["ga"] += ag
        a["gf"] += ag
        a["ga"] += hg

        # Home/away splits.
        h["home_played"] += 1
        h["home_gf"] += hg
        h["home_ga"] += ag
        a["away_played"] += 1
        a["away_gf"] += ag
        a["away_ga"] += hg

        if hg > ag:
            _win(h, "home")
            _loss(a, "away")
        elif hg == ag:
            _draw(h, "home")
            _draw(a, "away")
        else:
            _win(a, "away")
            _loss(h, "home")

    rows = []
    for standing in table.values():
        standing["goal_diff"] = standing["gf"] - standing["ga"]
        rows.append(standing)

    rows.sort(key=lambda x: (-x["points"], -x["goal_diff"], -x["gf"], x["team"]))

    # Competition ranking: equal (points, GD, GF) share a rank.
    prev_key, prev_rank = None, 0
    for i, r in enumerate(rows, 1):
        key = (r["points"], r["goal_diff"], r["gf"])
        if key != prev_key:
            prev_key, prev_rank = key, i
        r["rank"] = prev_rank

    _attach_form(matches_df, rows, form_n)
    return rows


def _win(row: dict, venue: str) -> None:
    row["won"] += 1
    row["points"] += 3
    row[f"{venue}_won"] += 1
    row[f"{venue}_points"] += 3


def _draw(row: dict, venue: str) -> None:
    row["draw"] += 1
    row["points"] += 1
    row[f"{venue}_draw"] += 1
    row[f"{venue}_points"] += 1


def _loss(row: dict, venue: str) -> None:
    row["lost"] += 1
    row[f"{venue}_lost"] += 1


def _attach_form(matches_df: pd.DataFrame, rows: list[dict], n: int) -> None:
    """W/D/L string for each team's last ``n`` matches (most recent last)."""
    df = matches_df.dropna(subset=["home_goals", "away_goals"]).sort_values("date")
    for row in rows:
        team = row["team"]
        sub = df[(df["home_team"] == team) | (df["away_team"] == team)].tail(n)
        chars = []
        for _, r in sub.iterrows():
            if r["home_team"] == team:
                chars.append(
                    "W"
                    if int(r["home_goals"]) > int(r["away_goals"])
                    else ("D" if int(r["home_goals"]) == int(r["away_goals"]) else "L")
                )
            else:
                chars.append(
                    "W"
                    if int(r["away_goals"]) > int(r["home_goals"])
                    else ("D" if int(r["away_goals"]) == int(r["home_goals"]) else "L")
                )
        row["form"] = "".join(chars)


def write_standings(
    conn: sqlite3.Connection,
    league: str,
    season: str,
    rows: list[dict],
    as_of: Optional[str] = None,
) -> int:
    """Replace the stored table for (league, season) with freshly computed rows."""
    as_of = as_of or _now()
    conn.execute(
        "DELETE FROM standings WHERE league = ? AND season = ?", (league, season)
    )
    cols = (
        ["league", "season", "as_of", "rank", "team"]
        + STANDING_FIELDS
        + ["goal_diff", "form"]
    )
    sql = f"INSERT INTO standings ({', '.join(cols)}) VALUES ({','.join('?' * len(cols))})"
    for r in rows:
        conn.execute(
            sql,
            tuple(
                [league, season, as_of, r.get("rank"), r["team"]]
                + [r[f] for f in STANDING_FIELDS]
                + [r["goal_diff"], r["form"]]
            ),
        )
    return len(rows)
