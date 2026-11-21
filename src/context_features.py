"""Gated, pluggable pre-match context features (Section C, items 1–7).

Many professional features (injuries, manager changes, UEFA midweek load,
weather, referee tendencies, attendance, lower-division priors, richer xG
families, opening->closing line movement) are missing purely because the free
pipeline carries no data for them. Rather than hard-code dead features, this
module provides a **gated join**: when someone supplies the data — as per-match
context CSVs in ``data/context/<LEAGUE>/<season>.csv`` — the columns below are
attached to the match frame keyed by ``(date, home_team, away_team)`` and the
ensemble picks them up automatically. Without the files, nothing changes.

LEAK-SAFETY: this table is for *pre-match* context only. Realised in-match
statistics (npxG, cards, xG for/against of the match itself) are NOT accepted
here — those belong to the xG training pipeline, which already uses them only
through leak-free filtered estimates. Columns that are not on the whitelist
are silently dropped, so a pasted results column can never leak into the
meta-learner.

Supported per-match columns (both home and away sides where sensible):

* ``injuries_home`` / ``injuries_away``          — unavailable players (count)
* ``manager_tenure_days_home`` / ``_away``       — coach tenure at kickoff
* ``new_manager_home`` / ``new_manager_away``    — 1 if appointed <= 14 days
* ``uefa_rest_days_home`` / ``_away``            — days since the side's last
                                                  UEFA match
* ``uefa_load_7d_home`` / ``_away``              — UEFA matches in the last 7d
* ``attendance``                                 — (approx) crowd
* ``temp_c``, ``wind_kmh``, ``rain_mm``          — kickoff weather forecast
* ``referee_yellow_avg``                         — referee's avg yellows/match
* ``promoted_prior_rating_home`` / ``_away``     — seeding from the division
                                                  below (see README)

The context CSVs use the same ``Date,HomeTeam,AwayTeam`` key columns as every
other file in the repo (dates dd/mm/yyyy or ISO), plus any of the columns
above; unknown columns are ignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

_CONTEXT_FEATURE_COLUMNS = [
    # Injuries / availability (C.1)
    "injuries_home",
    "injuries_away",
    # Manager effects (C.2)
    "manager_tenure_days_home",
    "manager_tenure_days_away",
    "new_manager_home",
    "new_manager_away",
    # European midweek load (C.3)
    "uefa_rest_days_home",
    "uefa_rest_days_away",
    "uefa_load_7d_home",
    "uefa_load_7d_away",
    # Crowd / weather / referee (C.4)
    "attendance",
    "temp_c",
    "wind_kmh",
    "rain_mm",
    "referee_yellow_avg",
    # Division-below seeding (C.6)
    "promoted_prior_rating_home",
    "promoted_prior_rating_away",
]

# Harmless non-feature columns allowed in the file (identifiers/descriptions).
_KEY_COLUMNS = {"date", "home_team", "away_team"}


def context_feature_columns() -> list[str]:
    """The whitelist of per-match pre-match features the pipeline understands."""
    return list(_CONTEXT_FEATURE_COLUMNS)


def _normalise_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise Date/HomeTeam/AwayTeam into canonical date/home_team/away_team
    when present (missing key columns are simply left missing)."""
    df = df.copy()
    rename = {}
    for src, dst in (
        ("Date", "date"),
        ("HomeTeam", "home_team"),
        ("AwayTeam", "away_team"),
    ):
        if src in df.columns and dst not in df.columns:
            rename[src] = dst
    df = df.rename(columns=rename)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"], format="mixed", dayfirst=True, errors="coerce"
        )
    for col in ("home_team", "away_team"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def read_context_season(
    path: Path,
) -> tuple[pd.DataFrame, list[str]]:
    """Read one context CSV; returns (normalised frame, whitelisted columns)."""
    if not Path(path).exists():
        return pd.DataFrame(), []
    df = pd.read_csv(path, dtype=str).fillna("")
    df = _normalise_frame(df)
    if not {"date", "home_team", "away_team"}.issubset(df.columns):
        return pd.DataFrame(), []
    keep = _KEY_COLUMNS | {c for c in _CONTEXT_FEATURE_COLUMNS if c in df.columns}
    used = sorted(c for c in _CONTEXT_FEATURE_COLUMNS if c in df.columns)
    for col in used:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[list(keep)], used


def available_context_files(context_dir: str | Path = "data/context") -> list[Path]:
    """All context season CSVs present under data/context/<LEAGUE>/."""
    return sorted(Path(context_dir).glob("*/*.csv"))


def join_context(
    df: pd.DataFrame,
    context_dir: str | Path = "data/context",
    league: Optional[str] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Left-join every available context file onto ``df``.

    Returns ``(df_with_columns, added_columns)``. When no context file exists
    (or nothing is on the whitelist), the frame is returned unchanged with an
    empty column list — the feature turns itself off.
    """
    files = available_context_files(context_dir)
    if league is not None:
        files = [f for f in files if f.parent.name == league]
    if not files:
        return df, []

    merged = df.copy()
    if "date" not in merged.columns and "Date" in merged.columns:
        merged["date"] = pd.to_datetime(
            merged["Date"], format="mixed", dayfirst=True, errors="coerce"
        )
    for f in files:
        ctx, used = read_context_season(f)
        if used:
            merged = merged.merge(
                ctx,
                on=["date", "home_team", "away_team"],
                how="left",
                suffixes=("", "_ctx"),
            )
    added = [c for c in _CONTEXT_FEATURE_COLUMNS if c in merged.columns]
    return merged, added
