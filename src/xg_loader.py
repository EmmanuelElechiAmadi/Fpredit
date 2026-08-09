"""
xG data loader — Understat (understat.com) match-level expected goals.

Understat provides free, scrapeable match-level xG for the big five European
leagues back to 2014-15. League-season pages embed the match data as a JSON
blob (`var matchesData = JSON.parse('...')`) — no API key required.

This module:
  - scrapes that data and normalizes team names to football-data.co.uk style,
  - caches per-season CSVs under data/xg/<LEAGUE>/<season>.csv so the pipeline
    works offline after the first scrape,
  - provides join_xg() to merge xG onto a football-data.co.uk results frame,
  - provides a synthetic generator for tests/demos.

IMPORTANT: xG exists only from 2014-15 onward. Pre-2014-15 seasons have no
coverage, so pipelines must fall back to goals-only features for the older
era (the ensemble handles this automatically when xG columns are absent).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.log import get_logger

log = get_logger(__name__)

# Our league keys -> Understat's league slugs
UNDERSTAT_LEAGUES = {
    "EPL": "EPL",
    "LALIGA": "La_liga",
    "SERIEA": "Serie_A",
}

FIRST_XG_SEASON = 2014


def _season_strs(start: int = FIRST_XG_SEASON, end: Optional[int] = None) -> list[str]:
    end = end or datetime.now().year
    return [f"{y}-{str(y + 1)[2:]}" for y in range(start, end + 1)]


# football-data.co.uk name -> Understat name (extend as needed; unmatched teams
# pass through unchanged with a warning).
TEAM_NAME_MAP = {
    # Premier League
    "Leeds": "Leeds United",
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Newcastle": "Newcastle United",
    "Nott'm Forest": "Nottingham Forest",
    "Spurs": "Tottenham",
    "West Brom": "West Bromwich Albion",
    "Wolves": "Wolverhampton Wanderers",
    "QPR": "Queens Park Rangers",
    "Hull": "Hull City",
    "Cardiff": "Cardiff City",
    "Ipswich": "Ipswich Town",
    # La Liga
    "Alaves": "Alaves",
    "Deportivo": "Deportivo La Coruna",
    "Sporting Gijon": "Sporting Gijon",
    # Serie A
    "Verona": "Hellas Verona",
}
_NAME_MAP_REVERSE = {v: k for k, v in TEAM_NAME_MAP.items()}


def _normalize_names(df: pd.DataFrame) -> pd.DataFrame:
    """Map Understat full names back to football-data.co.uk short names."""
    for col in ("home_team", "away_team"):
        mapped = df[col].map(lambda t: _NAME_MAP_REVERSE.get(t, t))
        df = df.assign(**{col: mapped})
    return df


def _parse_matches_json(raw_json: str) -> pd.DataFrame:
    """Parse the JSON blob embedded in an Understat league-season page."""
    data = __import__("json").loads(raw_json)
    rows = []
    for match in data:
        dt = datetime.fromisoformat(match.get("datetime", "").replace("Z", "+00:00"))
        rows.append(
            {
                "date": pd.Timestamp(dt.date()),
                "home_team": match["h"]["title"],
                "away_team": match["a"]["title"],
                "home_goals": int(match["goals"]["h"]),
                "away_goals": int(match["goals"]["a"]),
                "home_xg": float(match["xG"]["h"]),
                "away_xg": float(match["xG"]["a"]),
            }
        )
    df = pd.DataFrame(rows)
    df["result"] = np.where(
        df["home_goals"] > df["away_goals"],
        "H",
        np.where(df["home_goals"] == df["away_goals"], "D", "A"),
    )
    return _normalize_names(df)


def _season_year(season: str) -> int:
    """'2023-24' -> 2023 (Understat URLs use a single year)."""
    return int(season.split("-")[0])


def _scrape_season(league: str, season: str, timeout: int = 30) -> pd.DataFrame:
    """Scrape one Understat league season via the (gzipped) JSON API.

    Understat's league page now loads match data client-side from
    /getLeagueData/<slug>/<year>; the payload's `dates` array carries the
    fixture list including per-match xG. Requires network access.
    """
    import gzip
    import json

    import requests  # type: ignore[import-untyped]

    slug = UNDERSTAT_LEAGUES.get(league)
    if slug is None:
        raise ValueError(f"Unsupported league for Understat: {league}")
    year = _season_year(season)
    url = f"https://understat.com/getLeagueData/{slug}/{year}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
        ),
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"https://understat.com/league/{slug}/{year}",
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    raw = resp.content
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        data = json.loads(gzip.decompress(raw))

    rows = []
    for m in data.get("dates") or []:
        if not m.get("isResult", True):
            continue
        dt = datetime.strptime(m["datetime"], "%Y-%m-%d %H:%M:%S")
        rows.append(
            {
                "date": pd.Timestamp(dt.date()),
                "home_team": m["h"]["title"],
                "away_team": m["a"]["title"],
                "home_goals": int(m["goals"]["h"]),
                "away_goals": int(m["goals"]["a"]),
                "home_xg": float(m["xG"]["h"]),
                "away_xg": float(m["xG"]["a"]),
            }
        )
    df = pd.DataFrame(rows)
    df["result"] = np.where(
        df["home_goals"] > df["away_goals"],
        "H",
        np.where(df["home_goals"] == df["away_goals"], "D", "A"),
    )
    return _normalize_names(df)


def fetch_league_xg(
    league: str,
    seasons: Optional[list[str]] = None,
    data_dir: str = "data/xg",
    refresh: bool = False,
) -> pd.DataFrame:
    """Scrape (and cache per-season CSVs) Understat xG for a league.

    Network is only needed for seasons not already cached. Returns a single
    concatenated DataFrame sorted by date.
    """
    seasons = seasons or _season_strs()
    out_dir = Path(data_dir) / league
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for season in seasons:
        cache_path = out_dir / f"{season}.csv"
        if cache_path.exists() and not refresh:
            df = pd.read_csv(cache_path)
            log.info("Loaded cached xG %s %s (%d matches)", league, season, len(df))
        else:
            log.info("Scraping Understat %s %s ...", league, season)
            df = _scrape_season(league, season)
            df.to_csv(cache_path, index=False)
        frames.append(df)

    if not frames:
        raise ValueError(f"No xG data for {league}")
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.sort_values("date").reset_index(drop=True)


def load_league_xg(data_dir: str, league: str) -> pd.DataFrame:
    """Load already-cached xG CSVs for a league (offline-safe)."""
    out_dir = Path(data_dir) / league
    if not out_dir.exists():
        raise FileNotFoundError(f"No cached xG for {league} in {data_dir}")
    frames = []
    for path in sorted(out_dir.glob("*.csv")):
        frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No cached xG for {league} in {data_dir}")
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.sort_values("date").reset_index(drop=True)


def join_xg(df: pd.DataFrame, xg_df: pd.DataFrame) -> pd.DataFrame:
    """Merge xG columns onto a football-data.co.uk results frame.

    The merge is on (date, home_team, away_team); rows with no xG match keep
    NaN xG (the ensemble then falls back to goals-only features for them).
    """
    if xg_df is None or xg_df.empty:
        return df
    key = ["date", "home_team", "away_team"]
    cols = key + ["home_xg", "away_xg"]
    sub = xg_df[cols].drop_duplicates(subset=key, keep="first")
    out = df.merge(sub, on=key, how="left")
    return out


def generate_synthetic_xg(df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Generate plausible xG columns for an existing results frame (tests/demos).

    xG is drawn as a noisy version of the expected-goals rate implied by a
    synthetic strength model, so it is correlated with but less noisy than the
    realized goals. Not for real predictions.
    """
    rng = np.random.default_rng(seed)
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    strength = {t: float(rng.normal(0, 0.35)) for t in teams}
    out = df.copy()
    home_xg, away_xg = [], []
    for _, row in df.iterrows():
        lam = np.exp(0.35 + strength[row["home_team"]] - strength[row["away_team"]])
        mu = np.exp(strength[row["away_team"]] - strength[row["home_team"]])
        home_xg.append(float(np.clip(rng.normal(lam, 0.25), 0.05, 6.0)))
        away_xg.append(float(np.clip(rng.normal(mu, 0.25), 0.05, 6.0)))
    out["home_xg"] = home_xg
    out["away_xg"] = away_xg
    return out
