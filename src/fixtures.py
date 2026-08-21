"""Upcoming fixtures — new-season previews you can run the model on.

Fixtures live in ``data/fixtures/<LEAGUE_CODE>/<season>.csv`` (one row per
match) and are surfaced in the web UI's **Fixtures** view so you can see who
is playing and run the ensemble on any match.

Two CSV shapes are accepted:

1. football-data.co.uk style::

       Date,Time,HomeTeam,AwayTeam,Div,B365H,B365D,B365A,...
       14/08/2026,20:00,Arsenal,Chelsea,E0,1.75,3.60,4.50

2. Minimal::

       date,home_team,away_team,matchweek
       2026-08-15,Arsenal,Chelsea,1

When the official new-season file appears on football-data.co.uk, drop it here
(the UI + :func:`load_fixtures` pick it up automatically). Before that, use the
UI's add-fixture form or :func:`add_fixture`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from src.data_loader import LEAGUE_CODES, _find_column
from src.log import get_logger

log = get_logger(__name__)

DEFAULT_RAW_DIR = "data/fixtures"

# Column aliases for the two accepted CSV shapes.
_COLUMN_ALIASES = {
    "date": ["Date", "date", "Kickoff", "kickoff"],
    "home_team": ["HomeTeam", "Home", "home_team", "home"],
    "away_team": ["AwayTeam", "Away", "away_team", "away"],
    "matchweek": ["Matchweek", "matchweek", "MW", "Round", "round", "Wk", "wk"],
}

_REQUIRED = {"date", "home_team", "away_team"}


def _normalize(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Map a fixtures CSV to canonical columns (date, home_team, away_team, matchweek)."""
    mapping = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        col = _find_column(df, aliases)
        if col is not None:
            mapping[col] = canonical
        elif canonical in _REQUIRED:
            return None
    out = df.rename(columns=mapping)[list(mapping.values())]
    out = out.dropna(subset=["home_team", "away_team"])
    out["home_team"] = out["home_team"].astype(str).str.strip()
    out["away_team"] = out["away_team"].astype(str).str.strip()
    if "matchweek" not in out.columns:
        out["matchweek"] = pd.NA
    return out


def _league_folder(raw_dir: str, league: str) -> Path:
    return Path(raw_dir) / LEAGUE_CODES[league]


def season_for_date(d) -> str:
    """Northern-hemisphere football season label for a date: 2026-08-15 -> '2026-27'."""
    ts = pd.Timestamp(d)
    year = ts.year if ts.month >= 7 else ts.year - 1
    return f"{year}-{(year + 1) % 100:02d}"


def _parse_dates(s: pd.Series) -> pd.Series:
    """Parse mixed date formats: ISO (2026-08-15) and football-data style (14/08/2026).

    ``format='mixed'`` detects each value's format; ``dayfirst=True`` makes the
    football-data style (and genuinely ambiguous MM/DD vs DD/MM) read as day-first,
    which is what football-data.co.uk and the add-fixture form always write.
    """
    return pd.to_datetime(s, format="mixed", dayfirst=True, errors="coerce")


def load_fixtures(
    raw_dir: str = DEFAULT_RAW_DIR,
    league: str = "EPL",
    today: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Load all upcoming (future) fixtures for a league.

    Reads every CSV in ``data/fixtures/<LEAGUE_CODE>/`` and returns rows with
    columns: date, home_team, away_team, matchweek, season, source.
    Past-dated rows are dropped so the list always shows *who is playing next*.
    """
    folder = _league_folder(raw_dir, league)
    if not folder.exists():
        return _empty_fixtures()

    frames = []
    for f in sorted(folder.glob("*.csv")):
        try:
            df = pd.read_csv(f)
        except Exception as e:  # pragma: no cover - defensive
            log.warning("Could not read fixtures file %s: %s", f.name, e)
            continue
        norm = _normalize(df)
        if norm is None or norm.empty:
            log.warning("Skipping %s: could not map fixture columns", f.name)
            continue
        norm = norm.copy()
        norm["season"] = f.stem
        norm["source"] = "csv"
        frames.append(norm)

    if not frames:
        return _empty_fixtures()

    out = pd.concat(frames, ignore_index=True)
    out["date"] = _parse_dates(out["date"])
    out = out.dropna(subset=["date"])

    cutoff = (today or pd.Timestamp.now().normalize()) - pd.Timedelta(days=1)
    out = out[out["date"] >= cutoff]

    out = out.sort_values(["date", "home_team"]).reset_index(drop=True)
    out["matchweek"] = out["matchweek"].astype("Int64")
    return out[["date", "home_team", "away_team", "matchweek", "season", "source"]]


def _empty_fixtures() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["date", "home_team", "away_team", "matchweek", "season", "source"]
    )


def write_fixtures_batch(
    league: str,
    rows_df: pd.DataFrame,
    raw_dir: str = DEFAULT_RAW_DIR,
) -> int:
    """Efficiently merge a batch of fixtures into the league's season files.

    ``rows_df`` needs date, home_team, away_team and optional matchweek columns.
    Rows are deduplicated on (date, home_team, away_team) against both the
    existing files and each other, and each season file is rewritten once
    (unlike looping :func:`add_fixture`, which is O(n²) on large batches).

    Returns the number of rows actually added.
    """
    batch = rows_df.copy()
    batch["date"] = _parse_dates(batch["date"])
    batch = batch.dropna(subset=["date", "home_team", "away_team"])
    batch["home_team"] = batch["home_team"].astype(str).str.strip()
    batch["away_team"] = batch["away_team"].astype(str).str.strip()
    if "matchweek" not in batch.columns:
        batch["matchweek"] = pd.NA
    batch["season"] = batch["date"].apply(season_for_date)

    added = 0
    for season, grp in batch.groupby("season"):
        path = _league_folder(raw_dir, league) / f"{season}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)

        existing = pd.DataFrame(columns=["Date", "HomeTeam", "AwayTeam"])
        if path.exists():
            existing = pd.read_csv(path, dtype=str).fillna("")
            for col in ("Date", "HomeTeam", "AwayTeam"):
                if col not in existing.columns:
                    existing[col] = ""

        new_rows = pd.DataFrame(
            {
                "Date": grp["date"].dt.strftime("%d/%m/%Y"),
                "HomeTeam": grp["home_team"],
                "AwayTeam": grp["away_team"],
                "Matchweek": grp["matchweek"]
                .astype("Int64")
                .astype("string")
                .fillna(""),
            }
        )

        combined = pd.concat([existing, new_rows], ignore_index=True)
        combined["_d"] = combined["Date"].astype(str).str.strip()
        combined["_h"] = combined["HomeTeam"].astype(str).str.strip()
        combined["_a"] = combined["AwayTeam"].astype(str).str.strip()
        combined = combined.drop_duplicates(subset=["_d", "_h", "_a"], keep="first")
        combined = combined.drop(columns=["_d", "_h", "_a"])

        combined["_dt"] = _parse_dates(combined["Date"])
        combined = combined.sort_values("_dt").drop(columns=["_dt"])
        combined.to_csv(path, index=False)

        added += len(combined) - len(existing)

    log.info("Batch-wrote %d fixtures for %s", added, league)
    return added


def add_fixture(
    league: str,
    home_team: str,
    away_team: str,
    date,
    matchweek: Optional[int] = None,
    raw_dir: str = DEFAULT_RAW_DIR,
) -> dict:
    """Append one fixture to ``data/fixtures/<LEAGUE_CODE>/<season>.csv``.

    Existing rows in the season file are preserved. Returns the stored row.
    """
    home_team = (home_team or "").strip()
    away_team = (away_team or "").strip()
    if not home_team or not away_team:
        raise ValueError("Both home and away teams are required.")
    if home_team == away_team:
        raise ValueError("A team cannot play itself.")

    d = pd.Timestamp(date)
    season = season_for_date(d)
    path = _league_folder(raw_dir, league) / f"{season}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "Date": d.strftime("%d/%m/%Y"),
        "HomeTeam": home_team,
        "AwayTeam": away_team,
    }
    if matchweek is not None:
        row["Matchweek"] = int(matchweek)

    if path.exists():
        existing = pd.read_csv(path, dtype=str).fillna("")
        # Idempotent: skip when this exact (date, home, away) already exists.
        already = (
            (existing["Date"].astype(str).str.strip() == row["Date"])
            & (existing["HomeTeam"].astype(str).str.strip() == home_team)
            & (existing["AwayTeam"].astype(str).str.strip() == away_team)
        )
        if already.any():
            log.info(
                "Fixture %s vs %s (%s) already present; skipping",
                home_team,
                away_team,
                d.date(),
            )
            return {
                "date": d.date().isoformat(),
                "home_team": home_team,
                "away_team": away_team,
                "matchweek": matchweek,
                "season": season,
            }
        if matchweek is None:
            # Preserve the file's existing column layout even when no matchweek given.
            if "Matchweek" in existing.columns:
                row["Matchweek"] = ""
        existing = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        existing.to_csv(path, index=False)
    else:
        pd.DataFrame([row]).to_csv(path, index=False)

    log.info(
        "Added fixture %s vs %s (%s) -> %s", home_team, away_team, d.date(), path.name
    )
    return {
        "date": d.date().isoformat(),
        "home_team": home_team,
        "away_team": away_team,
        "matchweek": matchweek,
        "season": season,
    }


def delete_fixture(
    league: str,
    home_team: str,
    away_team: str,
    date,
    raw_dir: str = DEFAULT_RAW_DIR,
) -> bool:
    """Remove a fixture row from its season file. Returns True when a row was deleted."""
    d = pd.Timestamp(date)
    season = season_for_date(d)
    path = _league_folder(raw_dir, league) / f"{season}.csv"
    if not path.exists():
        return False

    df = pd.read_csv(path, dtype=str).fillna("")
    date_parsed = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    mask = (
        (df["HomeTeam"].astype(str).str.strip() == home_team.strip())
        & (df["AwayTeam"].astype(str).str.strip() == away_team.strip())
        & (date_parsed == d.normalize())
    )
    if not mask.any():
        return False

    remaining = df.loc[~mask]
    if remaining.empty:
        path.unlink(missing_ok=True)
        log.info("Removed last fixture; deleted %s", path.name)
    else:
        remaining.to_csv(path, index=False)
    log.info("Deleted fixture %s vs %s (%s)", home_team, away_team, d.date())
    return True


def generate_round_robin(
    teams: Iterable[str],
    start_date=None,
) -> pd.DataFrame:
    """Generate a full double round-robin placeholder schedule (circle method).

    Returns a DataFrame with columns date, home_team, away_team, matchweek.
    Dates are one week apart, defaulting to the first Saturday on/after
    ``today + 14 days`` so there is breathing room before kickoff.

    This is explicitly a *placeholder* schedule — swap in the official
    fixtures when they are released. Matchweek 1..N are assigned home/away,
    and the reverse fixtures follow in matchweeks N+1..2N.
    """
    teams = [t for t in dict.fromkeys(t.strip() for t in teams) if t]
    if len(teams) < 4:
        raise ValueError("At least 4 teams are needed for a round-robin.")

    n_even = len(teams)
    if n_even % 2 == 1:
        teams = teams + ["(bye)"]
        n_even = len(teams)

    today = pd.Timestamp.now().normalize()
    if start_date is None:
        start = today + pd.Timedelta(days=14)
        # First Saturday on/after the target date.
        start = start + pd.Timedelta(days=(5 - start.dayofweek) % 7)
    else:
        start = pd.Timestamp(start_date)

    rounds: list[dict] = []
    # Standard circle method: keep the first element fixed and rotate the rest.
    # Round r pairs index i with index (n_even-1-i), home/away alternated by parity.
    row = teams[:]
    for r in range(n_even - 1):
        for i in range(n_even // 2):
            home, away = row[i], row[n_even - 1 - i]
            if r % 2 == 1:
                home, away = away, home
            if "(bye)" not in (home, away):
                rounds.append(
                    {
                        "date": start + pd.Timedelta(days=7 * r),
                        "home_team": home,
                        "away_team": away,
                        "matchweek": r + 1,
                    }
                )
        row = [row[0]] + [row[-1]] + row[1:-1]

    # Second half: reverse home/away of the first half.
    for m in list(rounds):
        rounds.append(
            {
                "date": start + pd.Timedelta(days=7 * (n_even - 1 + m["matchweek"])),
                "home_team": m["away_team"],
                "away_team": m["home_team"],
                "matchweek": (n_even - 1) + m["matchweek"],
            }
        )

    return pd.DataFrame(rounds)
