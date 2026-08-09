"""
Loader for football-data.co.uk historical CSVs — the standard free source
for EPL, La Liga and Serie A results (no API key needed, updated weekly).

Download season files from:
  https://www.football-data.co.uk/englandm.php   (Premier League = E0)
  https://www.football-data.co.uk/spainm.php     (La Liga = SP1)
  https://www.football-data.co.uk/italym.php     (Serie A = I1)

Each season file (e.g. E0.csv, SP1.csv, I1.csv) has columns including:
  Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, plus closing odds (B365H etc.)

IMPORTANT: football-data.co.uk changed column naming conventions around 2016
(switched from upper-case to lower-case for some columns) and different seasons
may have slightly different column sets. This loader handles the most common
variants.

Drop the CSVs into data/raw/<LEAGUE_CODE>/<season>.csv and this loader
concatenates + standardizes them.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from src.log import get_logger

log = get_logger(__name__)

LEAGUE_CODES = {"EPL": "E0", "LALIGA": "SP1", "SERIEA": "I1"}

# football-data.co.uk has used various column name conventions over the years.
# We try multiple possible names for each canonical column.
_COLUMN_ALIASES = {
    "date": ["Date", "date"],
    "home_team": ["HomeTeam", "Home", "home_team", "home"],
    "away_team": ["AwayTeam", "Away", "away_team", "away"],
    "home_goals": ["FTHG", "HG", "home_goals", "ft_home_goals", "homegoals"],
    "away_goals": ["FTAG", "AG", "away_goals", "ft_away_goals", "awaygoals"],
    "result": ["FTR", "Res", "result", "ft_result", "full_time_result"],
}

REQUIRED_COLUMNS = {
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
}


def _find_column(df: pd.DataFrame, aliases: list[str]) -> Optional[str]:
    """Return the first column in df that matches one of the given aliases (case-insensitive)."""
    available = {c.lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in available:
            return available[alias.lower()]
    return None


def _normalize(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Rename a raw data frame to canonical column names. Returns None if required
    columns cannot be matched."""
    mapping = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        col = _find_column(df, aliases)
        if col is not None:
            mapping[col] = canonical
        elif canonical in REQUIRED_COLUMNS:
            log.warning(
                "Required column %s not found in %s", canonical, list(df.columns)
            )
            return None

    df = df.rename(columns=mapping)

    # Keep only canonical columns plus any betting odds columns we might want later
    keep = list(mapping.values())
    keep = [c for c in keep if c in df.columns]
    return df[keep]


def load_league_csvs(raw_dir: str, league: str) -> pd.DataFrame:
    """Load and concatenate all season CSVs for a given league.

    Parameters
    ----------
    raw_dir : str
        Path to the root data directory containing league-code subdirectories.
    league : str
        One of EPL, LALIGA, SERIEA.

    Returns
    -------
    pd.DataFrame
        Standardized data frame with columns:
        date, home_team, away_team, home_goals, away_goals, result, league
    """
    code = LEAGUE_CODES[league]
    folder = Path(raw_dir) / code
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSVs found in {folder}. Download season files from "
            f"football-data.co.uk for code '{code}' and place them there, "
            f"or use: python scripts/download_data.py --league {league}"
        )

    frames = []
    for f in files:
        try:
            # Try UTF-8 first, fall back to latin1 (older files)
            try:
                df = pd.read_csv(f, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(f, encoding="latin1")
        except Exception as e:
            log.warning("Could not read %s: %s", f.name, e)
            continue

        normalized = _normalize(df)
        if normalized is None or normalized.empty:
            log.warning("Skipping %s: could not map columns", f.name)
            continue

        # Defensive: discard files that belong to a different competition.
        # football-data.co.uk has been known to serve the wrong league's file for
        # a season path (e.g. SP1/2627 returning Scottish Championship rows), and
        # a wrong file would silently poison the training set.
        div_col = (
            "Div" if "Div" in df.columns else ("div" if "div" in df.columns else None)
        )
        if div_col is not None:
            codes = df[div_col].astype(str).str.strip()
            codes = codes[codes != ""]
            if codes.empty:
                log.warning("Skipping %s: Div column present but empty", f.name)
                continue
            top_code = codes.value_counts().index[0]
            if top_code != code:
                log.warning(
                    "Skipping %s: dominant Div value '%s' does not match expected "
                    "'%s' (file likely belongs to a different competition)",
                    f.name,
                    top_code,
                    code,
                )
                continue

        frames.append(normalized)
        log.info("Loaded %s (%d rows)", f.name, len(normalized))

    if not frames:
        raise ValueError(
            f"None of the {len(files)} CSV(s) in {folder} could be parsed. "
            f"Check they are valid football-data.co.uk exports."
        )

    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], dayfirst=True, errors="coerce")
    out = out.dropna(subset=["date", "home_goals", "away_goals"])

    # Force numeric types (some older CSVs store goals as strings)
    out["home_goals"] = pd.to_numeric(out["home_goals"], errors="coerce").astype(
        "Int64"
    )
    out["away_goals"] = pd.to_numeric(out["away_goals"], errors="coerce").astype(
        "Int64"
    )
    out = out.dropna(subset=["home_goals", "away_goals"])

    out = out.sort_values("date").reset_index(drop=True)
    out["league"] = league
    return out


def load_all_leagues(raw_dir: str, leagues: Optional[list[str]] = None) -> pd.DataFrame:
    """Load multiple leagues and concatenate into a single dataframe.

    Parameters
    ----------
    raw_dir : str
        Path to the root data directory.
    leagues : list of str, optional
        League codes to load. Defaults to all three (EPL, LALIGA, SERIEA).

    Returns
    -------
    pd.DataFrame
    """
    if leagues is None:
        leagues = list(LEAGUE_CODES.keys())

    frames = []
    for league in leagues:
        try:
            df = load_league_csvs(raw_dir, league)
            frames.append(df)
            log.info("Loaded %s: %d matches", league, len(df))
        except (FileNotFoundError, ValueError) as e:
            log.warning("Skipping %s: %s", league, e)

    if not frames:
        raise FileNotFoundError(
            f"No data could be loaded for any of {leagues} in {raw_dir}."
        )

    return (
        pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    )


def generate_synthetic_league(n_teams=18, n_seasons=3, seed=42) -> pd.DataFrame:
    """Generates a plausible synthetic league for demoing/testing the pipeline
    end-to-end when you don't have real CSVs loaded yet. NOT for real predictions."""
    import numpy as np

    rng = np.random.default_rng(seed)
    teams = [f"Team {chr(65+i)}" for i in range(n_teams)]
    true_strength = {t: rng.normal(0, 0.4) for t in teams}

    rows = []
    start = pd.Timestamp("2022-08-01")
    for season in range(n_seasons):
        season_start = start + pd.DateOffset(years=season)
        matchday = 0
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                date = season_start + pd.Timedelta(days=matchday * 3)
                matchday += 1
                lam = np.exp(0.35 + true_strength[home] - true_strength[away])
                mu = np.exp(true_strength[away] - true_strength[home])
                hg = rng.poisson(max(lam, 0.1))
                ag = rng.poisson(max(mu, 0.1))
                rows.append(
                    {
                        "date": date,
                        "home_team": home,
                        "away_team": away,
                        "home_goals": hg,
                        "away_goals": ag,
                        "result": "H" if hg > ag else ("D" if hg == ag else "A"),
                    }
                )
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["league"] = "SYNTHETIC"
    return df
