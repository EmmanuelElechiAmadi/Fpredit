"""Normalise a raw football-data.co.uk season CSV into canonical form.

- Combine ``Date`` + ``Time`` into a UTC kickoff timestamp.
- Map every team name through ``config/team_names.yaml`` (an unmapped name
  raises :class:`UnmappedTeamError` — the league is halted, never guessed).
- Coerce column types (ints for counts, floats for xG/odds).
- Split bookmaker odds out of the match row into a list of odds records.

The output matches DataFrame uses the spec's short stat names (``hs``, ``as``,
``hst`` …) and keeps raw values untouched.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.update.config import canonicalise

# source column (case-insensitive) -> canonical field
_FIELD_ALIASES = {
    "Date": "date",
    "Time": "time",
    "HomeTeam": "home",
    "AwayTeam": "away",
    "FTHG": "fthg",
    "FTAG": "ftag",
    "FTR": "ftr",
    "HTHG": "hthg",
    "HTAG": "htag",
    "HTR": "htr",
    "HS": "hs",
    "AS": "as",
    "HST": "hst",
    "AST": "ast",
    "HC": "hc",
    "AC": "ac",
    "HF": "hf",
    "AF": "af",
    "HY": "hy",
    "AY": "ay",
    "HR": "hr",
    "AR": "ar",
    "Referee": "referee",
}

_INT_FIELDS = [
    "fthg",
    "ftag",
    "hthg",
    "htag",
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
]

# Bookmaker 1X2 odds triples: (home, draw, away, bookmaker, is_closing).
_1X2_TRIPLES = [
    ("B365H", "B365D", "B365A", "B365", True),
    ("BbAvH", "BbAvD", "BbAvA", "BbAv", True),
    ("PSH", "PSD", "PSA", "PS", True),
    ("PH", "PD", "PA", "P", False),
    ("MaxH", "MaxD", "MaxA", "Max", True),
    ("AvgH", "AvgD", "AvgA", "Avg", True),
]

_OU_PAIRS = [
    ("B365>2.5", "B365<2.5", "B365", 2.5),
    ("Max>2.5", "Max<2.5", "Max", 2.5),
    ("Avg>2.5", "Avg<2.5", "Avg", 2.5),
]

_AH_TRIPLES = [
    ("B365AHH", "B365AHA", "B365", True),
    ("MaxAHH", "MaxAHA", "Max", True),
    ("AvgAHH", "AvgAHA", "Avg", True),
]


class UnmappedTeamError(RuntimeError):
    """One or more source team names have no canonical mapping."""

    def __init__(self, league: str, teams: set[str]):
        self.league = league
        self.teams = sorted(teams)
        super().__init__(
            f"Unmapped team(s) in {league}: {', '.join(self.teams)} — "
            "add them to config/team_names.yaml and re-run."
        )


def _col(df: pd.DataFrame, *names: str) -> Optional[str]:
    available = {c.strip().casefold(): c for c in df.columns}
    for name in names:
        if name.casefold() in available:
            return available[name.casefold()]
    return None


def _to_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _parse_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="mixed", dayfirst=True, errors="coerce")


def _derive_ftr(row) -> object:
    if pd.notna(row.get("ftr")) and str(row.get("ftr")).strip() in ("H", "D", "A"):
        return str(row["ftr"]).strip()
    if pd.isna(row.get("fthg")) or pd.isna(row.get("ftag")):
        return pd.NA
    h, a = int(row["fthg"]), int(row["ftag"])
    return "H" if h > a else ("D" if h == a else "A")


def normalise_season_csv(
    raw: pd.DataFrame,
    league: str,
    season: str,
    team_map: dict,
    source: str = "football-data.co.uk",
) -> tuple[pd.DataFrame, list[dict], set[str]]:
    """Return (matches_df, odds_records, unmapped_teams)."""
    # Rename source columns to canonical fields.
    rename = {}
    for src_name, canonical in _FIELD_ALIASES.items():
        col = _col(raw, src_name)
        if col is not None:
            rename[col] = canonical
    df = raw.rename(columns=rename)

    out = pd.DataFrame()
    for field in ["date", "time", "home", "away", "ftr", "htr", "referee"]:
        out[field] = df[field] if field in df.columns else pd.NA

    for field in _INT_FIELDS:
        out[field] = _to_int(df[field]) if field in df.columns else pd.NA

    # Dates -> UTC kickoff (naive UTC; football-data Time is the local kickoff).
    dates = (
        _parse_date(out["date"]) if "date" in out else pd.Series([pd.NaT] * len(out))
    )
    times = out["time"].astype(str).str.strip().str.replace("nan", "", regex=False)
    kickoffs = pd.to_datetime(
        dates.dt.strftime("%Y-%m-%d") + " " + times.where(times != "", "00:00"),
        format="%Y-%m-%d %H:%M",
        errors="coerce",
    )
    out["date"] = dates.dt.normalize()
    out["kickoff_utc"] = kickoffs

    # Team names -> canonical (halt on unknown).
    unmapped: set[str] = set()
    for col in ("home", "away"):

        def _map(name: str) -> str:
            canon = canonicalise(league, str(name), team_map)
            if canon is None:
                unmapped.add(str(name).strip())
                return str(name)
            return canon

        out[col] = out[col].map(_map)

    out["ftr"] = out.apply(_derive_ftr, axis=1)
    out["league"] = league
    out["season"] = season
    out["source"] = source

    def _key_of(r) -> str:
        d = r.get("date")
        dstr = d.date().isoformat() if pd.notna(d) else ""
        return f"{dstr}|{r.get('home')}|{r.get('away')}"

    out["_key"] = out.apply(_key_of, axis=1)

    # Odds split out of the raw row, keyed by the match natural key so they
    # can be joined back onto validated rows after quarantine filtering.
    odds_records: list[dict] = []
    for pos in range(len(out)):
        row_raw = raw.iloc[pos]
        key = out.iloc[pos]["_key"]
        for hc, dc, ac, book, closing in _1X2_TRIPLES:
            if all(c in raw.columns for c in (hc, dc, ac)):
                vals = [row_raw[hc], row_raw[dc], row_raw[ac]]
                if any(pd.isna(v) for v in vals):
                    continue
                odds_records.append(
                    {
                        "_key": key,
                        "bookmaker": book,
                        "market": "1X2",
                        "home": float(row_raw[hc]),
                        "draw": float(row_raw[dc]),
                        "away": float(row_raw[ac]),
                        "is_closing": int(closing),
                    }
                )
        for oc, uc, book, line in _OU_PAIRS:
            if all(c in raw.columns for c in (oc, uc)):
                if pd.isna(row_raw[oc]) or pd.isna(row_raw[uc]):
                    continue
                odds_records.append(
                    {
                        "_key": key,
                        "bookmaker": book,
                        "market": "OU",
                        "over": float(row_raw[oc]),
                        "under": float(row_raw[uc]),
                        "line": float(line),
                        "is_closing": 1,
                    }
                )
        ah_line_col = _col(raw, "AHh", "AHCh")
        for hc, ac, book, closing in _AH_TRIPLES:
            if all(c in raw.columns for c in (hc, ac)):
                if pd.isna(row_raw[hc]) or pd.isna(row_raw[ac]):
                    continue
                odds_records.append(
                    {
                        "_key": key,
                        "bookmaker": book,
                        "market": "AH",
                        "home": float(row_raw[hc]),
                        "away": float(row_raw[ac]),
                        "line": (
                            float(row_raw[ah_line_col])
                            if ah_line_col and pd.notna(row_raw[ah_line_col])
                            else None
                        ),
                        "is_closing": int(closing),
                    }
                )

    return out, odds_records, unmapped


DIV_TO_LEAGUE = {"E0": "EPL", "SP1": "LALIGA", "I1": "SERIEA"}


def _season_for_date(d) -> str:
    if d is None or pd.isna(d):
        return ""
    y = d.year if d.month >= 7 else d.year - 1
    return f"{y}-{(y + 1) % 100:02d}"


def normalise_fixtures_csv(
    raw: pd.DataFrame,
    team_map: dict,
    source: str = "football-data.co.uk",
) -> tuple[pd.DataFrame, set[str]]:
    """Parse ``fixtures.csv`` into (fixtures_df, unmapped_teams).

    Only the three supported leagues (E0/SP1/I1) are kept; team names are
    canonicalised; Date+Time become a UTC kickoff; and the latest pre-match
    1X2 odds (B365/Avg/Max) are carried inline for market-edge comparison.
    """
    if "Div" not in raw.columns:
        return pd.DataFrame(), set()
    raw = raw[raw["Div"].isin(DIV_TO_LEAGUE)].reset_index(drop=True)

    out = pd.DataFrame()
    out["league"] = raw["Div"].map(DIV_TO_LEAGUE)
    out["home"] = raw["HomeTeam"].astype(str).str.strip()
    out["away"] = raw["AwayTeam"].astype(str).str.strip()

    dates = _parse_date(raw["Date"])
    times = raw["Time"].astype(str).str.strip().str.replace("nan", "", regex=False)
    kickoffs = pd.to_datetime(
        dates.dt.strftime("%Y-%m-%d") + " " + times.where(times != "", "00:00"),
        format="%Y-%m-%d %H:%M",
        errors="coerce",
    )
    out["date"] = dates.dt.normalize()
    out["kickoff_utc"] = kickoffs
    out["season"] = dates.map(_season_for_date)
    out["referee"] = raw["Referee"] if "Referee" in raw.columns else pd.NA
    out["source"] = source

    unmapped: set[str] = set()

    def _canon(row, col: str) -> str:
        c = canonicalise(row["league"], str(row[col]), team_map)
        if c is None:
            unmapped.add(str(row[col]))
            return str(row[col])
        return c

    out["home"] = out.apply(lambda r: _canon(r, "home"), axis=1)
    out["away"] = out.apply(lambda r: _canon(r, "away"), axis=1)

    for out_col, src_col in (
        ("b365h", "B365H"),
        ("b365d", "B365D"),
        ("b365a", "B365A"),
        ("avgh", "AvgH"),
        ("avgd", "AvgD"),
        ("avga", "AvgA"),
        ("maxh", "MaxH"),
        ("maxd", "MaxD"),
        ("maxa", "MaxA"),
    ):
        out[out_col] = (
            pd.to_numeric(raw[src_col], errors="coerce")
            if src_col in raw.columns
            else pd.NA
        )

    out = (
        out[out["date"].notna()]
        .sort_values(["league", "kickoff_utc"])
        .reset_index(drop=True)
    )
    return out, unmapped
