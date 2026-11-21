#!/usr/bin/env python3
"""Optional football-data.org sync: standings, fixtures and finished results.

football-data.org (free tier needs a token) fills the gaps that the free CSV
download cannot: the *official* standings table, the *scheduled* fixture list,
and finished results on the day they happen (the football-data.co.uk CSVs are
only re-published weekly).

    export FOOTBALL_DATA_ORG_API_KEY=...
    python3 scripts/fetch_football_data_org.py --action all --league EPL
    python3 scripts/fetch_football_data_org.py --action standings

When no key is set the script exits non-zero with a hint; ``scripts/refresh_data.py``
uses this module automatically when the environment variable is present and
skips it gracefully otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from scripts.import_standings import TEAM_ALIASES, season_for_date, write_snapshot

API_BASE = "https://api.football-data.org/v4"
# App league key -> football-data.org competition code (API URLs).
COMPETITIONS = {"EPL": "PL", "LALIGA": "PD", "SERIEA": "SA"}
# App league key -> football-data.co.uk folder code (on-disk data/raw, Div col).
FOLDER_CODES = {"EPL": "E0", "LALIGA": "SP1", "SERIEA": "I1"}


class FetchError(RuntimeError):
    """Network / HTTP / payload problem talking to football-data.org."""


def _season_param(today: date) -> str:
    """football-data.org season label (start year): Sep 2026 -> 2026."""
    year = today.year if today.month >= 7 else today.year - 1
    return str(year)


def _request(path: str, api_key: str, query: Optional[dict] = None) -> dict:
    """GET ``path`` from the API and return the decoded JSON payload."""
    url = API_BASE + path
    if query:
        from urllib.parse import urlencode

        url += "?" + urlencode(query)
    req = Request(url, headers={"X-Auth-Token": api_key})
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        # 403 usually means the free-plan quota is exhausted for the hour.
        if e.code == 403:
            raise FetchError(
                "football-data.org returned 403 (free-tier quota exhausted?)"
            ) from e
        if e.code == 404:
            raise FetchError(f"football-data.org 404 for {path}") from e
        raise FetchError(f"football-data.org HTTP {e.code}: {e.reason}") from e
    except URLError as e:
        raise FetchError(
            f"Network error talking to football-data.org: {e.reason}"
        ) from e
    except ValueError as e:
        raise FetchError(f"Bad JSON from football-data.org: {e}") from e


def normalize_team(
    league: str, name: Optional[str], short_name: Optional[str] = None
) -> str:
    """Map a football-data.org team name/short-name onto the football-data.co.uk
    spelling used across this project (passes through unknown names)."""
    mapping = TEAM_ALIASES.get(league, {})
    folded = {k.casefold(): v for k, v in mapping.items()}

    def _try(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        key = " ".join(value.split())
        if key.casefold() in folded:
            return folded[key.casefold()]
        # Many API names carry a club suffix ("Arsenal FC", "Athletic Club").
        for suffix in (" FC", " CF", " Club", " SAD"):
            stem = key[: -len(suffix)] if key.endswith(suffix) else None
            if stem and stem.casefold() in folded:
                return folded[stem.casefold()]
        return key

    return _try(name) or _try(short_name) or (name or short_name or "").strip()


def standings_rows(payload: dict, league: str) -> list[dict]:
    """Map a /competitions/{code}/standings payload to canonical snapshot rows."""
    standings = payload.get("standings") or []
    table: Optional[list] = None
    for s in standings:
        if s.get("type") == "TOTAL":
            table = s.get("table") or []
            break
    if table is None and standings:
        table = standings[0].get("table") or []

    rows: list[dict] = []
    for entry in table or []:
        team = entry.get("team") or {}
        name = normalize_team(league, team.get("name"), team.get("shortName"))
        gf = int(entry.get("goalsFor", 0))
        ga = int(entry.get("goalsAgainst", 0))
        rows.append(
            {
                "position": int(entry.get("position", len(rows) + 1)),
                "team": name,
                "played": int(entry.get("playedGames", 0)),
                "won": int(entry.get("won", 0)),
                "draw": int(entry.get("draw", 0)),
                "lost": int(entry.get("lost", 0)),
                "gf": gf,
                "ga": ga,
                "goal_diff": int(entry.get("goalDifference", gf - ga)),
                "points": int(entry.get("points", 0)),
            }
        )
    return rows


def _parse_utc_date(value: str) -> date:
    return (
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        .astimezone(timezone.utc)
        .date()
    )


def fixture_rows(payload: dict, league: str) -> list[dict]:
    """Map a /competitions/{code}/matches payload to upcoming-fixture rows."""
    rows = []
    for m in payload.get("matches") or []:
        if m.get("status") not in ("SCHEDULED", "TIMED"):
            continue
        home = normalize_team(league, (m.get("homeTeam") or {}).get("name"))
        away = normalize_team(league, (m.get("awayTeam") or {}).get("name"))
        if not home or not away:
            continue
        rows.append(
            {
                "date": _parse_utc_date(m["utcDate"]).isoformat(),
                "home_team": home,
                "away_team": away,
                "matchweek": m.get("matchday"),
            }
        )
    return rows


def result_rows(payload: dict, league: str) -> list[dict]:
    """Map finished matches to minimal raw-CSV rows (date dd/mm/yyyy)."""
    rows = []
    for m in payload.get("matches") or []:
        if m.get("status") != "FINISHED":
            continue
        score = m.get("score") or {}
        ft = score.get("fullTime") or {}
        hg, ag = ft.get("home"), ft.get("away")
        if hg is None or ag is None:
            continue
        home = normalize_team(league, (m.get("homeTeam") or {}).get("name"))
        away = normalize_team(league, (m.get("awayTeam") or {}).get("name"))
        if not home or not away:
            continue
        hg, ag = int(hg), int(ag)
        rows.append(
            {
                "Div": FOLDER_CODES[league],
                "Date": _parse_utc_date(m["utcDate"]).strftime("%d/%m/%Y"),
                "HomeTeam": home,
                "AwayTeam": away,
                "FTHG": hg,
                "FTAG": ag,
                "FTR": "H" if hg > ag else ("A" if ag > hg else "D"),
            }
        )
    return rows


def merge_results(league: str, rows: list[dict], data_dir: str, season: str) -> int:
    """Append finished matches to ``data/raw/<CODE>/<season>.csv`` (dedup).

    Preserves any extra columns the existing file has (odds, xG, ...); the
    minimal API rows simply fill the core columns. Returns rows added.
    """
    code = FOLDER_CODES[league]
    path = Path(data_dir) / code / f"{season}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        df = pd.read_csv(path, dtype=str).fillna("")
        cols = list(df.columns)
    else:
        df = pd.DataFrame()
        cols = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

    existing = set()
    if not df.empty:
        existing = {
            (
                str(r["Date"]).strip(),
                str(r["HomeTeam"]).strip(),
                str(r["AwayTeam"]).strip(),
            )
            for _, r in df.iterrows()
        }

    added = 0
    new_rows = []
    for r in rows:
        key = (
            str(r["Date"]).strip(),
            str(r["HomeTeam"]).strip(),
            str(r["AwayTeam"]).strip(),
        )
        if key in existing:
            continue
        existing.add(key)
        new_rows.append({c: r.get(c, "") for c in cols})
        added += 1

    if added:
        append = pd.DataFrame(new_rows, columns=cols)
        out = pd.concat([df, append], ignore_index=True) if not df.empty else append
        out["_d"] = pd.to_datetime(out["Date"], dayfirst=True, errors="coerce")
        out = out.sort_values("_d").drop(columns=["_d"])
        out.to_csv(path, index=False)
    return added


def sync_standings(
    league: str,
    api_key: str,
    season: Optional[str] = None,
    out_dir: Optional[Path] = None,
    as_of: Optional[date] = None,
) -> Path:
    """Fetch the current official standings and write a standings snapshot."""
    today = as_of or date.today()
    season = season or _season_param(today)
    payload = _request(
        f"/competitions/{COMPETITIONS[league]}/standings",
        api_key,
        query={"season": season},
    )
    rows = standings_rows(payload, league)
    if not rows:
        raise FetchError(f"Empty standings payload for {league} season {season}.")
    code = COMPETITIONS[league]
    out_dir = out_dir or Path(__file__).resolve().parent.parent / "data" / "standings"
    return write_snapshot(
        league,
        rows,
        as_of=today,
        season=season_for_date(today),
        out_dir=out_dir,
        source_override=f"football-data.org v4 {code} standings (season {season})",
    )


def sync_fixtures(
    league: str,
    api_key: str,
    season: Optional[str] = None,
    fixtures_dir: Optional[Path] = None,
    replace: bool = False,
) -> int:
    """Fetch scheduled matches and merge them into the league's fixtures file.

    With ``replace=True`` (the automatic-refresh path) the current season's
    fixture file is replaced by the API's official list instead of merging —
    so a placeholder round-robin never lingers next to the real schedule.
    """
    today = date.today()
    season = season or _season_param(today)
    payload = _request(
        f"/competitions/{COMPETITIONS[league]}/matches",
        api_key,
        query={"season": season, "status": "SCHEDULED"},
    )
    rows = fixture_rows(payload, league)
    if not rows:
        return 0
    from src.fixtures import write_fixtures_batch

    fixtures_dir = (
        fixtures_dir or Path(__file__).resolve().parent.parent / "data" / "fixtures"
    )
    if replace:
        season_file = (
            Path(fixtures_dir) / FOLDER_CODES[league] / f"{season_for_date(today)}.csv"
        )
        season_file.unlink(missing_ok=True)
    return write_fixtures_batch(league, pd.DataFrame(rows), raw_dir=str(fixtures_dir))


def sync_results(
    league: str,
    api_key: str,
    season: Optional[str] = None,
    data_dir: Optional[Path] = None,
) -> int:
    """Fetch finished matches and merge them into the results CSV."""
    today = date.today()
    season = season or _season_param(today)
    payload = _request(
        f"/competitions/{COMPETITIONS[league]}/matches",
        api_key,
        query={"season": season, "status": "FINISHED"},
    )
    rows = result_rows(payload, league)
    data_dir = data_dir or Path(__file__).resolve().parent.parent / "data" / "raw"
    season_label = season_for_date(today)
    return merge_results(league, rows, str(data_dir), season_label)


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--action",
        choices=["standings", "fixtures", "results", "all"],
        default="all",
        help="What to sync (default: all).",
    )
    ap.add_argument(
        "--league",
        choices=list(COMPETITIONS),
        default=None,
        help="League to sync (default: all three).",
    )
    ap.add_argument(
        "--season",
        help="Football season start year (default: current). Useful to pull "
        "historical seasons back too.",
    )
    ap.add_argument("--api-key", help="Override FOOTBALL_DATA_ORG_API_KEY.")
    args = ap.parse_args(list(argv) if argv else None)

    api_key = args.api_key or os.environ.get("FOOTBALL_DATA_ORG_API_KEY", "").strip()
    if not api_key:
        print(
            "No API key. Set FOOTBALL_DATA_ORG_API_KEY (free token from "
            "https://www.football-data.org) or pass --api-key.",
            file=sys.stderr,
        )
        return 2

    leagues = [args.league] if args.league else list(COMPETITIONS)
    failed = False
    for league in leagues:
        try:
            if args.action in ("standings", "all"):
                p = sync_standings(league, api_key, season=args.season)
                print(f"[{league}] standings -> {p}")
            if args.action in ("fixtures", "all"):
                n = sync_fixtures(league, api_key, season=args.season)
                print(f"[{league}] merged {n} scheduled fixtures")
            if args.action in ("results", "all"):
                n = sync_results(league, api_key, season=args.season)
                print(f"[{league}] merged {n} finished results")
        except FetchError as e:
            failed = True
            print(f"[{league}] {e}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
