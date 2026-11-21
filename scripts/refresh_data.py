#!/usr/bin/env python3
"""One-command weekly data refresh: results + odds, standings, fixtures, retrain.

This is the "the app is only as current as the CSVs you paste" fix. It
orchestrates the whole data-freshness pipeline for the three supported leagues:

1. **Results + odds** — refresh the current in-progress season's football-data.co.uk
   CSV (``--refresh`` semantics) and backfill any missing historical seasons.
2. **Standings** — pull the official table. Either the free CSV is enough for
   results, but the *table* needs one of: a pasted table file
   (``--standings-table table.txt``, auto-detected league, understands the EPL /
   La Liga / Serie A vertical layouts), or a ``FOOTBALL_DATA_ORG_API_KEY`` to
   fetch it automatically (see ``scripts/fetch_football_data_org.py``).
3. **Fixtures** — prune rows that have already been played (a match now in the
   results CSV stops being an upcoming fixture); with an API key the scheduled
   fixtures are also pulled.
4. **Retrain/validate** (optional) — run the walk-forward backtest (and/or
   pre-fit the web models) so the new data is exercised before you commit it.

Usage::

    # Manual weekly refresh (downloads; network required):
    python3 scripts/refresh_data.py --seasons 1

    # With pasted standings (no API key):
    python3 scripts/refresh_data.py --standings-table /tmp/seriea.txt

    # Fully automatic (API key set): results + standings + fixtures:
    FOOTBALL_DATA_ORG_API_KEY=xxx python3 scripts/refresh_data.py

    # Scheduled jobs use the same entry point (see .github/workflows/weekly-refresh.yml)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT / "data" / "raw"
DEFAULT_FIXTURES_DIR = ROOT / "data" / "fixtures"
DEFAULT_STANDINGS_DIR = ROOT / "data" / "standings"
DEFAULT_REPORTS_DIR = ROOT / "reports"

LEAGUE_CODES = {"EPL": "E0", "LALIGA": "SP1", "SERIEA": "I1"}


def _season_label(d: date) -> str:
    """European season label for a date: Sep 2026 -> '2026-27'."""
    year = d.year if d.month >= 7 else d.year - 1
    return f"{year}-{(year + 1) % 100:02d}"


def _parse_date_col(series) -> pd.Series:
    """Parse dd/mm/yyyy (football-data) or ISO dates to a datetime series."""
    return pd.to_datetime(series, format="mixed", dayfirst=True, errors="coerce")


def prune_played_fixtures(
    league: str,
    raw_dir: Path,
    fixtures_dir: Path,
) -> int:
    """Drop fixture rows whose match now exists in the results CSVs.

    A row that has a result is no longer an upcoming fixture, so keeping it in
    ``data/fixtures`` is stale. Returns the number of rows pruned across the
    league's season files. Idempotent: rows without a result are left alone
    (postponed matches, placeholder rounds still in the future, etc.).
    """
    code = LEAGUE_CODES[league]
    raw_dir = Path(raw_dir)

    # (iso-date, home, away) keys from every results CSV for the league.
    played: set[tuple[str, str, str]] = set()
    for path in sorted((raw_dir / code).glob("*.csv")):
        try:
            df = pd.read_csv(path, dtype=str).fillna("")
        except Exception:
            continue
        if not {"Date", "HomeTeam", "AwayTeam"}.issubset(df.columns):
            continue
        dates = _parse_date_col(df["Date"])
        for dt, ht, at in zip(dates, df["HomeTeam"], df["AwayTeam"]):
            if pd.isna(dt):
                continue
            played.add((dt.date().isoformat(), str(ht).strip(), str(at).strip()))

    pruned = 0
    folder = Path(fixtures_dir) / code
    for path in sorted(folder.glob("*.csv")):
        df = pd.read_csv(path, dtype=str).fillna("")
        if not {"Date", "HomeTeam", "AwayTeam"}.issubset(df.columns):
            continue
        dates = _parse_date_col(df["Date"])
        keep = [
            pd.isna(dt)
            or (dt.date().isoformat(), str(ht).strip(), str(at).strip()) not in played
            for dt, ht, at in zip(dates, df["HomeTeam"], df["AwayTeam"])
        ]
        remaining = df[keep]
        if len(remaining) == len(df):
            continue
        if remaining.empty:
            path.unlink(missing_ok=True)
        else:
            remaining.to_csv(path, index=False)
        pruned += int(len(df) - len(remaining))
    return pruned


def import_standings_table(
    path: Path,
    out_dir: Path = DEFAULT_STANDINGS_DIR,
    as_of: Optional[date] = None,
) -> str:
    """Import one pasted league-table file into data/standings.

    The league is auto-detected from the table title; all three importer
    layouts are understood (one row per line, the split BBC-style layout, and
    the vertical labelled Serie A layout). Returns the league key written.
    """
    sys.path.insert(0, str(ROOT))
    from scripts.import_standings import (
        detect_league,
        parse_table,
        season_for_date,
        write_snapshot,
    )

    text = Path(path).read_text()
    league = detect_league(text)
    rows = parse_table(text, league)
    as_of = as_of or date.today()
    season = season_for_date(as_of)
    out = write_snapshot(league, rows, as_of, season, out_dir=out_dir)
    print(f"[standings] {league}: {len(rows)} teams -> {out}")
    return league


def run_backtest(league: str, data_dir: Path, output_dir: Path) -> Path:
    """Run the walk-forward backtest for one league and return the output CSV.

    Slow (minutes/league) — used by ``--backtest`` to validate fresh data the
    same way the web Backtest Lab does.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"backtest_{league}_{_season_label(date.today())}.csv"
    cmd = [
        sys.executable,
        str(ROOT / "backtest.py"),
        "--league",
        league,
        "--data-dir",
        str(data_dir),
        "--output",
        str(out),
    ]
    print(f"[backtest] {league}: " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"Backtest failed for {league} (exit {result.returncode}).")
    print(f"[backtest] {league}: wrote {out}")
    return out


def prewarm_models(leagues: list[str], data_dir: Path, cache_dir: Path) -> None:
    """Refit + cache the web models on the refreshed data.

    The running web app already re-fits lazily whenever the data fingerprint
    changes; this step just makes the first post-refresh request fast by doing
    it eagerly.
    """
    sys.path.insert(0, str(ROOT))
    from predict import get_model

    cache_dir = Path(cache_dir)
    for league in leagues:
        try:
            get_model(league, str(data_dir), demo=False, cache_dir=str(cache_dir))
            print(f"[retrain] {league}: model fitted")
        except Exception as e:  # noqa: BLE001 - keep refreshing the others
            print(f"[retrain] {league}: failed ({e})")


def _api_key_from_env() -> str:
    import os

    return os.environ.get("FOOTBALL_DATA_ORG_API_KEY", "").strip()


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--leagues",
        nargs="+",
        choices=list(LEAGUE_CODES),
        default=list(LEAGUE_CODES),
        help="Leagues to refresh (default: all three).",
    )
    ap.add_argument(
        "--seasons",
        type=int,
        default=1,
        help="Number of seasons to consider (default 1 = the current, in-progress "
        "season). Completed seasons are only downloaded when missing.",
    )
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--fixtures-dir", type=Path, default=DEFAULT_FIXTURES_DIR)
    ap.add_argument("--standings-dir", type=Path, default=DEFAULT_STANDINGS_DIR)
    ap.add_argument(
        "--no-download",
        action="store_true",
        help="Skip the results/odds download step.",
    )
    ap.add_argument(
        "--no-prune",
        action="store_true",
        help="Skip pruning already-played rows out of data/fixtures.",
    )
    ap.add_argument(
        "--standings-table",
        action="append",
        metavar="PATH",
        help="A pasted league-table file to import (auto-detects the league; "
        "repeat for several leagues).",
    )
    ap.add_argument("--as-of", help="Snapshot date YYYY-MM-DD (default: today).")
    ap.add_argument(
        "--api-key",
        help="football-data.org key (default: $FOOTBALL_DATA_ORG_API_KEY).",
    )
    ap.add_argument(
        "--no-api",
        action="store_true",
        help="Do not use the football-data.org API even if a key is present.",
    )
    ap.add_argument(
        "--backtest",
        action="store_true",
        help="Run the walk-forward backtest for each refreshed league (slow).",
    )
    ap.add_argument(
        "--retrain",
        action="store_true",
        help="Eagerly refit the model pickle caches (slow, ~15s/league).",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any download failed (for scheduled jobs).",
    )
    ap.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    args = ap.parse_args(list(argv) if argv else None)

    as_of = date.fromisoformat(args.as_of) if args.as_of else date.today()
    api_key = args.api_key if args.api_key is not None else _api_key_from_env()
    use_api = bool(api_key) and not args.no_api

    errors: list[str] = []
    download_summary = {"downloaded": 0, "refreshed": 0, "failed": 0}

    # 1) Results + odds (football-data.co.uk weekly CSVs).
    if not args.no_download:
        sys.path.insert(0, str(ROOT))
        from scripts.download_data import download_league

        for league in args.leagues:
            print(f"\n== Download/refresh results for {league} ==")
            summary = download_league(
                league, args.seasons, str(args.data_dir), refresh=True
            )
            download_summary["downloaded"] += summary["downloaded"]
            download_summary["refreshed"] += summary["refreshed"]
            download_summary["failed"] += summary["failed"]
    else:
        print("== Skipping results download (--no-download) ==")

    # 2) Move played fixtures -> results: prune rows that now have results.
    pruned = 0
    if not args.no_prune:
        for league in args.leagues:
            pruned += prune_played_fixtures(league, args.data_dir, args.fixtures_dir)
        print(f"\n== Pruned {pruned} played fixture row(s) ==")
    else:
        print("== Skipping fixture pruning (--no-prune) ==")

    # 3) Standings: pasted tables first, then the API when a key is present.
    standings_done: list[str] = []
    if args.standings_table:
        for path in args.standings_table:
            try:
                standings_done.append(
                    import_standings_table(Path(path), args.standings_dir, as_of)
                )
            except ValueError as e:
                errors.append(f"standings import {path}: {e}")
    if use_api and len(standings_done) < len(args.leagues):
        sys.path.insert(0, str(ROOT))
        from scripts.fetch_football_data_org import (
            FetchError,
            sync_fixtures,
            sync_results,
            sync_standings,
        )

        for league in args.leagues:
            if league in standings_done:
                continue
            try:
                p = sync_standings(
                    league, api_key, out_dir=args.standings_dir, as_of=as_of
                )
                print(f"[standings] {league}: API snapshot -> {p}")
                standings_done.append(league)
            except FetchError as e:
                errors.append(f"API standings {league}: {e}")
        # Optional API pulls fill in whatever the CSV download cannot yet serve.
        for league in args.leagues:
            try:
                n = sync_fixtures(
                    league, api_key, fixtures_dir=args.fixtures_dir, replace=True
                )
                print(f"[fixtures] {league}: merged {n} scheduled fixture(s) from API")
            except FetchError as e:
                errors.append(f"API fixtures {league}: {e}")
            try:
                n = sync_results(league, api_key, data_dir=args.data_dir)
                print(f"[results] {league}: merged {n} finished result(s) from API")
            except FetchError as e:
                errors.append(f"API results {league}: {e}")
    if not args.standings_table and not use_api:
        print(
            "\n== Standings: nothing to fetch automatically =="
            "\n   Paste today's tables and re-run with "
            "'--standings-table table.txt' (repeatable), or set "
            "FOOTBALL_DATA_ORG_API_KEY for a fully automatic run."
        )

    # 4) Optional heavy validation / retraining.
    if args.backtest:
        for league in args.leagues:
            try:
                run_backtest(league, args.data_dir, args.reports_dir)
            except RuntimeError as e:
                errors.append(str(e))
    if args.retrain:
        prewarm_models(args.leagues, args.data_dir, args.data_dir.parent / "models")

    # Summary.
    print("\n" + "=" * 60)
    print("Refresh summary")
    print("=" * 60)
    print(
        f"  results/odds:  downloaded {download_summary['downloaded']}, "
        f"refreshed {download_summary['refreshed']}, "
        f"failed {download_summary['failed']}"
    )
    print(f"  fixtures pruned (played -> results): {pruned}")
    print(f"  standings imported: {len(standings_done)} league(s)")
    print(
        "  commit: git add data/ && git commit -m 'Weekly data refresh'\n"
        "  redeploy or restart the web app to pick up the new data."
    )
    if errors:
        print("\nProblems encountered:")
        for msg in errors:
            print(f"  - {msg}")
    failed_downloads = args.strict and download_summary["failed"] > 0
    return 1 if (errors or failed_downloads) else 0


if __name__ == "__main__":
    raise SystemExit(main())
