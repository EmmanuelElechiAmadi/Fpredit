"""
Scrape Understat match-level xG for a league and cache per-season CSVs.

Understat (understat.com) provides free match-level xG for the big five
European leagues back to 2014-15. The scraper hits the same JSON endpoint the
league page uses and caches the result under data/xg/<LEAGUE>/<season>.csv so
the prediction/backtest pipeline runs offline afterwards.

Usage:
    python scripts/scrape_understat.py --league EPL --seasons 5
    python scripts/scrape_understat.py --league SERIEA --from 2014 --to 2026
    python scripts/scrape_understat.py --league LALIGA --refresh
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

# Make `src` importable when running directly: `python scripts/scrape_understat.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xg_loader import _season_strs, fetch_league_xg

LEAGUES = ["EPL", "LALIGA", "SERIEA"]


def main():
    ap = argparse.ArgumentParser(description="Scrape Understat match xG to data/xg")
    ap.add_argument("--league", default="EPL", choices=LEAGUES)
    ap.add_argument(
        "--seasons",
        type=int,
        default=5,
        help="Number of most recent seasons to scrape (default 5)",
    )
    ap.add_argument(
        "--from",
        dest="from_year",
        type=int,
        default=None,
        help="First season start year (e.g. 2014 for 2014-15). Overrides --seasons.",
    )
    ap.add_argument("--to", dest="to_year", type=int, default=None)
    ap.add_argument("--data-dir", default="data/xg")
    ap.add_argument(
        "--refresh",
        action="store_true",
        help="Re-scrape seasons even if a cached CSV already exists",
    )
    args = ap.parse_args()

    current_year = datetime.datetime.now().year
    # Most recent *complete* season starts in the previous calendar year.
    last_complete = current_year - 1
    if args.from_year is not None:
        end = args.to_year if args.to_year is not None else last_complete
        seasons = _season_strs(args.from_year, end)
    else:
        end = last_complete
        seasons = _season_strs(end - args.seasons + 1, end)

    print(f"Seasons to scrape for {args.league}: {seasons[0]} .. {seasons[-1]}")
    df = fetch_league_xg(
        args.league, seasons=seasons, data_dir=args.data_dir, refresh=args.refresh
    )
    print(f"Done: {len(df)} matches cached under {args.data_dir}/{args.league}/")


if __name__ == "__main__":
    main()
