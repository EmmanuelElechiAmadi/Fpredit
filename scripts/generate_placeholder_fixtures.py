"""Regenerate the placeholder 2026-27 fixture schedules under ``data/fixtures/``.

Each league's ``data/fixtures/<CODE>/2026-27.csv`` is a *placeholder* double
round-robin built from the shipped current-season roster
(``data/current_season_teams.json``) with the circle method
(:func:`src.fixtures.generate_round_robin`) — one matchweek per week from the
league's actual opening Saturday.

It is **not** the official fixture list: pairings are synthetic. Drop the
official football-data.co.uk CSV over the file as soon as it is available (the
loader understands their native layout and picks it up automatically), or use
the Fixtures view to edit individual rows. Matches whose dates have already
passed vanish from the *upcoming* view on their own — ``load_fixtures`` filters
by date — so past placeholder rounds never masquerade as predictions.

Usage::

    python3 scripts/generate_placeholder_fixtures.py            # all three leagues
    python3 scripts/generate_placeholder_fixtures.py --league EPL
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data_loader import LEAGUE_CODES
from src.fixtures import generate_round_robin, write_fixtures_batch

ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = ROOT / "data" / "fixtures"
ROSTER_PATH = ROOT / "data" / "current_season_teams.json"

# Opening Saturday of each league's 2026-27 season, anchored to the shipped
# results/standings (La Liga kicked off a week earlier than England and Italy).
SEASON_STARTS = {
    "EPL": pd.Timestamp("2026-08-22"),
    "LALIGA": pd.Timestamp("2026-08-15"),
    "SERIEA": pd.Timestamp("2026-08-22"),
}


def _validate(schedule: pd.DataFrame, n_teams: int) -> None:
    """Sanity-check a generated schedule before it overwrites shipped data."""
    expected = n_teams * (n_teams - 1)
    assert len(schedule) == expected, f"expected {expected} rows, got {len(schedule)}"
    assert schedule["matchweek"].nunique() == (n_teams - 1) * 2
    # Every club appears exactly once per matchweek (no bye weeks, no double-ups).
    appearances = pd.concat(
        [
            schedule[["matchweek", "home_team"]].rename(columns={"home_team": "team"}),
            schedule[["matchweek", "away_team"]].rename(columns={"away_team": "team"}),
        ]
    )
    counts = appearances.groupby(["matchweek", "team"]).size()
    assert (counts == 1).all(), "a team would play twice in one round"


def generate_league(league: str, rosters: dict) -> str:
    code = LEAGUE_CODES[league]
    teams = list(rosters[league])
    if not teams:
        raise SystemExit(f"No roster for {league} in {ROSTER_PATH.name}")
    start = SEASON_STARTS[league]

    schedule = generate_round_robin(teams, start_date=start)
    _validate(schedule, len(teams))

    path = FIXTURES_DIR / code / "2026-27.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)  # replace wholesale, never merge into the old one
    added = write_fixtures_batch(league, schedule, raw_dir=str(FIXTURES_DIR))
    n_mw = int(schedule["matchweek"].max())
    season = f"{start.year}-{(start.year + 1) % 100:02d}"
    print(
        f"[{league}] wrote {added} placeholder matches over {n_mw} matchweeks "
        f"({season}, MW1 {start.date()}) -> {path.relative_to(ROOT)}"
    )
    return str(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="(Re)generate placeholder round-robin fixtures for a league."
    )
    parser.add_argument(
        "--league",
        choices=list(LEAGUE_CODES),
        default=None,
        help="league to regenerate (default: all three)",
    )
    args = parser.parse_args(argv)

    try:
        rosters = json.loads(ROSTER_PATH.read_text())
    except (OSError, ValueError) as e:  # pragma: no cover - defensive
        raise SystemExit(f"Could not read {ROSTER_PATH}: {e}")

    leagues = [args.league] if args.league else list(SEASON_STARTS)
    for league in leagues:
        generate_league(league, rosters)

    print(
        "\nPlaceholder schedules written. These pairings are synthetic — replace "
        "each file with the official\nfixture list when football-data.co.uk posts "
        "it (see data/fixtures/README.md)."
    )


if __name__ == "__main__":
    main(sys.argv[1:])
