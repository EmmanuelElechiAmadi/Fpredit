"""Load the League Updates configuration and the canonical team-name map.

- ``config/league_updates.yaml`` drives the pipeline (sources, validation,
  retries, lock, ratings).
- ``config/team_names.yaml`` maps every source spelling to one canonical
  football-data.co.uk name per league.

Both files ship with defaults so the module works even if they are absent.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import yaml  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_UPDATE_CONFIG = CONFIG_DIR / "league_updates.yaml"
DEFAULT_TEAM_NAMES = CONFIG_DIR / "team_names.yaml"

# League key -> football-data.co.uk folder code (used across src/*).
LEAGUE_CODES = {"EPL": "E0", "LALIGA": "SP1", "SERIEA": "I1"}
# League key -> Understat league slug.
UNDERSTAT_LEAGUES = {"EPL": "EPL", "LALIGA": "La_liga", "SERIEA": "Serie_A"}


def _dict_to_ns(d: dict) -> SimpleNamespace:
    ns = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(ns, key, _dict_to_ns(value))
        else:
            setattr(ns, key, value)
    return ns


def _defaults() -> dict:
    return {
        "season": "auto",
        "leagues": {
            "EPL": {"football_data_code": "E0", "understat": "EPL"},
            "LALIGA": {"football_data_code": "SP1", "understat": "La_liga"},
            "SERIEA": {"football_data_code": "I1", "understat": "Serie_A"},
        },
        "sources": {
            "football_data_base": "https://www.football-data.co.uk/mmz4281",
            "fixtures_url": "https://www.football-data.co.uk/fixtures.csv",
            "understat": "enabled",
            "squad_news": "disabled",
        },
        "database": {"path": "data/updates.db"},
        "raw_snapshots_dir": "data/raw_snapshots",
        "reports_dir": "reports",
        "ratings": {"promoted_team_start": 1450.0, "season_regression": 0.25},
        "validation": {
            "max_missing_rate": 0.05,
            "odds_overround_range": [1.00, 1.20],
            "max_goals_per_team": 20,
            "max_shots": 80,
            "max_corners": 30,
            "max_fouls": 50,
            "max_cards": 15,
        },
        "retry": {"attempts": 4, "backoff_seconds": [5, 20, 60, 180]},
        "lock": {"path": "data/updates.lock", "stale_after_seconds": 1800},
    }


def load_update_config(path: Optional[str | Path] = None) -> SimpleNamespace:
    """Load league_updates.yaml as a SimpleNamespace (defaults when absent)."""
    p = Path(path) if path else DEFAULT_UPDATE_CONFIG
    if p.exists():
        with open(p) as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = {}
    merged = _defaults()
    for key, value in raw.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return _dict_to_ns(merged)


def load_team_names(path: Optional[str | Path] = None) -> dict[str, dict[str, str]]:
    """Load team_names.yaml -> {league: {source_spelling: canonical_name}}."""
    p = Path(path) if path else DEFAULT_TEAM_NAMES
    if not p.exists():
        return {}
    with open(p) as f:
        raw = yaml.safe_load(f) or {}
    return {league: dict(mapping) for league, mapping in raw.items() if mapping}


def canonicalise(
    league: str,
    name: str,
    team_map: dict[str, dict[str, str]],
) -> Optional[str]:
    """Map a source team name to its canonical name.

    Returns ``None`` when the team is unknown — callers must then HALT that
    league's update rather than guess.
    """
    mapping = team_map.get(league, {})
    key = " ".join((name or "").split())
    if not key:
        return None

    folded = {k.casefold(): v for k, v in mapping.items()}
    hit = folded.get(key.casefold())
    if hit is not None:
        return hit

    # Already-canonical spellings pass through without an identity entry.
    for canonical in set(mapping.values()):
        if canonical.casefold() == key.casefold():
            return canonical
    return None


def resolve_project_path(p: str) -> Path:
    """Resolve a config path relative to the project root when not absolute."""
    path = Path(p)
    return path if path.is_absolute() else (PROJECT_ROOT / path)
