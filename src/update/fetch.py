"""Download raw sources with retry/backoff and save timestamped snapshots.

Every raw download is written to ``raw_snapshots_dir/<timestamp>/`` *before*
parsing so any run can be replayed. A failed download raises ``FetchError``;
callers keep existing data and mark the source stale instead of corrupting it.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


class FetchError(RuntimeError):
    """A source could not be fetched after retries."""


def season_start_year(today: Optional[date] = None) -> int:
    """European-season start year for a date: Nov 2026 -> 2026."""
    d = today or date.today()
    return d.year if d.month >= 7 else d.year - 1


def season_label(today: Optional[date] = None) -> str:
    """Season label for a date: Nov 2026 -> '2026-27'."""
    year = season_start_year(today)
    return f"{year}-{(year + 1) % 100:02d}"


def season_short_code(today: Optional[date] = None) -> str:
    """football-data.co.uk 2-digit season code: 2026-27 -> '2627'."""
    year = season_start_year(today)
    return f"{year % 100:02d}{(year + 1) % 100:02d}"


def _snapshot(url: str, content: bytes, snapshots_dir: Path, label: str) -> Path:
    """Save raw bytes to a timestamped snapshot path and return it."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = snapshots_dir / ts
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{label}.csv"
    path.write_bytes(content)
    return path


def download_with_retry(
    url: str,
    attempts: int = 4,
    backoff_seconds: Optional[list] = None,
    timeout: int = 30,
) -> bytes:
    """GET a URL, retrying with exponential backoff on failure."""
    backoff = list(backoff_seconds) if backoff_seconds else [5, 20, 60, 180]
    last_error: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            with urlopen(url, timeout=timeout) as resp:
                return resp.read()
        except (HTTPError, URLError, TimeoutError, OSError) as e:
            last_error = e
            if attempt + 1 < attempts:
                delay = backoff[min(attempt, len(backoff) - 1)]
                time.sleep(delay)
    raise FetchError(f"Failed to fetch {url} after {attempts} attempts: {last_error}")


def fetch_league_csv(
    league: str,
    football_data_code: str,
    season_code: str,
    base_url: str,
    dest_dir: Path,
    snapshots_dir: Path,
) -> Path:
    """Download one football-data.co.uk season file into dest/<CODE>/<season>.csv.

    The current season's file is re-downloaded each run (new matchweeks). On a
    404 (season not published yet) it raises FetchError so the caller marks the
    source stale rather than writing an empty file.
    """
    url = f"{base_url}/{season_code}/{football_data_code}.csv"
    content = download_with_retry(url)
    if not content.strip():
        raise FetchError(f"Empty body from {url}")

    dest = (
        Path(dest_dir)
        / football_data_code
        / f"20{season_code[:2]}-{season_code[2:]}.csv"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    _snapshot(url, content, snapshots_dir, f"{football_data_code}_{season_code}")
    return dest


def fetch_fixtures_csv(
    fixtures_url: str,
    dest_dir: Path,
    snapshots_dir: Path,
) -> Path:
    """Download the fixtures.csv file into dest/."""
    content = download_with_retry(fixtures_url)
    if not content.strip():
        raise FetchError(f"Empty body from {fixtures_url}")
    dest = Path(dest_dir) / "fixtures.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    _snapshot(fixtures_url, content, snapshots_dir, "fixtures")
    return dest


def fetch_understat_xg(
    league: str,
    seasons: list[str],
    data_dir: Path,
) -> Optional[Path]:
    """Scrape + cache Understat match xG for ``league`` (reuses src.xg_loader).

    Returns the cached directory, or None when Understat is unavailable.
    """
    from src.xg_loader import fetch_league_xg

    try:
        fetch_league_xg(league, seasons=seasons, data_dir=str(data_dir))
        return Path(data_dir) / league
    except Exception:  # network / parse failures: mark stale, keep existing
        return None
