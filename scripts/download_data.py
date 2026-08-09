#!/usr/bin/env python3
"""
Download historical football CSV data from football-data.co.uk.

Usage:
    python scripts/download_data.py --league EPL --seasons 5
    python scripts/download_data.py --all --seasons 3 --data-dir data/raw

Downloads season files into data/raw/<LEAGUE_CODE>/<season>.csv
No API key required. Updated weekly during the season.
"""
import argparse
import csv
import io
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

LEAGUE_URLS = {
    "EPL": {
        "name": "English Premier League",
        "code": "E0",
        "url": "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv",
    },
    "LALIGA": {
        "name": "Spanish La Liga",
        "code": "SP1",
        "url": "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv",
    },
    "SERIEA": {
        "name": "Italian Serie A",
        "code": "I1",
        "url": "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv",
    },
}


# football-data.co.uk uses a 2-digit short code for seasons:
# 2024-25 -> "2425", 2023-24 -> "2324", etc.
def _season_code(season_start_year: int) -> str:
    return f"{season_start_year % 100:02d}{(season_start_year + 1) % 100:02d}"


def _validate_downloaded_csv(out_path: Path, expected_code: str) -> bool:
    """Sanity-check a freshly downloaded CSV.

    football-data.co.uk sometimes serves a *different* competition's file when a
    season path is wrong (e.g. SP1/2627 returned Scottish Championship SC1 rows).
    This would silently corrupt training data, so we verify the ``Div`` column —
    when present — contains (or is dominated by) the expected league code.

    Returns True if the file looks valid, False if it should be discarded.
    """
    try:
        raw = out_path.read_bytes()
        # Strip UTF-8 BOM if present (older files use it)
        if raw.startswith(b"\xef\xbb\xbf"):
            raw = raw[3:]
        text = raw.decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        fieldnames = {f.strip() for f in (reader.fieldnames or [])}
        if "Div" not in fieldnames and "div" not in fieldnames:
            # No Div column -> we can't validate; treat as OK but warn.
            print("  [WARN] no Div column; cannot verify competition identity")
            return True

        div_col = "Div" if "Div" in fieldnames else "div"
        codes_seen: dict[str, int] = {}
        n_rows = 0
        for row in reader:
            div = (row.get(div_col) or "").strip()
            if not div:
                continue
            codes_seen[div] = codes_seen.get(div, 0) + 1
            n_rows += 1
            if n_rows >= 200:
                break

        if not codes_seen:
            print("  [WARN] Div column exists but is empty; cannot verify")
            return True

        top_code, top_count = max(codes_seen.items(), key=lambda kv: kv[1])
        total = sum(codes_seen.values())
        if top_code == expected_code:
            return True

        print(
            f"  [ERROR] file contains competition '{top_code}' "
            f"({top_count}/{total} rows), expected '{expected_code}'"
        )
        return False
    except Exception as e:
        print(f"  [WARN] could not validate {out_path.name}: {e}")
        return False


def download_league(
    league: str, n_seasons: int, data_dir: str, current_year: Optional[int] = None
):
    if current_year is None:
        from datetime import date

        today = date.today()
        current_year = today.year if today.month >= 7 else today.year - 1

    info = LEAGUE_URLS[league]
    dest = Path(data_dir) / info["code"]
    dest.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for i in range(n_seasons):
        season_start = current_year - i
        code = _season_code(season_start)
        url = info["url"].format(season=code, code=info["code"])
        out_path = dest / f"{season_start}-{season_start + 1 - 2000:02d}.csv"

        if out_path.exists():
            print(f"[SKIP] {out_path.name} already exists")
            continue

        print(
            f"[DL] {info['name']} {season_start}-{season_start + 1 - 2000:02d} ... ",
            end="",
            flush=True,
        )
        try:
            with urlopen(url, timeout=30) as response:
                content = response.read()
            with open(out_path, "wb") as f:
                f.write(content)
            print(f"OK ({len(content):,} bytes)")

            if not _validate_downloaded_csv(out_path, info["code"]):
                out_path.unlink(missing_ok=True)
                print(f"  Deleted {out_path.name}: wrong competition data discarded")
                continue

            downloaded += 1
        except HTTPError as e:
            if e.code == 404:
                print("NOT FOUND (season may be too old or not yet available)")
            else:
                print(f"HTTP Error {e.code}: {e.reason}")
        except URLError as e:
            print(f"Network error: {e.reason}")
        except Exception as e:
            print(f"Unexpected error: {e}")

    if downloaded == 0:
        print(f"\nNo new files downloaded for {league}. Data is already up-to-date.")
    else:
        print(f"\nDownloaded {downloaded} season file(s) to {dest.resolve()}")


def main():
    ap = argparse.ArgumentParser(
        description="Download football-data.co.uk historical CSVs"
    )
    ap.add_argument(
        "--league", choices=list(LEAGUE_URLS.keys()), help="Specific league to download"
    )
    ap.add_argument("--all", action="store_true", help="Download all leagues")
    ap.add_argument(
        "--seasons",
        type=int,
        default=5,
        help="Number of past seasons to download (default: 5)",
    )
    ap.add_argument(
        "--data-dir",
        default="data/raw",
        help="Directory to save CSV files (default: data/raw)",
    )
    args = ap.parse_args()

    if not args.league and not args.all:
        ap.error("Specify --league or --all")

    leagues = list(LEAGUE_URLS.keys()) if args.all else [args.league]
    for league in leagues:
        download_league(league, args.seasons, args.data_dir)
        print()


if __name__ == "__main__":
    main()
