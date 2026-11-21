#!/usr/bin/env python3
"""
Import a posted league table (standings) into data/standings/<CODE>/<season>.json.

The web app's Dashboard/Team-Intelligence standings prefer these authoritative
snapshots for a league's current season over the table it computes from the
(usually lagging) results CSV. Paste the league table you see on the website:

    python3 scripts/import_standings.py --input /tmp/epl.txt
    python3 scripts/import_standings.py --input /tmp/laliga.txt --as-of 2026-09-06
    cat /tmp/epl.txt | python3 scripts/import_standings.py

The league is auto-detected from the table's title ("Premier League",
"Spanish La Liga", "Serie A"). Three row layouts are accepted:

1. One row per line::

       1 Manchester City 3 3 0 0 7 2 +5 9

2. The copy-pasted website layout, where each row is split across three
   lines (position / team name / stats)::

       1
       Manchester City
       3  3  0  0  7  2  +5  9

3. The *vertical* labelled layout (common on the official Serie A / mobile
   sites): each club is a small block with its position, name, and stats on
   their own lines, stats written as ``Label: value`` in Italian or English::

       1
       Roma
       Giocate: 3
       Vittorie: 3
       Pareggi: 0
       Sconfitte: 0
       Gol fatti: 7
       Gol subiti: 2
       Differenza reti: +5
       Punti: 9

   Labels may be English ("Played", "Won", "Drawn", ...) or Italian
   ("Giocate", "Vittorie", "Pareggi", ...), in any order; position/team lines
   can also be combined ("1 Roma").

Team names are mapped to football-data.co.uk spellings (e.g. "Manchester
City" -> "Man City", "Athletic Club" -> "Ath Bilbao") so the standings join
onto the rest of the dataset.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Iterable, Optional

# data/standings/<LEAGUE_CODE>/<season>.json
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "standings"

LEAGUE_CODES = {"EPL": "E0", "LALIGA": "SP1", "SERIEA": "I1"}

# Table title substrings that identify the league when the user pastes the
# official website's table (BBC-style "Last updated: ..." pages).
_TITLE_ALIASES = {
    "EPL": ["premier league"],
    "LALIGA": ["la liga"],
    "SERIEA": ["serie a"],
}

# Display names (as published on the league-table website) -> football-data
# spellings used everywhere in this project.
TEAM_ALIASES = {
    "EPL": {
        "Arsenal": "Arsenal",
        "Aston Villa": "Aston Villa",
        "Bournemouth": "Bournemouth",
        "Brentford": "Brentford",
        "Brighton and Hove Albion": "Brighton",
        "Brighton": "Brighton",
        "Chelsea": "Chelsea",
        "Coventry City": "Coventry",
        "Coventry": "Coventry",
        "Crystal Palace": "Crystal Palace",
        "Everton": "Everton",
        "Fulham": "Fulham",
        "Hull City": "Hull",
        "Hull": "Hull",
        "Ipswich Town": "Ipswich Town",
        "Leeds United": "Leeds",
        "Leeds": "Leeds",
        "Liverpool": "Liverpool",
        "Manchester City": "Man City",
        "Manchester United": "Man United",
        "Newcastle United": "Newcastle",
        "Newcastle": "Newcastle",
        "Nottingham Forest": "Nott'm Forest",
        "Sunderland": "Sunderland",
        "Tottenham Hotspur": "Tottenham",
        "Tottenham": "Tottenham",
        "West Ham": "West Ham",
        "West Ham United": "West Ham",
        "Wolves": "Wolves",
        "Wolverhampton Wanderers": "Wolves",
    },
    "LALIGA": {
        "Alaves": "Alaves",
        "Athletic Bilbao": "Ath Bilbao",
        "Athletic Club": "Ath Bilbao",
        "Atletico Madrid": "Ath Madrid",
        "Atlético Madrid": "Ath Madrid",
        "Barcelona": "Barcelona",
        "Celta Vigo": "Celta",
        "Celta": "Celta",
        "Deportivo La Coruna": "Dep. A Coruna",
        "Deportivo La Coruña": "Dep. A Coruna",
        "Elche": "Elche",
        "Espanyol": "Espanol",
        "Español": "Espanol",
        "Getafe": "Getafe",
        "Girona": "Girona",
        "Las Palmas": "Las Palmas",
        "Leganes": "Leganes",
        "Levante": "Levante",
        "Malaga": "Malaga",
        "Mallorca": "Mallorca",
        "Osasuna": "Osasuna",
        "Racing Santander": "Santander",
        "Rayo Vallecano": "Vallecano",
        "Real Betis": "Betis",
        "Real Madrid": "Real Madrid",
        "Real Sociedad": "Sociedad",
        "Sevilla": "Sevilla",
        "Valencia": "Valencia",
        "Valladolid": "Valladolid",
        "Villarreal": "Villarreal",
    },
    "SERIEA": {
        "AS Roma": "Roma",
        "Atalanta": "Atalanta",
        "Bologna": "Bologna",
        "Cagliari": "Cagliari",
        "Como": "Como",
        "Empoli": "Empoli",
        "Fiorentina": "Fiorentina",
        "Frosinone": "Frosinone",
        "Genoa": "Genoa",
        "Hellas Verona": "Verona",
        "Verona": "Verona",
        "Inter": "Inter",
        "Internazionale": "Inter",
        "Inter Milan": "Inter",
        "Juventus": "Juventus",
        "Lazio": "Lazio",
        "Lecce": "Lecce",
        "Milan": "Milan",
        "AC Milan": "Milan",
        "Monza": "Monza",
        "Napoli": "Napoli",
        "Parma": "Parma",
        "Roma": "Roma",
        "Sassuolo": "Sassuolo",
        "Torino": "Torino",
        "Udinese": "Udinese",
        "Venezia": "Venezia",
    },
}

# position, team, P, W, D, L, GF, GA, GD, Pts — all on one line.
_ONE_LINE_ROW = re.compile(
    r"^(\d{1,2})\s+(.+?)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([+-]?\d+)\s+(\d+)$"
)
# A pure stats row from the split layout: P W D L GF GA GD Pts.
_STATS_ROW = re.compile(
    r"^(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([+-]?\d+)\s+(\d+)$"
)


def detect_league(text: str) -> str:
    """Identify the league from the table title in the pasted text."""
    low = text.lower()
    for league, aliases in _TITLE_ALIASES.items():
        for alias in aliases:
            if alias in low:
                return league
    raise ValueError(
        "Could not detect the league — include the table title (e.g. "
        "'Premier League') or pass --league EPL|LALIGA|SERIEA."
    )


def season_for_date(d: date) -> str:
    """Northern-hemisphere football season label for a date (Sep 2026 -> 2026-27)."""
    year = d.year if d.month >= 7 else d.year - 1
    return f"{year}-{(year + 1) % 100:02d}"


def _canonical_name(league: str, display: str) -> str:
    """Map a website display name to the football-data spelling, passing through
    already-canonical names unchanged."""
    key = display.strip()
    mapping = TEAM_ALIASES.get(league, {})
    if key in mapping:
        return mapping[key]
    # Case/whitespace tolerant fallback.
    folded = {k.casefold(): v for k, v in mapping.items()}
    hit = folded.get(key.casefold())
    if hit is not None:
        return hit
    return key  # already canonical (or unknown — kept as-is)


# Stat labels in the "vertical" layout, English + Italian. Keys are the
# canonical row fields; each maps to the label words that may prefix a value
# (labels are matched longest-first so "goals against" beats "goals").
_STAT_LABELS: dict[str, list[str]] = {
    "position": ["posizione", "position", "pos"],
    "played": ["partite giocate", "giocate", "partite", "played"],
    "won": ["vittorie", "vinte", "vittoria", "won", "wins"],
    "draw": ["pareggi", "pareggiate", "pareggio", "draws", "drawn"],
    "lost": ["sconfitte", "perse", "sconfitta", "losses", "lost"],
    "gf": ["gol fatti", "reti fatte", "goals for", "goals scored"],
    "ga": ["gol subiti", "reti subite", "goals against", "goals conceded"],
    "gd": ["differenza reti", "differenza goal", "goal difference", "goal diff"],
    "points": ["punti", "punti totali", "pts", "points"],
}
# Synonym -> canonical field, built once (longest synonym first).
_LABEL_TO_FIELD: dict[str, str] = {}
for _field, _syns in _STAT_LABELS.items():
    for _syn in _syns:
        _LABEL_TO_FIELD.setdefault(_syn.casefold(), _field)
_LABEL_RE = re.compile(
    r"(?<![A-Za-zÀ-ÿ])(?:"
    + "|".join(
        f"({re.escape(s)})" for s in sorted(_LABEL_TO_FIELD, key=len, reverse=True)
    )
    + r")\s*[:=]?\s*([+-]?\d+)",
    re.IGNORECASE,
)
# A lone position number at the start of a club's block ("1", "1.", "1)").
_POS_LINE = re.compile(r"^\s*(\d{1,2})\s*[.)]?\s*$")
# A position + name on one line ("1 Roma", "1) Roma").
_POS_NAME_LINE = re.compile(r"^\s*(\d{1,2})\s*[.)]?\s+(.+?)\s*$")
_NAME_LINE = re.compile(r"^[^\d]*[A-Za-zÀ-ÿ][^\d]*$")


def _label_pairs(line: str) -> dict[str, int]:
    """All ``Label: value`` pairs on a line, keyed by canonical stat field."""
    pairs: dict[str, int] = {}
    for m in _LABEL_RE.finditer(line):
        syn = next((g for g in m.groups() if g), "")
        num_index = m.lastindex if m.lastindex is not None else len(m.groups())
        pairs[_LABEL_TO_FIELD[syn.casefold()]] = int(m.group(num_index))
    return pairs


def _parse_vertical(text: str, league: str) -> Optional[list[dict]]:
    """Parse the vertical labelled standings layout (Serie A / mobile sites).

    Each club is a small block: a position number and a name line, followed by
    ``Label: value`` stat lines (Italian or English, any order). Position/team
    may share a line ("1 Roma"). Returns ``None`` when ``text`` does not look
    like this layout (so the other parsers get a chance).
    """
    lines = [" ".join(ln.split()) for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]

    pair_count = sum(len(_label_pairs(ln)) for ln in lines)
    if pair_count < 4:  # not enough labelled stats -> not the vertical layout
        return None

    rows: list[dict] = []
    pos: Optional[int] = None
    name: Optional[str] = None
    stats: dict[str, int] = {}

    _REQUIRED_STATS = {"played", "won", "draw", "lost", "gf", "ga", "points"}

    def flush() -> None:
        """Close the current club block once it has a name + core stats."""
        nonlocal pos, name, stats
        if name and _REQUIRED_STATS.issubset(stats):
            rows.append(
                {
                    "position": pos if pos is not None else len(rows) + 1,
                    "team": _canonical_name(league, name),
                    "played": stats["played"],
                    "won": stats["won"],
                    "draw": stats["draw"],
                    "lost": stats["lost"],
                    "gf": stats["gf"],
                    "ga": stats["ga"],
                    "goal_diff": stats.get("gd", stats["gf"] - stats["ga"]),
                    "points": stats["points"],
                }
            )
        pos = None
        name = None
        stats = {}

    for line in lines:
        pairs = _label_pairs(line)
        if pairs:
            stats.update(pairs)
            continue
        m_pos_name = _POS_NAME_LINE.match(line)
        if m_pos_name:
            flush()
            pos = int(m_pos_name.group(1))
            name = m_pos_name.group(2).strip()
            continue
        m_pos = _POS_LINE.match(line)
        if m_pos:
            # A fresh position number always starts a new club block.
            flush()
            pos = int(m_pos.group(1))
            continue
        if _NAME_LINE.match(line):
            if name is None and pos is not None:
                name = line
            elif name is not None and stats and _REQUIRED_STATS.issubset(stats):
                # Previous block is complete and a bare name line follows ->
                # new club without an explicit position.
                flush()
                name = line
            elif name is None and pos is None and not stats and not rows:
                pass  # header/title line above the first club
    flush()

    if not rows:
        raise ValueError(
            "Detected a labelled/vertical table but could not parse any clubs — "
            "expected position + team lines followed by 'Label: value' stat "
            "lines (e.g. 'Giocate: 3', 'Punti: 9')."
        )
    return rows


def parse_table(text: str, league: str) -> list[dict]:
    """Parse the pasted league table into standings rows (in table order).

    Accepts both the split copy-paste layout (position line, name line, stats
    line) and a single-line-per-row layout.
    """
    lines = [" ".join(ln.split()) for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]

    vertical = _parse_vertical(text, league)
    if vertical is not None:
        return vertical

    rows: list[dict] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _ONE_LINE_ROW.match(line)
        if m:
            pos, name, pl, w, d, los, gf, ga, gd, pts = m.groups()
            rows.append(
                {
                    "position": int(pos),
                    "team": _canonical_name(league, name),
                    "played": int(pl),
                    "won": int(w),
                    "draw": int(d),
                    "lost": int(los),
                    "gf": int(gf),
                    "ga": int(ga),
                    "goal_diff": int(gd),
                    "points": int(pts),
                }
            )
            i += 1
            continue

        m = _STATS_ROW.match(line)
        if m:
            # Split layout: previous non-consumed lines hold the position and
            # the team name (one per line, in either order).
            pl, w, d, los, gf, ga, gd, pts = m.groups()
            row_pos: Optional[int] = None
            team_name: Optional[str] = None
            for j in (i - 1, i - 2, i - 3):
                if j < 0:
                    break
                prev = lines[j]
                if team_name is None and prev and prev[0].isalpha():
                    team_name = prev
                elif row_pos is None and prev.isdigit():
                    row_pos = int(prev)
            if team_name is None:
                raise ValueError(
                    f"Could not find the team name before stats row: {line!r}"
                )
            if row_pos is None:
                row_pos = len(rows) + 1
            rows.append(
                {
                    "position": row_pos,
                    "team": _canonical_name(league, team_name),
                    "played": int(pl),
                    "won": int(w),
                    "draw": int(d),
                    "lost": int(los),
                    "gf": int(gf),
                    "ga": int(ga),
                    "goal_diff": int(gd),
                    "points": int(pts),
                }
            )
            i += 1
            continue

        # Header / title / blank lines are skipped.
        i += 1

    if not rows:
        raise ValueError("No standings rows parsed — check the pasted table format.")

    # Sanity: P = W + D + L and Pts = 3W + D.
    for r in rows:
        if r["played"] != r["won"] + r["draw"] + r["lost"]:
            raise ValueError(
                f"Row for {r['team']}: played ({r['played']}) != "
                f"W + D + L ({r['won']} + {r['draw']} + {r['lost']})."
            )
        if r["points"] != 3 * r["won"] + r["draw"]:
            raise ValueError(
                f"Row for {r['team']}: points ({r['points']}) != 3*W + D "
                f"({3 * r['won']} + {r['draw']})."
            )
    return rows


def write_snapshot(
    league: str,
    rows: list[dict],
    as_of: date,
    season: str,
    out_dir: Path = DEFAULT_OUT_DIR,
    source_override: Optional[str] = None,
) -> Path:
    """Write data/standings/<CODE>/<season>.json (creating folders as needed)."""
    code = LEAGUE_CODES[league]
    folder = out_dir / code
    folder.mkdir(parents=True, exist_ok=True)
    title = {
        "EPL": "Premier League",
        "LALIGA": "Spanish La Liga",
        "SERIEA": "Serie A",
    }[league]
    source = source_override or f"{title} table (football-data.co.uk {code}, {season})"
    payload = {
        "league": league,
        "season": season,
        "as_of": as_of.isoformat(),
        "source": source,
        "rows": rows,
    }
    path = folder / f"{season}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        help="Path to a text file with the pasted league table "
        "(default: read from stdin).",
    )
    ap.add_argument(
        "--league", choices=sorted(LEAGUE_CODES), help="Override auto-detect."
    )
    ap.add_argument(
        "--as-of", type=_parse_date, help="Table date (YYYY-MM-DD). Default: today."
    )
    ap.add_argument(
        "--season", help="Season label (YYYY-YY). Default: derived from --as-of."
    )
    ap.add_argument(
        "--source",
        help="Override the 'source' string stored in the snapshot file.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR}).",
    )
    args = ap.parse_args(list(argv) if argv else None)

    text = Path(args.input).read_text() if args.input else sys.stdin.read()
    league = args.league or detect_league(text)
    rows = parse_table(text, league)

    as_of = args.as_of or date.today()
    season = args.season or season_for_date(as_of)

    seen = [r["team"] for r in rows]
    if len(seen) != len(set(seen)):
        raise ValueError("Duplicate team in the parsed table — check the paste.")
    if sorted(seen) != sorted(set(seen)):
        raise ValueError("Standings rows are out of order or duplicated.")

    path = write_snapshot(
        league, rows, as_of, season, out_dir=args.out_dir, source_override=args.source
    )
    print(f"Wrote {path} ({len(rows)} teams, as of {as_of.isoformat()}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
