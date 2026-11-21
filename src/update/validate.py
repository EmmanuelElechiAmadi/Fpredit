"""Section 5 validation — every row is checked before entering main tables.

Rows that fail go to ``quarantine`` with a reason; the update continues with
the valid rows and flags the issue in the report.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Optional

import pandas as pd

_REQUIRED = ["date", "home", "away", "fthg", "ftag"]
_STAT_FIELDS = [
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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize(value):
    """Make a single value JSON-safe for quarantine storage."""
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _quarantine(row, reason: str, ingested_at: str) -> dict:
    raw = {k: _sanitize(v) for k, v in row.to_dict().items()}
    return {
        "league": row.get("league"),
        "season": row.get("season"),
        "date": _sanitize(row.get("date")),
        "home": row.get("home"),
        "away": row.get("away"),
        "reason": reason,
        "raw": json.dumps(raw, default=str),
        "ingested_at": ingested_at,
    }


def validate_matches(
    df: pd.DataFrame,
    cfg,
    today: Optional[date] = None,
    ingested_at: Optional[str] = None,
) -> tuple[pd.DataFrame, list[dict], dict[str, float]]:
    """Return (valid_df, quarantined, missing_rates)."""
    today = today or date.today()
    ingested_at = ingested_at or _now()
    valid: list[dict] = []
    quarantined: list[dict] = []
    seen_keys: set[tuple] = set()

    v = cfg.validation
    season_start = _season_start(df.iloc[0]["season"]) if len(df) else None

    for _, row in df.iterrows():
        if any(pd.isna(row.get(c)) for c in _REQUIRED):
            quarantined.append(_quarantine(row, "missing required field", ingested_at))
            continue

        date_val = row["date"]
        if pd.notna(date_val):
            d = (
                date_val.date()
                if hasattr(date_val, "date")
                else pd.Timestamp(date_val).date()
            )
            if d > today:
                quarantined.append(
                    _quarantine(row, "match date in the future", ingested_at)
                )
                continue
            if season_start and d < season_start:
                quarantined.append(
                    _quarantine(row, "match date before season start", ingested_at)
                )
                continue

        # Count sanity (negatives, shots-on-target, plausibility limits) first,
        # so a negative goal is reported as such rather than as a result mismatch.
        bad = _count_violation(row, v)
        if bad:
            quarantined.append(_quarantine(row, bad, ingested_at))
            continue

        fthg, ftag = int(row["fthg"]), int(row["ftag"])
        expected = "H" if fthg > ftag else ("D" if fthg == ftag else "A")
        if str(row.get("ftr")).strip() not in ("H", "D", "A"):
            quarantined.append(_quarantine(row, "missing/invalid result", ingested_at))
            continue
        if str(row["ftr"]).strip() != expected:
            quarantined.append(
                _quarantine(row, "result disagrees with goals", ingested_at)
            )
            continue

        key = (
            row["league"],
            row["season"],
            str(date_val.date()),
            row["home"],
            row["away"],
        )
        if key in seen_keys:
            quarantined.append(_quarantine(row, "duplicate match key", ingested_at))
            continue
        seen_keys.add(key)
        valid.append(row.to_dict())

    valid_df = pd.DataFrame(valid) if valid else pd.DataFrame(columns=df.columns)
    missing_rates = _missing_rates(valid_df, _REQUIRED + _STAT_FIELDS)
    return valid_df, quarantined, missing_rates


def _count_violation(row, v) -> Optional[str]:
    """Return a reason string when a count sanity check fails, else None."""
    for field in ["fthg", "ftag"] + _STAT_FIELDS:
        val = row.get(field)
        if pd.notna(val) and int(val) < 0:
            return f"{field} negative"
    pairs = [("hst", "hs"), ("ast", "as"), ("hthg", "fthg"), ("htag", "ftag")]
    for smaller, larger in pairs:
        if (
            pd.notna(row.get(smaller))
            and pd.notna(row.get(larger))
            and int(row[smaller]) > int(row[larger])
        ):
            return f"{smaller} > {larger}"
    limits = [
        ("fthg", v.max_goals_per_team),
        ("ftag", v.max_goals_per_team),
        ("hs", v.max_shots),
        ("as", v.max_shots),
        ("hc", v.max_corners),
        ("ac", v.max_corners),
        ("hf", v.max_fouls),
        ("af", v.max_fouls),
        ("hy", v.max_cards),
        ("ay", v.max_cards),
    ]
    for field, limit in limits:
        if pd.notna(row.get(field)) and int(row[field]) > limit:
            return f"{field} exceeds limit {limit}"
    return None


def _missing_rates(df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    if df.empty:
        return {c: 1.0 for c in columns}
    return {c: float(df[c].isna().mean()) for c in columns if c in df.columns}


def _season_start(season: str) -> Optional[date]:
    try:
        year = int(str(season).split("-")[0])
        return date(year, 7, 1)
    except (ValueError, IndexError):
        return None


def validate_odds(
    odds_records: list[dict],
    overround_range=(1.00, 1.20),
) -> tuple[list[dict], list[dict]]:
    """Split odds into valid and rejected (price/overround sanity)."""
    valid, rejected = [], []
    for rec in odds_records:
        if rec.get("market") == "1X2":
            h, d, a = rec["home"], rec["draw"], rec["away"]
            if any(p is None or p <= 1.0 for p in (h, d, a)):
                rejected.append({**rec, "reason": "odds <= 1.00"})
                continue
            overround = 1 / h + 1 / d + 1 / a
            if not (overround_range[0] <= overround <= overround_range[1]):
                rejected.append(
                    {**rec, "reason": f"overround {overround:.3f} out of range"}
                )
                continue
        valid.append(rec)
    return valid, rejected


def validate_xg(
    home_xg: float, away_xg: float, lo: float = 0.0, hi: float = 8.0
) -> bool:
    """True when both xG values are plausible (spec range 0–8)."""
    if home_xg is None or away_xg is None:
        return False
    return lo <= float(home_xg) <= hi and lo <= float(away_xg) <= hi
