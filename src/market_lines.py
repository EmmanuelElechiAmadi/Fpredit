"""Derivative market pricing from a Dixon-Coles score matrix.

Item C.8: the score matrix already exists — expose clean pricing for the
derivative markets that professional bettors actually trade:

* **Over/Under total goals** at every half-goal line (0.5, 1.5, ...),
* **Asian Handicap** on the home team at quarter-ball precision (bookies split a
  -0.25 line into -0.0 / -0.5 halves and average the two fair prices).

Everything is "fair" pricing straight from the matrix probabilities (no
margin). The UI/app layers may add their own margin/vig when they want a
bookmaker-style price.
"""

from __future__ import annotations

import numpy as np

DEFAULT_OU_LINES = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
DEFAULT_AH_LINES = [
    -2.5,
    -2.0,
    -1.5,
    -1.0,
    -0.75,
    -0.5,
    -0.25,
    0.0,
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    2.5,
]


def _fair_odds(prob: float) -> float | None:
    """Decimal odds with no margin (None when the probability is ~0)."""
    if not np.isfinite(prob) or prob <= 1e-9:
        return None
    return float(round(1.0 / prob, 2))


def total_goals_distribution(mat: np.ndarray) -> dict[int, float]:
    """P(total goals == t) for every reachable total from the score matrix."""
    totals: dict[int, float] = {}
    n = mat.shape[0]
    for i in range(n):
        for j in range(n):
            totals[i + j] = totals.get(i + j, 0.0) + float(mat[i, j])
    return totals


def over_under_prices(
    mat: np.ndarray,
    lines: list[float] | None = None,
) -> list[dict]:
    """Priced Over/Under markets for every total-goals line.

    ``mat`` is a square score matrix where ``mat[i, j]`` = P(home scores i and
    away scores j). A match clears the *over* when ``i + j > line``.
    """
    lines = list(lines if lines is not None else DEFAULT_OU_LINES)
    dist = total_goals_distribution(mat)
    out = []
    for line in lines:
        over = sum(p for t, p in dist.items() if t > line)
        under = sum(p for t, p in dist.items() if t < line)
        # Exact half-line -> no push possible; still normalise against float error.
        push = 1.0 - over - under
        total = over + under + push
        if total > 0:
            over, under = over / total, under / total
        out.append(
            {
                "line": float(line),
                "over": round(over, 4),
                "under": round(under, 4),
                "over_odds": _fair_odds(over),
                "under_odds": _fair_odds(under),
            }
        )
    return out


def _is_quarter(handicap: float) -> bool:
    """True for quarter-ball lines (…, ±0.25, ±0.75, ±1.25, …)."""
    return abs((handicap * 2) % 1.0 - 0.5) < 1e-9


def _line_probs(mat: np.ndarray, line: float) -> dict[str, float]:
    """P(home covers | push | away covers) for a single Asian line.

    Home receives the handicap ``line`` (positive = head start). Outcome uses
    the net score ``home_goals + line - away_goals``: positive = home covers,
    zero = push (stake returned), negative = away covers. Works for whole and
    half lines alike.
    """
    home = away = push = 0.0
    n = mat.shape[0]
    for i in range(n):
        for j in range(n):
            net = i + line - j
            if net > 0:
                home += float(mat[i, j])
            elif net < 0:
                away += float(mat[i, j])
            else:
                push += float(mat[i, j])
    total = home + away + push
    if total <= 0:  # degenerate / empty matrix — no probability mass
        return {"home": 0.0, "push": 0.0, "away": 0.0}
    return {
        "home": home / total,
        "push": push / total,
        "away": away / total,
    }


def asian_handicap_prices(
    mat: np.ndarray,
    handicaps: list[float] | None = None,
) -> list[dict]:
    """Priced Asian Handicap markets (home team receives the handicap).

    Whole and half lines price straight off the matrix. Quarter lines are the
    50/50 average of the two adjacent lines (bookmakers stake half on each), so
    a -0.25 line is half at -0.0 (push on a draw) and half at -0.5.
    """
    handicaps = list(handicaps if handicaps is not None else DEFAULT_AH_LINES)
    out = []
    for hcap in handicaps:
        if _is_quarter(hcap):
            lo = _line_probs(mat, hcap - 0.25)
            hi = _line_probs(mat, hcap + 0.25)
            home = 0.5 * (lo["home"] + hi["home"])
            away = 0.5 * (lo["away"] + hi["away"])
            push = 0.5 * (lo["push"] + hi["push"])
        else:
            p = _line_probs(mat, hcap)
            home, away, push = p["home"], p["away"], p["push"]
        out.append(
            {
                "handicap": float(hcap),
                "home": round(home, 4),
                "away": round(away, 4),
                "push": round(push, 4),
                "home_odds": _fair_odds(home),
                "away_odds": _fair_odds(away),
            }
        )
    return out


def market_lines_from_matrix(
    mat: np.ndarray,
    ou_lines: list[float] | None = None,
    ah_handicaps: list[float] | None = None,
) -> dict:
    """Bundle Over/Under + Asian Handicap prices for a score matrix."""
    return {
        "over_under": over_under_prices(mat, ou_lines),
        "asian_handicap": asian_handicap_prices(mat, ah_handicaps),
    }
