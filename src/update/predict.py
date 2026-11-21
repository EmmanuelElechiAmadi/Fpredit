"""Generate predictions for upcoming fixtures.

Uses a blend of Dixon-Coles and Elo (both reused from ``src/``) and compares
the model probability to the market-implied probability to report an edge. If a
fitted ensemble is available it can be swapped in for the final probabilities.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.dixon_coles import DixonColes
from src.elo import EloEngine
from src.market import implied_probabilities


def _odds_from_fixture(row) -> Optional[tuple[float, float, float]]:
    for hc, dc, ac in (
        ("B365H", "B365D", "B365A"),
        ("AvgH", "AvgD", "AvgA"),
        ("MaxH", "MaxD", "MaxA"),
    ):
        vals = [row.get(hc), row.get(dc), row.get(ac)]
        if all(pd.notna(v) for v in vals):
            return float(vals[0]), float(vals[1]), float(vals[2])
    return None


def predict_match(
    dc: DixonColes,
    elo: EloEngine,
    home: str,
    away: str,
    market_odds: Optional[tuple[float, float, float]] = None,
) -> dict:
    """Blend Dixon-Coles and Elo into a final H/D/A probability vector."""
    dc_p = dc.match_probabilities(home, away, max_goals=6)
    dc_vec = [dc_p["home_win"], dc_p["draw"], dc_p["away_win"]]
    elo_vec = list(elo.win_draw_loss_prob(home, away))
    blended = [(a + b) / 2.0 for a, b in zip(dc_vec, elo_vec)]

    result = {
        "home": home,
        "away": away,
        "p_home": blended[0],
        "p_draw": blended[1],
        "p_away": blended[2],
        "market_implied": None,
        "edge": None,
        "pick": None,
    }
    if market_odds is not None:
        implied = implied_probabilities(*market_odds)
        result["market_implied"] = list(implied)
        edges = [m - imp for m, imp in zip(blended, implied)]
        best = max(range(3), key=lambda i: edges[i])
        result["edge"] = edges[best]
        result["pick"] = ("home", "draw", "away")[best] if edges[best] > 0 else None
    return result


def generate_predictions(
    dc: DixonColes,
    elo: EloEngine,
    fixtures: pd.DataFrame,
) -> list[dict]:
    """Predict every upcoming fixture and return a list of prediction dicts."""
    out = []
    for _, row in fixtures.iterrows():
        home = str(row.get("home_team") or row.get("home"))
        away = str(row.get("away_team") or row.get("away"))
        odds = _odds_from_fixture(row)
        pred = predict_match(dc, elo, home, away, market_odds=odds)
        pred["date"] = str(row.get("date"))
        pred["league"] = row.get("league")
        out.append(pred)
    return out
