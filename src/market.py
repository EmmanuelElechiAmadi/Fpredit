"""
Market odds analysis — value betting and model-vs-bookmaker comparison.

Football-data.co.uk CSVs include closing odds (B365H/D/A = Bet365,
BbAvH/D/A = BetBrain average). Comparing our model probabilities against
the implied probabilities of these odds lets us:

  1. Find value bets (model probability > bookmaker implied probability,
     with an edge large enough to beat the bookmaker's margin).
  2. Backtest whether our "value" calls actually beat the market long-run
     (profit/loss excluding stake, strike rate, ROI).
  3. Report calibration / Brier score against odds as a sanity check that
     our probabilities are well-scaled (they should never be wildly more
     confident than the market, or we are overfitting).
"""

import numpy as np
import pandas as pd

from src.log import get_logger

log = get_logger(__name__)

# Canonical decimal-odds columns, in order (home, draw, away).
# BbAv* (BetBrain average) is more stable than a single bookmaker's line,
# so it is preferred when present; falls back to B365*.
_PRIMARY_ODDS = ["BbAvH", "BbAvD", "BbAvA"]
_FALLBACK_ODDS = ["B365H", "B365D", "B365A"]


def implied_probabilities(odds_home, odds_draw, odds_away):
    """Raw implied probabilities from decimal odds before removing the bookmaker margin."""
    inv = [1.0 / o for o in (odds_home, odds_draw, odds_away)]
    total = sum(inv)
    return [p / total for p in inv]


def odds_columns_available(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in _PRIMARY_ODDS) or all(
        c in df.columns for c in _FALLBACK_ODDS
    )


def _pick_odds(df: pd.DataFrame) -> tuple:
    """Return the odds column triple to use (best available source), or (None, None, None)."""
    if all(c in df.columns for c in _PRIMARY_ODDS):
        return _PRIMARY_ODDS
    if all(c in df.columns for c in _FALLBACK_ODDS):
        return _FALLBACK_ODDS
    return (None, None, None)


def add_implied_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns implied_home/draw/away (margin-normalized) using best available odds."""
    hc, dc, ac = _pick_odds(df)
    out = df.copy()
    if hc is None:
        for c in ("implied_home", "implied_draw", "implied_away"):
            out[c] = np.nan
        return out

    probs = implied_probabilities(out[hc], out[dc], out[ac])
    out["implied_home"], out["implied_draw"], out["implied_away"] = (
        probs[0],
        probs[1],
        probs[2],
    )
    return out


def value_bets(
    df: pd.DataFrame,
    model_probs: pd.DataFrame,
    edge_threshold: float = 0.05,
    min_probability: float = 0.15,
) -> pd.DataFrame:
    """Find value bets where the model's probability beats the bookmaker's implied
    probability by at least `edge_threshold` (in probability points) and the model
    is confident enough to bet (min_probability).

    Parameters
    ----------
    df : pd.DataFrame
        Matches with odds columns (and output of backtest).
    model_probs : pd.DataFrame
        Same index/order as df with columns home_win, draw, away_win.
    edge_threshold : float
        Minimum edge (model - implied) in probability units to flag a bet.
    min_probability : float
        Minimum model probability to bet (avoids betting 12% long-shots the
        bookmaker already prices at 8%).

    Returns
    -------
    pd.DataFrame
        One row per value bet with columns: date, home_team, away_team, result,
        market (H/D/A), model_prob, implied_prob, edge, odds, stake_ret (0 or 1),
        pnl (profit/loss on 1 unit).
    """
    df = df.reset_index(drop=True)
    model_probs = model_probs.reset_index(drop=True)
    hc, dc, ac = _pick_odds(df)
    if hc is None:
        log.warning("value_bets: no odds columns available; returning empty frame")
        return pd.DataFrame(
            columns=[
                "date",
                "home_team",
                "away_team",
                "result",
                "market",
                "model_prob",
                "implied_prob",
                "edge",
                "odds",
                "stake_ret",
                "pnl",
            ]
        )

    rows = []
    markets = {"H": (0, hc, "home_win"), "D": (1, dc, "draw"), "A": (2, ac, "away_win")}
    for i, row in df.iterrows():
        for market, (prob_idx, odds_col, prob_col) in markets.items():
            implied = 1.0 / float(row[odds_col])
            model_p = float(model_probs.loc[i, prob_col])
            edge = model_p - implied
            if edge >= edge_threshold and model_p >= min_probability:
                won = 1.0 if row["result"] == market else 0.0
                ret = won * (float(row[odds_col]) - 1.0) - (1.0 - won)
                rows.append(
                    {
                        "date": row["date"],
                        "home_team": row["home_team"],
                        "away_team": row["away_team"],
                        "result": row["result"],
                        "market": market,
                        "model_prob": model_p,
                        "implied_prob": implied,
                        "edge": edge,
                        "odds": float(row[odds_col]),
                        "stake_ret": ret + 1.0,  # gross return on 1 unit staked
                        "pnl": ret,  # net P&L on 1 unit staked
                    }
                )
    return pd.DataFrame(rows)


def market_comparison(df_with_odds: pd.DataFrame, model_probs: pd.DataFrame) -> dict:
    """Summary statistics comparing model probabilities to bookmaker implied
    probabilities over past matches.

    Returns
    -------
    dict with keys:
      n, brier_model, brier_market, log_loss_model, log_loss_market,
      calibration buckets, and whether model beats market on Brier.
    """
    df = add_implied_probabilities(df_with_odds.reset_index(drop=True))
    model_probs = model_probs.reset_index(drop=True)
    has_odds = df[["implied_home", "implied_draw", "implied_away"]].notna().all(axis=1)
    if not has_odds.any() or len(model_probs) != len(df):
        return {
            "n": 0,
            "brier_model": None,
            "brier_market": None,
            "log_loss_model": None,
            "log_loss_market": None,
            "calibration": [],
            "note": "No odds columns available in the loaded data for this period.",
        }

    df, mp = df[has_odds], model_probs[has_odds]

    # One-hot actual outcomes
    y = np.zeros((len(df), 3))
    for i, r in enumerate(df["result"]):
        y[i, {"H": 0, "D": 1, "A": 2}[r]] = 1.0

    model_arr = mp[["home_win", "draw", "away_win"]].to_numpy()
    market_arr = df[["implied_home", "implied_draw", "implied_away"]].to_numpy()

    brier_model = float(np.mean(np.sum((model_arr - y) ** 2, axis=1)))
    brier_market = float(np.mean(np.sum((market_arr - y) ** 2, axis=1)))

    log_loss_model = float(np.mean(-np.sum(y * np.log(np.clip(model_arr, 1e-9, 1.0)), axis=1)))
    log_loss_market = float(np.mean(-np.sum(y * np.log(np.clip(market_arr, 1e-9, 1.0)), axis=1)))

    # Calibration: bin predicted probability (for the actual outcome chosen)
    cal = []
    conf = np.max(model_arr, axis=1)
    acc = (np.argmax(model_arr, axis=1) == np.argmax(y, axis=1)).astype(float)
    for lo in np.arange(1 / 3, 0.95, 0.1):
        mask = (conf >= lo) & (conf < lo + 0.1)
        if mask.sum() >= 20:
            cal.append(
                {
                    "bin": round(lo, 2),
                    "n": int(mask.sum()),
                    "confidence": round(float(conf[mask].mean()), 3),
                    "accuracy": round(float(acc[mask].mean()), 3),
                }
            )

    return {
        "n": int(len(df)),
        "brier_model": brier_model,
        "brier_market": brier_market,
        "log_loss_model": log_loss_model,
        "log_loss_market": log_loss_market,
        "beats_market": brier_model < brier_market,
        "calibration": cal,
    }