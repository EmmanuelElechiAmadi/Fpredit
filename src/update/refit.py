"""Refit models and rebuild features after new results land.

- Dixon-Coles (bivariate Poisson) refit with time-decay weighting (reuses
  ``src.dixon_coles.DixonColes``).
- Rolling features rebuild (reuses ``src.features`` — every feature is computed
  strictly from information available before kickoff, so rebuilding from the
  stored history is leak-free).
- Walk-forward log-loss vs the market (reuses ``src.market`` implied
  probabilities) so every report can state whether the model still beats the
  closing line.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.dixon_coles import DixonColes
from src.features import fixture_congestion, head_to_head, rolling_form
from src.market import implied_probabilities


def build_features(
    df: pd.DataFrame,
    form_window: int = 5,
    h2h_lookback: int = 5,
    congestion_days: int = 8,
    load_days: int = 14,
) -> pd.DataFrame:
    """Rebuild rolling features from a chronologically sorted engine frame."""
    df = df.sort_values("date").reset_index(drop=True)
    out = rolling_form(df, window=form_window)
    out["h2h_home_ppg_norm"] = head_to_head(out, lookback_matches=h2h_lookback).values
    out = fixture_congestion(out, congestion_days=congestion_days, load_days=load_days)
    return out


def refit_dixon_coles(
    df: pd.DataFrame,
    xi: float = 0.0018,
    shrinkage: float = 0.0,
    ref_date: Optional[pd.Timestamp] = None,
) -> DixonColes:
    """Fit Dixon-Coles on completed matches (latest match as the decay reference)."""
    train = df.dropna(subset=["home_goals", "away_goals"]).sort_values("date")
    records = train.to_dict("records")
    return DixonColes(xi=xi, shrinkage=shrinkage).fit(records, ref_date=ref_date)


def _pick_odds(row) -> Optional[tuple[float, float, float]]:
    for hc, dc, ac in (("B365H", "B365D", "B365A"), ("BbAvH", "BbAvD", "BbAvA")):
        vals = [row.get(hc), row.get(dc), row.get(ac)]
        if all(pd.notna(v) for v in vals):
            return float(vals[0]), float(vals[1]), float(vals[2])
    return None


def walk_forward_eval(
    df: pd.DataFrame,
    xi: float = 0.0018,
    shrinkage: float = 0.0,
    min_train: int = 30,
    step: int = 50,
) -> dict:
    """Walk-forward log-loss of a Dixon-Coles model vs the closing market line.

    The model is refit every ``step`` matches using only prior matches (no
    leakage); the market's implied probabilities are scored on the same rows.
    """
    df = (
        df.dropna(subset=["home_goals", "away_goals"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    y = np.zeros((len(df), 3))
    for i, r in enumerate(df["result"]):
        y[i, {"H": 0, "D": 1, "A": 2}[r]] = 1.0

    model_probs = np.full((len(df), 3), np.nan)
    market_probs = np.full((len(df), 3), np.nan)
    dc: Optional[DixonColes] = None

    for i in range(len(df)):
        if i < min_train:
            continue
        if dc is None or i % step == 0:
            dc = refit_dixon_coles(df.iloc[:i], xi=xi, shrinkage=shrinkage)
        row = df.iloc[i]
        p = dc.match_probabilities(row["home_team"], row["away_team"], max_goals=6)
        model_probs[i] = [p["home_win"], p["draw"], p["away_win"]]
        odds = _pick_odds(row)
        if odds is not None:
            market_probs[i] = implied_probabilities(*odds)

    mask = ~np.isnan(model_probs).any(axis=1)
    if mask.sum() == 0:
        return {"n": 0, "log_loss_model": None, "log_loss_market": None}

    def _logloss(probs: np.ndarray, m: np.ndarray) -> float:
        return float(np.mean(-np.sum(m * np.log(np.clip(probs, 1e-9, 1.0)), axis=1)))

    out: dict = {
        "n": int(mask.sum()),
        "log_loss_model": _logloss(model_probs[mask], y[mask]),
    }
    mkt_mask = mask & ~np.isnan(market_probs).any(axis=1)
    out["log_loss_market"] = (
        _logloss(market_probs[mkt_mask], y[mkt_mask]) if mkt_mask.sum() else None
    )
    return out


def refit_ensemble(df: pd.DataFrame, cfg=None, xg_df: Optional[pd.DataFrame] = None):
    """Build + fit the full stacked ensemble (reuses src.model_factory)."""
    from src.model_factory import build_ensemble
    from src.xg_loader import join_xg

    model = build_ensemble(cfg)
    train = df.copy()
    if xg_df is not None and not xg_df.empty:
        train = join_xg(train, xg_df)
    model.fit(train, xg_df=xg_df)
    return model
