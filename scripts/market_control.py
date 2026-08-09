"""
Market sanity-check control: "bet the market, no model opinion".

The honest negative result (residual log loss +0.038, Kelly Sharpe -0.58,
value-bet ROI ~-12%) is only trustworthy if a *trivial market-only* strategy
through the same pipeline loses roughly the bookmaker's margin. If the trivial
control comes back much closer to zero than -vig, something in the
residual/staking/backtest chain is off (sign error, leakage direction,
off-by-one in the walk-forward window) and the "no edge" conclusion is not
safe to accept.

Strategies compared over the same out-of-sample window:
  - market-favorite flat 1u       (control: no model opinion)
  - model-favorite flat 1u
  - model value-bet fractional Kelly   (the shipped staking layer)
  - market Kelly (zero bets by construction: the market can't edge itself)

Also reports the residual-vs-market log loss measured against EACH closing
line available (Bet365 and Pinnacle close) on the same predictions, so a
positive result against Pinnacle specifically is a sharper statement than one
against Bet365.

Usage:
    python scripts/market_control.py --results /tmp/bt_epl_tuned.csv
    python scripts/market_control.py --league EPL --xg-dir data/xg   # runs a backtest first
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.market import implied_probabilities, market_comparison, value_bets
from src.staking import covariance_adjusted_stakes, portfolio_report

OUTCOMES = ["H", "D", "A"]


def _implied_from_odds(row, cols: tuple[str, str, str]) -> list[float] | None:
    vals = [row.get(c) for c in cols]
    if any(pd.isna(v) for v in vals):
        return None
    return implied_probabilities(float(vals[0]), float(vals[1]), float(vals[2]))


def _margin(row, cols: tuple[str, str, str]) -> float | None:
    vals = [row.get(c) for c in cols]
    if any(pd.isna(v) for v in vals):
        return None
    return sum(1.0 / float(v) for v in vals) - 1.0


def _flat_favorite_pnl(df: pd.DataFrame, probs, label: str) -> dict:
    """Flat 1u on the favorite according to `probs` (list of [pH,pD,pA] or None)."""
    rets, stakes = [], []
    for i, row in df.iterrows():
        p = probs[i]
        if p is None:
            continue
        pick = OUTCOMES[int(np.argmax(p))]
        odds_col = {"H": "B365H", "D": "B365D", "A": "B365A"}[pick]
        odds = row.get(odds_col)
        if pd.isna(odds):
            continue
        won = 1.0 if row["actual"] == pick else 0.0
        pnl = won * (float(odds) - 1.0) - (1.0 - won)
        rets.append(pnl)
        stakes.append(1.0)
    report = portfolio_report(pd.Series(rets), pd.Series(stakes))
    report["label"] = label
    report["n"] = len(rets)
    report["strike"] = (
        float(np.mean([1.0 if r > 0 else 0.0 for r in rets])) if rets else None
    )
    return report


def _model_value_staking(df: pd.DataFrame) -> dict:
    """The shipped covariance-adjusted Kelly layer on model value bets."""
    mp = df[["pred_home_win", "pred_draw", "pred_away_win"]].rename(
        columns={
            "pred_home_win": "home_win",
            "pred_draw": "draw",
            "pred_away_win": "away_win",
        }
    )
    vb = value_bets(df, mp)
    if vb.empty:
        return {"label": "model value-bet Kelly", "n": 0}
    stakes = covariance_adjusted_stakes(vb)
    report = portfolio_report(vb["pnl"] * stakes.values, stakes)
    report["label"] = "model value-bet Kelly"
    report["n"] = int((stakes > 0).sum())
    report["strike"] = float((vb["stake_ret"] > 1.0).mean())
    return report


def run_control(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["result"] = df["actual"]

    # Market-implied probabilities from each available closing line.
    b365 = [
        _implied_from_odds(r, ("B365H", "B365D", "B365A")) for _, r in df.iterrows()
    ]
    model = [
        [r["pred_home_win"], r["pred_draw"], r["pred_away_win"]]
        for _, r in df.iterrows()
    ]

    # Average bookmaker margin (vig) on the primary closing line.
    margins_all = [_margin(r, ("B365H", "B365D", "B365A")) for _, r in df.iterrows()]
    margins: list[float] = [m for m in margins_all if m is not None]
    avg_vig = float(np.mean(margins)) if margins else None

    rows = [
        _flat_favorite_pnl(df, b365, "market favorite (control)"),
        _flat_favorite_pnl(df, model, "model favorite"),
    ]
    rows.append(_model_value_staking(df))

    # Market Kelly: betting the market's own implied probs has zero edge by
    # construction (the vig makes p*odds - 1 < 0 for every outcome).
    rows.append({"label": "market Kelly (zero edge by construction)", "n": 0})

    out = pd.DataFrame(rows)
    out["avg_margin"] = avg_vig

    # Residual-vs-market log loss per closing line (same model predictions).
    mp = df[["pred_home_win", "pred_draw", "pred_away_win"]].rename(
        columns={
            "pred_home_win": "home_win",
            "pred_draw": "draw",
            "pred_away_win": "away_win",
        }
    )
    resid = {}
    for label, cols in [
        ("vs Bet365 close", ("B365H", "B365D", "B365A")),
        ("vs Pinnacle close", ("PSH", "PSD", "PSA")),
    ]:
        d = df.copy()
        impl = [_implied_from_odds(r, cols) for _, r in df.iterrows()]
        d["implied_home"] = [x[0] if x else np.nan for x in impl]
        d["implied_draw"] = [x[1] if x else np.nan for x in impl]
        d["implied_away"] = [x[2] if x else np.nan for x in impl]
        mc = market_comparison(d, mp)
        resid[label] = (mc["log_loss_model"], mc["log_loss_market"])
    out.attrs["residual_by_line"] = resid
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=None, help="Path to a backtest results CSV")
    ap.add_argument("--league", default="EPL", choices=["EPL", "LALIGA", "SERIEA"])
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--xg-dir", default=None)
    ap.add_argument("--min-train", type=int, default=380)
    ap.add_argument("--step", type=int, default=190)
    args = ap.parse_args()

    if args.results:
        df = pd.read_csv(args.results, parse_dates=["date"])
    else:
        import backtest as bt
        from src.config import load_config
        from src.data_loader import load_league_csvs
        from src.xg_loader import join_xg, load_league_xg

        cfg = load_config()
        df = load_league_csvs(args.data_dir, args.league)
        xg = load_league_xg(args.xg_dir, args.league) if args.xg_dir else None
        if xg is not None:
            df = join_xg(df, xg)
        df = bt.walk_forward_backtest(
            df,
            min_train_matches=args.min_train,
            step_matches=args.step,
            cfg=cfg,
            xg_df=xg,
        )

    df = df[df["pred_home_win"].notna()]
    if df.empty or "B365H" not in df.columns:
        print("No odds-carrying predictions available in the given results.")
        return

    print(f"\nOut-of-sample window: {len(df)} matches with closing odds\n")

    c = run_control(df)
    cols = ["label", "n", "total_staked", "roi", "sharpe", "max_drawdown", "strike"]
    print(c[cols].to_string(index=False))

    vig = c["avg_margin"].iloc[0]
    print(f"\nAverage bookmaker margin (Bet365 close): {vig:.2%}")

    print("\nResidual log loss (model minus line; negative = model beats line):")
    for label, (ll_m, ll_mkt) in c.attrs["residual_by_line"].items():
        print(
            f"  {label:>22}: model {ll_m:.4f} vs line {ll_mkt:.4f}  ->  {ll_m - ll_mkt:+.4f}"
        )


if __name__ == "__main__":
    main()
