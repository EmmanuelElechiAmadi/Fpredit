"""
Walk-forward backtest: the ONLY honest way to evaluate a sports model.

Never train and test on shuffled/random splits of match data -- matches are
time-ordered and team strength drifts, so a random split leaks future
information (the model "sees" a team's May form while predicting their
September match). We train on everything before a cutoff, predict the next
block, then roll the cutoff forward.

Metrics reported:
  - Log loss (lower is better; this is what betting markets are effectively
    scored on, it heavily punishes confident wrong predictions)
  - Brier score (lower is better; mean squared error of the probability vector)
  - Accuracy (informative but NOT the metric to optimize -- always predicting
    "home win" scores ~45-46% accuracy in most leagues due to home advantage,
    so a model needs to clear that baseline meaningfully, and accuracy alone
    hides whether the probabilities themselves are well-calibrated)
  - Baseline comparison: "always predict home win" and "market-share prior"
    (constant H/D/A rates), so you can see the actual uplift from modeling.
  - Market comparison (when odds columns are present): Brier/log-loss vs the
    bookmaker-implied probabilities, plus value-bet P&L.

Usage: python backtest.py --demo
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from src.config import load_config
from src.data_loader import generate_synthetic_league, load_league_csvs
from src.market import market_comparison, value_bets
from src.model_factory import build_ensemble
from src.staking import covariance_adjusted_stakes, portfolio_report

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def _season_start_year(ts) -> int:
    """European-season start year: Aug-May spans belong to the starting year.

    A 2023-08-11 kickoff -> 2023; a 2024-05-19 kickoff (same season) -> 2023.
    """
    return ts.year if ts.month >= 7 else ts.year - 1


def split_holdout(df: pd.DataFrame, n_seasons: int = 1):
    """Split into (tuning, holdout) by the most recent n complete seasons.

    The holdout is meant to be touched exactly once, at final evaluation —
    never for training, tuning, or feature selection.
    """
    df = df.sort_values("date").reset_index(drop=True)
    starts = df["date"].map(_season_start_year)
    holdout_start = sorted(set(starts))[-n_seasons]
    train = df[starts < holdout_start].copy()
    holdout = df[starts >= holdout_start].copy()
    return train, holdout


def walk_forward_backtest(
    df: pd.DataFrame,
    min_train_matches=380,
    step_matches=190,
    cfg=None,
    xg_df: pd.DataFrame | None = None,
):
    from src.xg_loader import join_xg

    df = df.sort_values("date").reset_index(drop=True)
    if xg_df is not None and not xg_df.empty:
        df = join_xg(df, xg_df)
    records = []

    odds_cols = [
        "B365H",
        "B365D",
        "B365A",
        "BbAvH",
        "BbAvD",
        "BbAvA",
        "PH",
        "PD",
        "PA",
        "PSH",
        "PSD",
        "PSA",
    ]

    dc_state = None
    cutoff = min_train_matches
    while cutoff + step_matches <= len(df):
        train = df.iloc[:cutoff]
        test = df.iloc[cutoff : cutoff + step_matches]

        model = build_ensemble(cfg)
        model.fit(train, xg_df=None, dc_warm_start=dc_state)
        dc_state = {
            "attack": dict(model.dc.attack),
            "defense": dict(model.dc.defense),
            "home_adv": float(model.dc.home_adv),
            "rho": float(model.dc.rho),
        }

        try:
            pred_df = model.predict_frame(test)
        except Exception as e:
            log.warning("Window prediction failed: %s", e)
            cutoff += step_matches
            continue

        for idx, row in test.iterrows():
            try:
                p = pred_df.loc[idx]
                if pd.isna(p["home_win"]):
                    log.warning(
                        "Skipping %s vs %s (%s): team has no training history",
                        row["home_team"],
                        row["away_team"],
                        row["date"],
                    )
                    continue
                rec = {
                    "date": row["date"],
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "actual": row["result"],
                    "pred_home_win": p["home_win"],
                    "pred_draw": p["draw"],
                    "pred_away_win": p["away_win"],
                    "pred_over_2_5": p["over_2_5"],
                    "pred_btts_yes": p["btts_yes"],
                    "expected_home_goals": p["expected_home_goals"],
                    "expected_away_goals": p["expected_away_goals"],
                }
                # Carry any odds columns through so market comparison is possible.
                for oc in odds_cols:
                    if oc in test.columns:
                        rec[oc] = row.get(oc, np.nan)
                records.append(rec)
            except Exception as e:
                log.warning(
                    "Prediction failed for %s vs %s (%s): %s",
                    row["home_team"],
                    row["away_team"],
                    row["date"],
                    e,
                )
                continue

        cutoff += step_matches

    result_df = pd.DataFrame(records)
    if result_df.empty:
        log.warning(
            "Backtest produced 0 predictions — check that the model can "
            "predict matches given the training data (team names, history, etc.)"
        )
    return result_df


def evaluate(result_df: pd.DataFrame, with_odds: bool = True, cfg=None):
    """Evaluate backtest predictions.

    Parameters
    ----------
    result_df : pd.DataFrame
        Output of walk_forward_backtest (with optional odds columns).
    with_odds : bool
        Also compute market comparison + value bet stats when odds are present.
    cfg : optional
        Config namespace used for staking hyperparameters (falls back to
        sensible defaults when None).

    Returns
    -------
    dict of metrics (plus 'market' and 'value_bets' sub-dicts when available).
    """
    label_map = {"H": 0, "D": 1, "A": 2}
    y_true = np.array([label_map[a] for a in result_df["actual"]])
    prob_cols = ["pred_home_win", "pred_draw", "pred_away_win"]
    preds = result_df[prob_cols].values
    y_pred_class = preds.argmax(axis=1)

    ll = log_loss(y_true, preds, labels=[0, 1, 2])
    acc = accuracy_score(y_true, y_pred_class)

    # one-vs-rest Brier, averaged
    brier = np.mean(
        [brier_score_loss((y_true == k).astype(int), preds[:, k]) for k in range(3)]
    )

    # baselines
    baseline_home_acc = np.mean(y_true == 0)
    class_rates = np.bincount(y_true, minlength=3) / len(y_true)
    baseline_prior_preds = np.tile(class_rates, (len(y_true), 1))
    baseline_ll = log_loss(y_true, baseline_prior_preds, labels=[0, 1, 2])

    print(f"n matches evaluated: {len(y_true)}")
    print(f"Model log loss:        {ll:.4f}   (lower is better)")
    print(f"Constant-prior baseline log loss: {baseline_ll:.4f}")
    print(f"Model Brier score:     {brier:.4f}   (lower is better)")
    print(f"Model accuracy:        {acc:.1%}")
    print(f"'Always home win' accuracy baseline: {baseline_home_acc:.1%}")
    print(
        f"\nModel beats constant-prior baseline: {'YES' if ll < baseline_ll else 'NO -- needs work'}"
    )

    metrics = {
        "log_loss": ll,
        "brier": brier,
        "accuracy": acc,
        "baseline_log_loss": baseline_ll,
    }

    if with_odds:
        has_odds = all(
            c in result_df.columns for c in ("BbAvH", "BbAvD", "BbAvA")
        ) or all(c in result_df.columns for c in ("B365H", "B365D", "B365A"))
        if has_odds:
            model_probs = result_df[prob_cols].rename(
                columns={
                    "pred_home_win": "home_win",
                    "pred_draw": "draw",
                    "pred_away_win": "away_win",
                }
            )
            # market_comparison / value_bets expect a `result` column; the
            # backtest emits `actual`.
            market_df = result_df.rename(columns={"actual": "result"})
            market = market_comparison(market_df, model_probs)
            vb = value_bets(market_df, model_probs)
            # Residual-vs-market: the model's log loss minus the market's.
            # Negative = the model adds information beyond the closing line.
            if (
                market["log_loss_model"] is not None
                and market["log_loss_market"] is not None
            ):
                market["residual_log_loss"] = (
                    market["log_loss_model"] - market["log_loss_market"]
                )
            metrics["market"] = market

            if not vb.empty:
                # Does the model's predicted edge actually predict winning?
                edge_corr = float(
                    vb["edge"].corr(vb["stake_ret"].gt(1.0).astype(float))
                )
                metrics["edge_corr"] = edge_corr

                # Covariance-adjusted fractional-Kelly staking on the slate
                stakes = covariance_adjusted_stakes(
                    vb,
                    kelly_fraction=cfg.model.kelly_fraction if cfg else 0.25,
                    max_stake=cfg.model.kelly_max_stake if cfg else 0.10,
                    corr=cfg.model.kelly_corr if cfg else 0.05,
                    cov_shrinkage=cfg.model.kelly_cov_shrinkage if cfg else 0.9,
                )
                vb = vb.assign(
                    stake=stakes.values, pnl_staked=stakes.values * vb["pnl"].values
                )
                metrics["staking"] = portfolio_report(vb["pnl_staked"], vb["stake"])
                metrics["value_bets"] = {
                    "n_bets": len(vb),
                    "strike_rate": float(vb["stake_ret"].gt(1.0).mean()),
                    "profit_units": float(vb["pnl"].sum()),
                    "roi": float(vb["pnl"].sum() / len(vb)),
                }
                print(f"\nMarket comparison ({market['n']} matches with odds):")
                print(
                    f"  Model Brier {market['brier_model']:.4f} vs market {market['brier_market']:.4f}"
                )
                print(
                    f"  Model log loss {market['log_loss_model']:.4f} vs market {market['log_loss_market']:.4f}"
                )
                print(
                    f"  Residual log loss {market.get('residual_log_loss'):+.4f} "
                    f"(negative = model beats market)"
                )
                if metrics.get("edge_corr") is not None:
                    print(f"  Edge-win correlation: {metrics['edge_corr']:+.3f}")
                st = metrics["staking"]
                print(
                    f"  Value bets: {len(vb)} (P&L {vb['pnl'].sum():+.2f} units, ROI {vb['pnl'].sum()/len(vb):+.1%})"
                )
                print(
                    f"  Kelly staking: {st['n']} bets, {st['total_staked']:.2f} staked, "
                    f"profit {st['profit_units']:+.3f}, Sharpe {st['sharpe'] if st['sharpe'] is not None else float('nan'):.2f}, "
                    f"maxDD {st['max_drawdown']:.3f}, CVaR95 {st['cvar95'] if st['cvar95'] is not None else float('nan'):.3f}"
                )
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="EPL", choices=["EPL", "LALIGA", "SERIEA"])
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument(
        "--xg-dir",
        default=None,
        help="Directory of cached Understat xG CSVs (data/xg). When set, xG "
        "features are enabled for the backtest.",
    )
    ap.add_argument("--demo", action="store_true")
    ap.add_argument(
        "--with-xg",
        action="store_true",
        help="Demo mode: also attach synthetic xG columns (tests the xG path).",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Path to save detailed predictions CSV (e.g. backtest_results.csv)",
    )
    ap.add_argument(
        "--holdout-seasons",
        type=int,
        default=0,
        help="Treat the most recent N seasons as a held-out evaluation set: "
        "train once on everything before them, predict the holdout, and report "
        "only that window (pre-registered protocol — see "
        "docs/edge_test_preregistration.md).",
    )
    args = ap.parse_args()

    cfg = load_config()

    df = (
        generate_synthetic_league(n_seasons=4)
        if args.demo
        else load_league_csvs(args.data_dir, args.league)
    )
    xg_df = None
    if args.xg_dir:
        from src.xg_loader import load_league_xg

        xg_df = load_league_xg(args.xg_dir, args.league)
        print(f"Loaded xG for {args.league} ({len(xg_df)} matches)")
    elif args.demo and args.with_xg:
        from src.xg_loader import generate_synthetic_xg

        xg_df = generate_synthetic_xg(df)
        print("Demo xG attached (synthetic)")

    min_train = cfg.backtest.min_train_matches
    step = cfg.backtest.step_matches
    if args.holdout_seasons > 0:
        train_df, holdout_df = split_holdout(df, args.holdout_seasons)
        min_train = len(train_df)
        step = len(holdout_df)
        print(
            f"Holdout protocol: training on {min_train} matches "
            f"(through {train_df['date'].max().date()}), evaluating only "
            f"{step} held-out matches (from {holdout_df['date'].min().date()})."
        )
        if xg_df is not None:
            xg_df = xg_df[xg_df["date"] < holdout_df["date"].min()].copy()
            print(f"xG restricted to the training window ({len(xg_df)} matches)")

    result_df = walk_forward_backtest(
        df,
        min_train_matches=min_train,
        step_matches=step,
        cfg=cfg,
        xg_df=xg_df,
    )
    if result_df.empty:
        print(
            "No predictions could be made. See warnings above (run with --verbose "
            "or check the data)."
        )
        raise SystemExit(1)
    metrics = evaluate(result_df, cfg=cfg)

    if args.output:
        result_df.to_csv(args.output, index=False)
        print(f"\nDetailed predictions saved to: {args.output}")
