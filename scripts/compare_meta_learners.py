#!/usr/bin/env python3
"""Honest A/B of meta-learner variants (Section C.9).

Compares the multinomial-logistic meta-learner against the gradient-boosted
tree and the isotonic/Platt-calibrated variants on the SAME walk-forward
backtest (identical windows, features and data), and reports the metrics side
by side. The only thing that changes between runs is the meta-learner
configuration — so any log-loss/Brier difference is attributable to it.

    python3 scripts/compare_meta_learners.py                 # EPL, all variants
    python3 scripts/compare_meta_learners.py --league LALIGA --learners logistic gbm
    python3 scripts/compare_meta_learners.py --demo          # tiny synthetic check

Variants are expressed as ``learner`` or ``learner+calibration``, e.g.
``gbm+isotonic``, ``logistic+sigmoid``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backtest as bt_mod
from src.config import load_config
from src.data_loader import generate_synthetic_league, load_league_csvs
from src.market import add_implied_probabilities
from src.xg_loader import load_league_xg

DEFAULT_VARIANTS = ["logistic", "gbm", "logistic+sigmoid", "gbm+isotonic"]


def _parse_variant(spec: str) -> tuple[str, str | None]:
    parts = spec.split("+", 1)
    learner = parts[0]
    if learner not in ("logistic", "gbm"):
        raise SystemExit(f"Unknown meta-learner '{learner}' (use 'logistic' or 'gbm').")
    calibration = parts[1] if len(parts) > 1 else None
    if calibration not in (None, "sigmoid", "isotonic"):
        raise SystemExit(
            f"Unknown calibration '{calibration}' (use sigmoid or isotonic)."
        )
    return learner, calibration


def run_variant(league: str, df, xg_df, learner: str, calibration: str | None) -> dict:
    cfg = load_config()
    cfg.model.meta_learner = learner
    cfg.model.meta_calibration = calibration

    result_df = bt_mod.walk_forward_backtest(
        df,
        min_train_matches=cfg.backtest.min_train_matches,
        step_matches=cfg.backtest.step_matches,
        cfg=cfg,
        xg_df=xg_df,
    )
    if result_df.empty:
        raise RuntimeError(f"{learner}+{calibration}: no predictions produced.")
    metrics = bt_mod.evaluate(result_df, cfg=cfg)
    return {
        "log_loss": metrics["log_loss"],
        "brier": metrics["brier"],
        "accuracy": metrics["accuracy"],
        "baseline_log_loss": metrics["baseline_log_loss"],
        "n": len(result_df),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--league", default="EPL", choices=["EPL", "LALIGA", "SERIEA"])
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--xg-dir", default=None)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument(
        "--learners",
        nargs="+",
        default=DEFAULT_VARIANTS,
        help=f"Variants to compare (default: {' '.join(DEFAULT_VARIANTS)}).",
    )
    args = ap.parse_args(list(argv) if argv else None)

    variants = [_parse_variant(v) for v in args.learners]

    df = (
        generate_synthetic_league(n_seasons=6)
        if args.demo
        else load_league_csvs(args.data_dir, args.league)
    )
    df = add_implied_probabilities(df).sort_values("date").reset_index(drop=True)

    xg_df = None
    if args.xg_dir:
        xg_df = load_league_xg(args.xg_dir, args.league)
    elif args.demo:
        from src.xg_loader import generate_synthetic_xg

        xg_df = generate_synthetic_xg(df)

    print(f"\nMeta-learner A/B — {args.league} ({len(df)} matches)")
    print("Walk-forward windows identical for every variant.\n")

    results = []
    for learner, calibration in variants:
        label = learner if calibration is None else f"{learner}+{calibration}"
        try:
            m = run_variant(args.league, df, xg_df, learner, calibration)
            results.append((label, m))
            print(
                f"  {label:<20} log_loss {m['log_loss']:.4f}  brier {m['brier']:.4f}  "
                f"acc {m['accuracy']:.1%}  (n={m['n']})"
            )
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"  {label:<20} FAILED: {e}")

    if len(results) > 1:
        best = min(results, key=lambda kv: kv[1]["log_loss"])
        print(
            f"\nBest by log loss: {best[0]} "
            f"({best[1]['log_loss']:.4f} vs constant-prior baseline "
            f"{best[1]['baseline_log_loss']:.4f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
