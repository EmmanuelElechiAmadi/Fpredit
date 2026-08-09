"""
Walk-forward hyperparameter calibration for the ensemble model.

Runs the same honest walk-forward backtest the app uses, but over a grid of
model hyperparameters (ss_q, ss_q_xg, dc_shrinkage, dc_xi, meta_C) and staking
parameters (kelly_fraction, kelly_corr). Every combo is evaluated strictly
out-of-sample:

  - overall log loss (vs the constant-prior baseline)
  - residual-vs-market log loss (model minus closing line; negative = edge)
  - edge correlation (does the predicted value edge predict winning?)
  - Kelly staking Sharpe / ROI / max drawdown

A composite score prefers residual log loss when odds data is available and
falls back to plain log loss otherwise; staking stage prefers Sharpe.

Usage:
    python scripts/calibrate_model.py --league EPL --xg-dir data/xg --grid quick
    python scripts/calibrate_model.py --league EPL --xg-dir data/xg --grid full
    python scripts/calibrate_model.py --league EPL --write-config   # apply best
    python scripts/calibrate_model.py --demo                        # synthetic sanity
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

# Make `src`/`backtest` importable when running directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yaml  # type: ignore[import-untyped]

from backtest import evaluate, walk_forward_backtest
from src.config import load_config
from src.data_loader import generate_synthetic_league, load_league_csvs
from src.xg_loader import join_xg, load_league_xg

# Per-parameter candidate values. The first value is the current config default
# so the baseline run reports the un-tuned result.
GRIDS = {
    "quick": {
        "ss_q": [0.01, 0.005, 0.02],
        "ss_q_xg": [0.005, 0.002, 0.01],
        "dc_shrinkage": [0.0, 0.05, 0.1],
        "dc_xi": [0.0018, 0.0035],
        "meta_C": [1.0, 0.5, 2.0],
        "kelly_fraction": [0.25, 0.5],
        "kelly_corr": [0.05, 0.0, 0.1],
    },
    "full": {
        "ss_q": [0.001, 0.005, 0.01, 0.02, 0.05],
        "ss_q_xg": [0.001, 0.002, 0.005, 0.01],
        "dc_shrinkage": [0.0, 0.05, 0.1, 0.2],
        "dc_xi": [0.0018, 0.0025, 0.0035],
        "meta_C": [0.5, 1.0, 2.0],
        "kelly_fraction": [0.1, 0.25, 0.5],
        "kelly_corr": [0.0, 0.03, 0.05, 0.1],
    },
}

STAKING_PARAMS = {
    "kelly_fraction",
    "kelly_corr",
    "kelly_max_stake",
    "kelly_cov_shrinkage",
}


def _set_params(cfg, params: dict):
    """Mutate a copied config namespace with the given model.* overrides."""
    for key, value in params.items():
        setattr(cfg.model, key, value)
    return cfg


def _metric_row(tag, params, metrics) -> dict:
    mkt = metrics.get("market") or {}
    st = metrics.get("staking") or {}
    return {
        "combo": tag,
        **{k: v for k, v in params.items()},
        "n_matches": metrics.get("n_matches", 0),
        "log_loss": round(metrics["log_loss"], 4),
        "brier": round(metrics["brier"], 4),
        "baseline_log_loss": round(metrics["baseline_log_loss"], 4),
        "residual_log_loss": (
            round(mkt["residual_log_loss"], 4)
            if mkt.get("residual_log_loss") is not None
            else None
        ),
        "edge_corr": (
            round(metrics["edge_corr"], 4)
            if metrics.get("edge_corr") is not None
            else None
        ),
        "value_bets": (metrics.get("value_bets") or {}).get("n_bets", 0),
        "kelly_roi": round(st["roi"], 4) if st.get("roi") is not None else None,
        "kelly_sharpe": (
            round(st["sharpe"], 4) if st.get("sharpe") is not None else None
        ),
        "kelly_maxdd": (
            round(st["max_drawdown"], 4) if st.get("max_drawdown") is not None else None
        ),
    }


def _model_score(row) -> float:
    """Lower is better. Prefer residual-vs-market log loss when we have odds."""
    if row["residual_log_loss"] is not None:
        return row["residual_log_loss"]
    return row["log_loss"]


def _staking_score(row) -> float:
    """Higher is better (Sharpe first, then ROI)."""
    if row["kelly_sharpe"] is None:
        return float("-inf")
    return row["kelly_sharpe"]


def calibrate(
    df,
    cfg,
    xg_df=None,
    grid_name="quick",
    write_config=False,
    min_train_matches=760,
    step_matches=380,
    max_combos=None,
):
    base_cfg = copy.deepcopy(cfg)
    grid = GRIDS[grid_name]
    rows: list[dict] = []

    def run(params, tag):
        cfg_run = copy.deepcopy(base_cfg)
        cfg_run = _set_params(cfg_run, params)
        t0 = time.time()
        result_df = walk_forward_backtest(
            df,
            min_train_matches=min_train_matches,
            step_matches=step_matches,
            cfg=cfg_run,
            xg_df=xg_df,
        )
        if result_df.empty:
            print(f"  {tag}: no predictions (check data / team coverage)")
            return None
        metrics = evaluate(result_df, with_odds=True, cfg=cfg_run)
        metrics["n_matches"] = len(result_df)
        row = _metric_row(tag, params, metrics)
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        resid = (
            ""
            if row["residual_log_loss"] is None
            else f"{row['residual_log_loss']:+.4f}"
        )
        edge = "" if row["edge_corr"] is None else f"{row['edge_corr']:+.3f}"
        sharpe = "" if row["kelly_sharpe"] is None else f"{row['kelly_sharpe']:.2f}"
        print(
            f"  {tag:<34} ll={row['log_loss']:.4f} resid={resid} edge={edge} "
            f"sharpe={sharpe} ({row['seconds']:.0f}s)"
        )
        return row

    # ---- Stage 1: model hyperparameters (coordinate descent) ----
    print("\n=== Stage 1: model hyperparameters (walk-forward OOS) ===")
    best_params = {k: grid[k][0] for k in grid if k not in STAKING_PARAMS}
    baseline = run(best_params, "baseline")
    n_run = 1
    for key in ["ss_q", "ss_q_xg", "dc_shrinkage", "dc_xi", "meta_C"]:
        candidates = [v for v in grid[key] if v != best_params[key]]
        best_score = _model_score(baseline)
        best_val = best_params[key]
        for val in candidates:
            if max_combos is not None and n_run >= max_combos:
                break
            trial = {**best_params, key: val}
            row = run(trial, f"tune {key}={val}")
            n_run += 1
            if row is not None and _model_score(row) < best_score - 1e-6:
                best_score = _model_score(row)
                best_val = val
        best_params[key] = best_val
    print(f"  Best model params: {best_params}")

    # ---- Stage 2: staking params on the best model ----
    print("\n=== Stage 2: staking hyperparameters (best model) ===")
    staking_best = run({**best_params}, "best model (default staking)")
    n_run += 1
    best_staking = {
        "kelly_fraction": grid["kelly_fraction"][0],
        "kelly_corr": grid["kelly_corr"][0],
    }
    if staking_best is not None:
        best_s = _staking_score(staking_best)
        for key in ["kelly_fraction", "kelly_corr"]:
            candidates = [v for v in grid[key] if v != best_staking[key]]
            for val in candidates:
                if max_combos is not None and n_run >= max_combos:
                    break
                trial = {**best_params, **best_staking, key: val}
                row = run(trial, f"tune {key}={val}")
                n_run += 1
                if row is not None and _staking_score(row) > best_s + 1e-6:
                    best_s = _staking_score(row)
                    best_staking[key] = val

    full_best = {**best_params, **best_staking}
    print(f"  Final best combo: {full_best}")

    # ---- Report + optional config write ----
    report = pd.DataFrame(rows)
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    out_path = report_dir / f"calibration_{grid_name}.csv"
    report.to_csv(out_path, index=False)
    print(f"\nCalibration report saved to {out_path}")
    print("\nTop-5 by composite score:")
    score_col = (
        "residual_log_loss" if report["residual_log_loss"].notna().any() else "log_loss"
    )
    print(report.sort_values(score_col).head(5).to_string(index=False))

    if write_config:
        _write_config(full_best)
    return full_best, report


def _write_config(params: dict) -> None:
    path = Path("config.yaml")
    with open(path) as f:
        cfg_yaml = yaml.safe_load(f) or {}
    cfg_yaml.setdefault("model", {})
    changed = []
    for key, value in params.items():
        if key not in cfg_yaml["model"] or cfg_yaml["model"][key] != value:
            cfg_yaml["model"][key] = value
            changed.append(key)
    with open(path, "w") as f:
        yaml.safe_dump(cfg_yaml, f, sort_keys=False)
    if changed:
        print(f"\nconfig.yaml updated: {', '.join(changed)}")
    else:
        print("\nconfig.yaml already matches the recommended values.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--league", default="EPL", choices=["EPL", "LALIGA", "SERIEA"])
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--xg-dir", default=None, help="data/xg when xG should be enabled")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--grid", default="quick", choices=["quick", "full"])
    ap.add_argument("--write-config", action="store_true")
    ap.add_argument(
        "--min-train",
        type=int,
        default=760,
        help="Matches in the first training window (smaller = more windows = slower)",
    )
    ap.add_argument("--step", type=int, default=380)
    ap.add_argument(
        "--max-combos",
        type=int,
        default=None,
        help="Cap total backtests run (for quick iterations)",
    )
    args = ap.parse_args()

    cfg = load_config()
    if args.demo:
        df = generate_synthetic_league(n_seasons=4)
        xg_df = None
        print("Running on synthetic data (sanity check only).")
    else:
        df = load_league_csvs(args.data_dir, args.league)
        xg_df = load_league_xg(args.xg_dir, args.league) if args.xg_dir else None
        if xg_df is not None:
            df = join_xg(df, xg_df)
            print(f"Loaded {args.league}: {len(df)} matches + xG")

    calibrate(
        df,
        cfg,
        xg_df=xg_df,
        grid_name=args.grid,
        write_config=args.write_config,
        min_train_matches=args.min_train,
        step_matches=args.step,
        max_combos=args.max_combos,
    )


if __name__ == "__main__":
    main()
