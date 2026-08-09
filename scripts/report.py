#!/usr/bin/env python3
"""
Generate a detailed prediction report and calibration dashboard.

Produces:
  - A CSV of all predictions with actual results
  - A terminal summary table of metrics
  - Calibration plots (reliability diagrams) saved as PNG
  - P&L if you had bet on every match with model-implied odds

Usage:
    python scripts/report.py --demo
    python scripts/report.py --league EPL --data-dir data/raw --output-dir reports
"""
import argparse
import sys
import warnings
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

# Make `src` importable when running directly: `python scripts/report.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import generate_synthetic_league
from src.log import get_logger

log = get_logger(__name__)


def _brier_decomposition(y_true, probs):
    """Return refinement, calibration, uncertainty components of Brier score."""
    n = len(y_true)
    uncertainty = np.mean(y_true) * (1 - np.mean(y_true))
    calibration = 0.0
    refinement = 0.0
    bins = np.linspace(0, 1, 11)
    for i in range(len(bins) - 1):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        obs = y_true[mask].mean()
        pred = probs[mask].mean()
        weight = mask.sum() / n
        calibration += weight * (obs - pred) ** 2
        refinement += weight * obs * (1 - obs)
    return refinement, calibration, uncertainty


def generate_report(
    df: pd.DataFrame,
    result_df: pd.DataFrame,
    output_dir: str = "reports",
    league: str = "SYNTHETIC",
) -> dict:
    """Run metrics and optionally save plots/reports.

    Parameters
    ----------
    df : pd.DataFrame
        Full match dataframe (needed for class priors).
    result_df : pd.DataFrame
        Backtest results with actual, pred_home_win, pred_draw, pred_away_win.
    output_dir : str
        Directory for report files (CSV, plots).
    league : str
        League name for labeling.

    Returns
    -------
    dict of metrics
    """
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save raw predictions
    csv_path = out / f"predictions_{league}.csv"
    result_df.to_csv(csv_path, index=False)
    log.info("Predictions saved to %s", csv_path)

    label_map = {"H": 0, "D": 1, "A": 2}
    y_true = np.array([label_map[a] for a in result_df["actual"]])
    prob_cols = ["pred_home_win", "pred_draw", "pred_away_win"]
    preds = result_df[prob_cols].values
    y_pred_class = preds.argmax(axis=1)

    # Core metrics
    ll = log_loss(y_true, preds, labels=[0, 1, 2])
    acc = accuracy_score(y_true, y_pred_class)

    # Brier components
    brier_home = brier_score_loss((y_true == 0).astype(int), preds[:, 0])
    brier_draw = brier_score_loss((y_true == 1).astype(int), preds[:, 1])
    brier_away = brier_score_loss((y_true == 2).astype(int), preds[:, 2])
    brier_avg = (brier_home + brier_draw + brier_away) / 3.0

    # Baseline comparison
    class_rates = np.bincount(y_true, minlength=3) / len(y_true)
    baseline_preds = np.tile(class_rates, (len(y_true), 1))
    baseline_ll = log_loss(y_true, baseline_preds, labels=[0, 1, 2])

    # Calibration (for home win binning)
    _, cal_home, _ = _brier_decomposition((y_true == 0).astype(int), preds[:, 0])

    # P&L with simple staking: bet 1 unit on the model's highest confidence pick
    # where confidence > 50%. Flat stake, no Kelly.
    confidence = preds.max(axis=1)
    high_conf = confidence > 0.50
    n_bets = high_conf.sum()
    if n_bets > 0:
        bets_won = (y_pred_class[high_conf] == y_true[high_conf]).sum()
        roi = (bets_won / n_bets) * 2.0 - 1.0  # odds of 2.0 (even money)
    else:
        bets_won = 0
        roi = 0.0

    metrics = {
        "league": league,
        "n_matches": len(y_true),
        "log_loss": ll,
        "baseline_log_loss": baseline_ll,
        "log_loss_improvement": baseline_ll - ll,
        "brier_home": brier_home,
        "brier_draw": brier_draw,
        "brier_away": brier_away,
        "brier_avg": brier_avg,
        "calibration_error_home": cal_home,
        "accuracy": acc,
        "baseline_home_accuracy": class_rates[0],
        "high_confidence_bets": n_bets,
        "high_confidence_wins": int(bets_won),
        "roi_at_even_money": roi,
    }

    _print_summary(metrics)
    _save_summary_csv(metrics, out / f"metrics_{league}.csv")

    try:
        _plot_calibration(y_true, preds, str(out / f"calibration_{league}.png"))
    except ImportError:
        log.warning("matplotlib not installed; skipping calibration plot")

    return metrics


def _print_summary(m: dict):
    print(f"\n{'='*55}")
    print(f"  REPORT: {m['league']} ({m['n_matches']:,} matches)")
    print(f"{'='*55}")
    print(f"  Log loss:           {m['log_loss']:.4f}")
    print(f"  Baseline log loss:  {m['baseline_log_loss']:.4f}")
    print(f"  Improvement:        {m['log_loss_improvement']:+.4f}")
    print(f"  Brier score (avg):  {m['brier_avg']:.4f}")
    print(f"  Calibration error:  {m['calibration_error_home']:.4f}")
    print(f"  Accuracy:           {m['accuracy']:.1%}")
    print(f"  Baseline (home):    {m['baseline_home_accuracy']:.1%}")
    print(
        f"  High-conf bets:     {m['high_confidence_bets']} ({m['high_confidence_wins']} won)"
    )
    print(f"  ROI (even money):   {m['roi_at_even_money']:+.1%}")
    print(f"{'='*55}\n")


def _save_summary_csv(metrics: dict, path: Union[str, Path]):
    pd.DataFrame([metrics]).to_csv(path, index=False)
    log.info("Metrics saved to %s", path)


def _plot_calibration(y_true, probs, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    labels = ["Home Win", "Draw", "Away Win"]
    for k in range(3):
        ax = axes[k]
        target = (y_true == k).astype(int)
        bins = np.linspace(0, 1, 11)
        means = []
        mids = []
        for i in range(len(bins) - 1):
            mask = (probs[:, k] >= bins[i]) & (probs[:, k] < bins[i + 1])
            if mask.sum() < 5:
                continue
            means.append(target[mask].mean())
            mids.append((bins[i] + bins[i + 1]) / 2)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        if means:
            ax.plot(mids, means, "o-", label="Model")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(labels[k])
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    log.info("Calibration plot saved to %s", path)


def main():
    ap = argparse.ArgumentParser(
        description="Generate prediction report and calibration dashboard"
    )
    ap.add_argument("--league", default="EPL", choices=["EPL", "LALIGA", "SERIEA"])
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--output-dir", default="reports")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        df = generate_synthetic_league(n_teams=6, n_seasons=2)
    else:
        from src.data_loader import load_league_csvs

        df = load_league_csvs(args.data_dir, args.league)

    from backtest import walk_forward_backtest

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result_df = walk_forward_backtest(
            df, min_train_matches=len(df) // 2, step_matches=len(df) // 4
        )
        if len(result_df) < 10:
            log.warning(
                "Not enough test matches (%d) for meaningful report; using full backtest",
                len(result_df),
            )
            result_df = walk_forward_backtest(df)
        metrics = generate_report(
            df, result_df, args.output_dir, args.league if not args.demo else "DEMO"
        )

    return metrics


if __name__ == "__main__":
    main()
