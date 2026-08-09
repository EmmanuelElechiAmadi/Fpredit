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

Usage: python backtest.py --demo
"""

import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from src.data_loader import generate_synthetic_league, load_league_csvs
from src.ensemble import FootballEnsemble

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def walk_forward_backtest(df: pd.DataFrame, min_train_matches=380, step_matches=190):
    df = df.sort_values("date").reset_index(drop=True)
    records = []

    cutoff = min_train_matches
    while cutoff + step_matches <= len(df):
        train = df.iloc[:cutoff]
        test = df.iloc[cutoff : cutoff + step_matches]

        model = FootballEnsemble()
        model.fit(train)

        for _, row in test.iterrows():
            try:
                p = model.predict(
                    row["home_team"], row["away_team"], as_of_date=row["date"]
                )
                records.append(
                    {
                        "date": row["date"],
                        "home_team": row["home_team"],
                        "away_team": row["away_team"],
                        "actual": row["result"],
                        "pred_home_win": p["home_win"],
                        "pred_draw": p["draw"],
                        "pred_away_win": p["away_win"],
                        "pred_over_2_5": p["over_2_5_goals"],
                        "pred_btts_yes": p["btts_yes"],
                        "expected_home_goals": p["expected_goals"][0],
                        "expected_away_goals": p["expected_goals"][1],
                    }
                )
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


def evaluate(result_df: pd.DataFrame):
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

    return {
        "log_loss": ll,
        "brier": brier,
        "accuracy": acc,
        "baseline_log_loss": baseline_ll,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="EPL", choices=["EPL", "LALIGA", "SERIEA"])
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument(
        "--output",
        default=None,
        help="Path to save detailed predictions CSV (e.g. backtest_results.csv)",
    )
    args = ap.parse_args()

    df = (
        generate_synthetic_league(n_seasons=4)
        if args.demo
        else load_league_csvs(args.data_dir, args.league)
    )
    result_df = walk_forward_backtest(df)
    if result_df.empty:
        print(
            "No predictions could be made. See warnings above (run with --verbose "
            "or check the data)."
        )
        raise SystemExit(1)
    metrics = evaluate(result_df)

    if args.output:
        result_df.to_csv(args.output, index=False)
        print(f"\nDetailed predictions saved to: {args.output}")
