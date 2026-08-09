"""
Statistical power for the pre-registered edge test.

Football outcomes are high-variance: a real 2-3% edge over the closing line
can hide inside sampling noise for a long time. This script computes, from the
*actual* walk-forward predictions and odds, the per-match variance of the
residual log loss and edge, then answers the two pre-registration questions:

  1. Given the data we already have, what is the smallest true edge we could
     plausibly detect (minimum detectable effect) at alpha=0.05, power=0.80?
  2. How many matches would a new-data effort need to reliably detect an edge
     of a given size (1%, 2%, 3%)?

Metrics analysed:
  - residual log loss  (model LL - market LL per match; negative = edge)
  - edge correlation   (Spearman between predicted edge and win indicator)
  - value-bet ROI      (per-bet return on flagged value bets)

Usage:
    python scripts/power_analysis.py --results /tmp/bt_epl_tuned.csv
    python scripts/power_analysis.py --results a.csv b.csv
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ALPHA = 0.05
POWER = 0.80
Z_ALPHA = 1.6448536269514722  # one-sided z(0.95)
Z_BETA = 0.8416212335729143  # z(0.80)


def _implied(row, cols) -> np.ndarray | None:
    vals = [row.get(c) for c in cols]
    if any(pd.isna(v) for v in vals):
        return None
    inv = 1.0 / np.array([float(v) for v in vals], dtype=float)
    return inv / inv.sum()


def _per_match_series(df: pd.DataFrame) -> pd.DataFrame:
    """Per-match residual contributions and edge values (the sampling unit)."""
    out = []
    for _, r in df.iterrows():
        mk = _implied(r, ("B365H", "B365D", "B365A"))
        if mk is None:
            continue
        y = {"H": 0, "D": 1, "A": 2}[r["actual"]]
        q = np.array(
            [r["pred_home_win"], r["pred_draw"], r["pred_away_win"]], dtype=float
        )
        # per-match residual: model LL contribution minus market LL contribution
        resid = -math.log(max(q[y], 1e-9)) + math.log(max(mk[y], 1e-9))
        edge = float(q[y] - mk[y])  # predicted minus implied prob on realized outcome
        out.append(
            {
                "date": r["date"],
                "resid": resid,
                "edge_on_winner": edge,
                "won": 1.0 if q[y] == q.max() else 0.0,
                "pred_edge": float(q.max() - mk[int(np.argmax(q))]),
            }
        )
    return pd.DataFrame(out)


def _min_n_for_effect(sigma: float, effect: float) -> float:
    """One-sample t-test: n needed to detect mean `effect` (signed) at alpha/power."""
    return ((Z_ALPHA + Z_BETA) * sigma / abs(effect)) ** 2


def _min_effect_for_n(sigma: float, n: int) -> float:
    return (Z_ALPHA + Z_BETA) * sigma / math.sqrt(n)


def _fisher_z_power_n(rho: float) -> float:
    """Approx. sample size to detect a correlation rho at alpha/power."""
    z = 0.5 * math.log((1 + rho) / (1 - rho))
    return ((Z_ALPHA + Z_BETA) / z) ** 2 + 3


def analyse(df: pd.DataFrame, label: str) -> dict:
    s = _per_match_series(df)
    if len(s) < 30:
        return {"label": label, "n_matches": len(s)}

    resid = s["resid"].to_numpy()
    sigma = float(resid.std(ddof=1))
    mean_resid = float(resid.mean())
    n = len(s)

    # Minimum detectable edge for THIS sample, and n needed for common edges.
    mde = _min_effect_for_n(sigma, n)
    n_1pct = _min_n_for_effect(sigma, 0.01)
    n_2pct = _min_n_for_effect(sigma, 0.02)
    n_3pct = _min_n_for_effect(sigma, 0.03)

    # Edge correlation: Spearman between predicted edge and win indicator.
    rho, p_val = _spearman(s["pred_edge"].to_numpy(), s["won"].to_numpy())
    n_corr = _fisher_z_power_n(abs(rho) if abs(rho) >= 0.03 else 0.03)

    return {
        "label": label,
        "n_matches": n,
        "mean_residual": round(mean_resid, 4),
        "per_match_sd": round(sigma, 4),
        "min_detectable_edge_1y": round(mde, 4),
        "n_for_1pct_edge": round(n_1pct, 0),
        "n_for_2pct_edge": round(n_2pct, 0),
        "n_for_3pct_edge": round(n_3pct, 0),
        "edge_corr": round(rho, 4),
        "edge_corr_p": round(p_val, 4),
        "n_for_corr_0.03": round(n_corr, 0),
    }


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    from scipy.stats import spearmanr

    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results", nargs="+", required=True, help="Backtest result CSV(s)"
    )
    ap.add_argument("--labels", nargs="*", default=None)
    args = ap.parse_args()

    rows = []
    combined = []
    for i, path in enumerate(args.results):
        df = pd.read_csv(path, parse_dates=["date"])
        df = df[df["pred_home_win"].notna() & df["B365H"].notna()].copy()
        label = (
            args.labels[i] if args.labels and i < len(args.labels) else Path(path).stem
        )
        rows.append(analyse(df, label))
        combined.append(df)

    print("\nPer-match residual log-loss variance and power requirements")
    print("(one-sided alpha=0.05, power=0.80)\n")
    cols = [
        "label",
        "n_matches",
        "mean_residual",
        "per_match_sd",
        "min_detectable_edge_1y",
        "n_for_1pct_edge",
        "n_for_2pct_edge",
        "n_for_3pct_edge",
        "edge_corr",
        "edge_corr_p",
        "n_for_corr_0.03",
    ]
    out = pd.DataFrame(rows)
    print(out[cols].to_string(index=False))

    if len(combined) > 1:
        pooled = pd.concat(combined, ignore_index=True)
        row = analyse(pooled, "ALL LEAGUES POOLED")
        print("\nPooled across leagues:")
        print(pd.DataFrame([row])[cols].to_string(index=False))

    print(
        "\nReading: n_for_Xpct_edge is how many out-of-sample matches a future "
        "data-collection effort would need to reliably detect a true edge of "
        "that size. If it exceeds the matches you can realistically gather, a "
        "negative result should be interpreted as 'insufficient power to find "
        "a small edge', not 'no edge exists'."
    )


if __name__ == "__main__":
    main()
