"""
Calibrate the Elo draw probability function from empirical data.

The current Elo draw width (0.44) and clamping [0.12, 0.34] are heuristic
values. This script:
  1. Loads historical match data
  2. Bins matches by Elo rating gap (|diff|)
  3. Computes the empirical draw rate per bin
  4. Fits a function draw(diff) = a * (1 - min(|diff|/b, c))
  5. Outputs recommended draw_width and clamping bounds

Usage:
    python scripts/calibrate_elo_draw.py --league EPL --data-dir data/raw
    python scripts/calibrate_elo_draw.py --demo          # synthetic data
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# Make `src` importable when running directly: `python scripts/calibrate_elo_draw.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import generate_synthetic_league, load_league_csvs
from src.elo import EloEngine


def _draw_model(x, a, b, c):
    """Model: draw_rate = a * (1 - min(|x| / b, c)).
    x = rating gap, a = draw_width, b = gap where draw prob hits a*(1-c)."""
    return np.clip(a * (1.0 - np.minimum(np.abs(x) / b, c)), 0.0, 0.5)


def calibrate(df: pd.DataFrame, n_bins: int = 20):
    """Run Elo over the dataset, bin by rating gap, compute empirical draw %. Fit curve."""
    elo = EloEngine(k=20.0, home_advantage=75.0)
    matches = df.to_dict("records")

    gap_list = []
    draw_list = []

    for m in matches:
        rh = elo.get(m["home_team"]) + elo.home_advantage
        ra = elo.get(m["away_team"])
        gap = abs(rh - ra)
        gap_list.append(gap)
        draw_list.append(1.0 if m["home_goals"] == m["away_goals"] else 0.0)
        elo.update(
            m["date"], m["home_team"], m["away_team"], m["home_goals"], m["away_goals"]
        )

    gaps = np.array(gap_list)
    actual_draws = np.array(draw_list)

    # Bin by rating gap
    bins = np.linspace(0, gaps.max() + 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_draw_rates = np.array(
        [
            actual_draws[(gaps >= bins[i]) & (gaps < bins[i + 1])].mean()
            for i in range(n_bins)
        ]
    )

    # Fit curve
    valid = ~np.isnan(bin_draw_rates)
    popt, _ = curve_fit(
        _draw_model,
        bin_centers[valid],
        bin_draw_rates[valid],
        p0=[0.44, 800.0, 0.9],
        bounds=([0.0, 100.0, 0.0], [1.0, 2000.0, 1.0]),
    )
    a_fit, b_fit, c_fit = popt

    print("=" * 60)
    print("Elo Draw Probability Calibration")
    print("=" * 60)
    print(f"  Fitted draw_width (a):          {a_fit:.4f}")
    print(f"  Fitted gap scale (b):           {b_fit:.1f} Elo points")
    print(f"  Fitted clamp ratio (c):         {c_fit:.4f}")
    print(f"  Effective max gap:              {b_fit * c_fit:.0f} Elo points")
    print(f"  Theoretical max draw prob:      {a_fit:.2%}")
    print("  Recommended config.yaml values:")
    print(f"    elo_draw_width: {a_fit:.4f}")
    print(f"    # (clamping [a*(1-c), a] = [{a_fit*(1-c_fit):.2%}, {a_fit:.2%}])")
    print()

    # Print per-bin comparison
    print(f"{'Gap bin':>12s}  {'Empirical':>10s}  {'Fitted':>10s}  {'Residual':>10s}")
    print("-" * 46)
    for i in range(n_bins):
        if valid[i]:
            fitted = _draw_model(bin_centers[i], *popt)
            print(
                f"  {bin_centers[i]:6.0f} ± {bins[1]-bins[0]:3.0f}  {bin_draw_rates[i]:10.4f}  {fitted:10.4f}  {bin_draw_rates[i]-fitted:+10.4f}"
            )

    return {
        "draw_width": float(a_fit),
        "gap_scale": float(b_fit),
        "clamp_ratio": float(c_fit),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="EPL", choices=["EPL", "LALIGA", "SERIEA"])
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    df = (
        generate_synthetic_league(n_seasons=4)
        if args.demo
        else load_league_csvs(args.data_dir, args.league)
    )
    calibrate(df)
