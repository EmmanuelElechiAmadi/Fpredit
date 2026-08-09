# Pre-registered edge test protocol

Written before any further tuning or evaluation of the held-out season. This is
the test the whole pipeline exists to run. Registering the criterion *before*
looking at the held-out result is what turns "we found an edge" from a story
into a claim.

## Status

- **Date:** 2026-08-09
- **State:** pre-registered. The 2025-26 season in each league is **untouched**
  as an evaluation holdout. Do not train, tune, or select features on it.
- **Evidence locked at registration:** the walk-forward results on
  2021-22..2024-25 (the tuning window) show residual log loss **+0.038**
  (model worse than closing line), edge correlation **-0.095**, value-bet ROI
  **-12.7%**, Kelly Sharpe **-0.58**.

## Primary hypothesis

**H1:** The ensemble's probabilities add information beyond the closing line,
i.e. on the held-out season the model's mean residual log loss is negative:

```
residual_ll = mean(-ln q_model(y)) - mean(-ln q_market(y))   <  0
```

**H0:** no edge — residual_ll >= 0.

## Success criteria (pre-registered)

The test is a **pass** only if all of the following hold on the held-out
2025-26 season (~380 matches/league):

1. **Primary:** residual log loss <= **-0.005** (a small but non-trivial edge),
   with a one-sided 95% CI that excludes 0.
2. **Persistence:** the edge is negative in at least **2 of the 3 leagues**.
3. **Independent confirmation:** edge-win correlation **> +0.02** AND
   value-bet ROI **> 0** on the held-out window.

A result that fails any of the three is a **null / failure**, regardless of how
nice the component charts look.

## What is NOT allowed after seeing the holdout

- Picking a different threshold because the result was close.
- Tuning hyperparameters on the holdout and reporting the tuned number.
- Re-running the backtest with the holdout included in training.
- Dropping leagues from the average to make criterion 2 pass.

## Statistical power (computed from the tuning window)

- Per-match residual log loss SD ≈ **0.46**.
- With one season per league (n≈380): minimum detectable edge ≈ **5.8%**.
- With all three leagues' holdouts (n≈1,140): minimum detectable edge ≈ **3.3%**.
- To reliably detect a **2%** edge (80% power, alpha 0.05): ≈ **3,200** OOS
  matches; for a **1%** edge: ≈ **12,900**.

Consequences, pre-stated:

- The observed tuning-window residual (+0.038, SE≈0.012) is a *confident* null
  for edges >= ~3% — the model is significantly worse than the closing line,
  not "kind of maybe."
- A null on the single held-out season does **not** rule out a true 1-2% edge;
  it only rules out edges the sample has power to see. A final verdict on small
  edges requires ~3,200-12,900 OOS matches (roughly 8-34 league-seasons).
- Because the tuning-window edge correlation is already **negative and
  significant** (-0.095, p≈0.0002), a held-out *positive* edge correlation is
  the surprise that would justify further investment.

## Analysis plan (final evaluation, run once)

1. Train the ensemble on all data strictly before the 2025-26 holdout, with the
   config tuned on the 2022-25 window only.
2. Predict the holdout (walk-forward, as in `backtest.py --holdout-seasons 1`).
3. Compute residual_ll vs the B365 closing line (primary), edge correlation,
   value-bet ROI and Kelly Sharpe (secondary).
4. Compare against the thresholds above; report pass/fail/insufficient-power.
5. Publish the number either way. "No demonstrable edge" is a legitimate result.

## How to run it

```bash
# Tune ONLY on 2021-22..2024-25 (never touches 2025-26):
python scripts/calibrate_model.py --league EPL --xg-dir data/xg \
    --exclude-seasons 2025-26 --grid quick

# Single, pre-registered evaluation on the held-out season:
python backtest.py --league EPL --data-dir data/raw --xg-dir data/xg \
    --holdout-seasons 1 --output backtest_results_holdout.csv

# Power framing:
python scripts/power_analysis.py --results backtest_results_holdout.csv
```
