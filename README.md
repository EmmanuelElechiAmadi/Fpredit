# Football match predictor — EPL / La Liga / Serie A

Input two teams, get calibrated probabilities. Not a magic "winning model" —
football is genuinely uncertain (a well-modeled 65% favorite still loses
about a third of the time). The goal here is the same one you apply to your
trading systems: **well-calibrated probabilities you can trust, backtested
honestly, with a clear edge over the naive baseline.**

## How it works

Three models are stacked into one ensemble (see the diagram above):

1. **Dixon-Coles bivariate Poisson** (`src/dixon_coles.py`) — the academic/
   industry-standard statistical baseline. Gives each team an attack and
   defense rating, models goals as time-decayed Poisson processes, and
   corrects for the known under-prediction of low scores (0-0, 1-0, 1-1).
   Outputs a full scoreline probability matrix, not just W/D/L.

2. **Elo ratings** (`src/elo.py`) — reacts faster to hot/cold streaks than
   Dixon-Coles, football-adapted (draws, margin-of-victory scaling, home
   advantage constant).

3. **Rolling form + head-to-head features** (`src/features.py`) — points per
   game over the last 5 matches, goal differential trend, rest days, H2H
   record. Every feature is computed strictly from information available
   *before* kickoff — no leakage.

A logistic regression meta-learner (`src/ensemble.py`) learns how much to
trust each signal and outputs final Home/Draw/Away probabilities, plus
expected goals, most likely scoreline, over/under 2.5, and BTTS.

## Getting real data (free, no API key)

Download historical CSVs from **football-data.co.uk**:
- Premier League: https://www.football-data.co.uk/englandm.php (code `E0`)
- La Liga: https://www.football-data.co.uk/spainm.php (code `SP1`)
- Serie A: https://www.football-data.co.uk/italym.php (code `I1`)

Place season files here:
```
data/raw/E0/2021-22.csv
data/raw/E0/2022-23.csv
data/raw/SP1/2022-23.csv
...
```
Aim for at least 5-6 seasons per league for the Dixon-Coles fit to be stable.

For upcoming *fixtures* (not historical results) and live odds, you'd want
an API like API-Football or the football-data.org API — those need paid/free
API keys I can't fetch from this sandbox, but `predict.py` is written to
accept any two team names, so once you have a fixture list you just loop
over it.

## Running it

```bash
pip install -r requirements.txt

# See it work immediately with generated synthetic data (no download needed):
python predict.py --demo
python backtest.py --demo

# With real data:
python predict.py "Arsenal" "Chelsea" --league EPL --data-dir data/raw
python backtest.py --league EPL --data-dir data/raw
```

Team names must match football-data.co.uk's spelling exactly (e.g. "Man
United" not "Manchester United") — check `model.dc.teams` after loading if
predictions error out on an unknown team.

## Always backtest before trusting a prediction

`backtest.py` runs a **walk-forward** evaluation: train on everything up to
a date, predict the next block of matches, roll forward, repeat. This is the
only honest way to test a time-series model — a random train/test split
leaks future form into the training set and will make the model look better
than it is. It reports log loss and Brier score against a constant-prior
baseline so you can see the actual uplift, not just accuracy (which is a
misleading metric here: always picking the home team already scores ~45-46%
in most leagues).

## Where the real edge comes from (next steps)

- **xG data** (understat.com, has free scrapeable pages) instead of just
  goals — shots quality is a better predictor than realized goals, which are
  noisy over a single match.
- **Player-level availability** (injuries/suspensions/rotation) — none of
  this is captured yet and it's often the single biggest single-match
  signal a model is missing.
- **Closing odds as a feature or calibration check** — the market is very
  efficient; if you have odds columns from football-data.co.uk, comparing
  your model's implied probability to the market's is the fastest way to
  tell if you have a real edge or are just reproducing consensus.
- **Re-fit cadence**: re-run Elo/Dixon-Coles weekly during a season; only
  promote a new ensemble version if it beats the current one on held-out
  log loss (see the backtest loop in the diagram above).

## Files

```
src/
  elo.py           Elo rating engine
  dixon_coles.py   Dixon-Coles Poisson model
  features.py      rolling form + head-to-head feature engineering
  ensemble.py      stacks the three into calibrated final probabilities
  data_loader.py   football-data.co.uk CSV loader + synthetic data generator
predict.py         CLI: input two teams, get a prediction
backtest.py        walk-forward evaluation harness
```
