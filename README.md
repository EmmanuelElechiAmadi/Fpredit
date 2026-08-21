# Football match predictor — EPL / La Liga / Serie A

Input two teams, get calibrated probabilities. Not a magic "winning model" —
football is genuinely uncertain (a well-modeled 65% favorite still loses
about a third of the time). The goal here is the same one you apply to your
trading systems: **well-calibrated probabilities you can trust, backtested
honestly, with a clear edge over the naive baseline.**

## How it works

A stacked ensemble over five families of signal, blended by a logistic
regression meta-learner:

1. **Dynamic state-space (Kalman-filtered) team strength** (`src/state_space.py`)
   — each team's attack/defense evolves as a latent random walk (Rue &
   Salvesen 2000 style) and is filtered after every match. Filtered estimates
   are **leak-free by construction** (each match's rating uses only prior
   matches), which also fixes the subtle in-sample leakage a static fit
   produces when predicting its own data.
2. **Dixon-Coles bivariate Poisson** (`src/dixon_coles.py`) — the static
   statistical baseline; still used for the scoreline matrix, expected goals,
   over/under and BTTS outputs, and as a reported component. Supports ridge
   shrinkage toward the league mean (`dc_shrinkage`) for small-sample teams
   (newly promoted / mid-season replacements).
3. **Elo ratings** (`src/elo.py`) — reacts faster to hot/cold streaks.
4. **Context features** (`src/features.py`) — rolling form, head-to-head,
   fixture congestion ("3 in 8 days", cumulative load), league position
   (dead rubbers / six-pointers), and PageRank transitive strength. Every
   feature is computed strictly from information available *before* kickoff —
   no leakage.
5. **Market-residual layer** (`src/market.py`) — when odds columns are
   present, the closing implied probabilities (and opening→closing line
   movement, if available) are fed to the meta-learner, so it can only
   contribute signal the market has not yet priced. `predict()` accepts
   `market_odds` to apply this at inference time.
6. **xG features** (`src/xg_loader.py`) — match-level expected goals scraped
   from Understat (2014-15 onward), fed to a second dynamic model filtered on
   the far less noisy xG observation. Falls back to goals-only when absent.

A covariance-adjusted fractional-Kelly staking layer (`src/staking.py`) sizes
a slate of bets as a portfolio — with a shrunk structural covariance between
correlated matches — and reports Sharpe, max drawdown and CVaR.

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

Match-level xG from **Understat** (2014-15 onward, no API key). The team-name
map covers all three leagues, so a single command per league caches clean
`data/xg/<LEAGUE>/<season>.csv` files that join onto the results by
(date, home_team, away_team) — including fixtures off by a day (timezone /
postponed matches, joined within a ±14-day window):
```
python scripts/scrape_understat.py --league EPL --seasons 5      # data/xg/EPL/
python scripts/scrape_understat.py --league LALIGA --seasons 5   # data/xg/LALIGA/
python scripts/scrape_understat.py --league SERIEA --seasons 5   # data/xg/SERIEA/
python backtest.py --league EPL --xg-dir data/xg                 # xG features enabled
python predict.py "Arsenal" "Chelsea" --xg-dir data/xg
```

## New-season fixtures (who's playing next)

Before the season starts there is nothing to predict yet, so the web UI has a
dedicated **Fixtures** view that shows *upcoming* matches and lets you run the
ensemble on any of them:

- **Add fixtures** — pick a date, home/away team and (optionally) matchweek from
  the league's current team list. Fixtures are stored as CSVs in
  `data/fixtures/<LEAGUE_CODE>/<season>.csv`.
- **Drop in the official file** — when football-data.co.uk posts the new
  season's CSV, just save it as `data/fixtures/E0/2026-27.csv` (their native
  `Date,HomeTeam,AwayTeam,...` layout is understood; so is a minimal
  `date,home_team,away_team[,matchweek]` layout). It is picked up automatically.
- **Predict** — each fixture row has a **🎯 Predict** button that jumps to the
  Match Predictor pre-loaded with those two teams, and a **⚡ Predict all**
  button that runs the whole slate and renders an H/D/A + xG table inline.
- **Placeholder round-robin** — with no official file yet, you can generate a
  full double round-robin schedule from the league's current 20 teams (380
  matches, 38 matchweeks) as an explicit stand-in, then replace rows once the
  real fixtures are out.

Newly promoted / unknown teams are flagged with a *new* badge and predicted
from the league-mean prior (the market features then carry the fixture when
odds are supplied).

`--seasons N` always means "the N most recent *complete* seasons" (in August
2026 that is 2021-22 .. 2025-26). Use `--from 2021 --to 2025` for an explicit
range.

Odds columns: recent football-data.co.uk files carry `B365H/D/A` and Pinnacle
closing `PSH/PSD/PSA`, but **not** the BetBrain averages (`BbAv*`) or Pinnacle
opening lines (`PH/PD/PA`). The market-residual layer therefore uses the B365
closing line. The opening→closing **line-movement features cannot be activated
from football-data.co.uk at all** — a full sweep of E0 files from 2005-06 to
2025-26 found no season with `PH/PD/PA` (only closing `PSH/PSD/PSA`, from
2012-13 onward). Enabling line movement requires a different source that
publishes opening 1X2 odds (e.g. OddsPortal scrapes or an odds API).

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

# Web app (served at http://localhost:8000):
make web            # or: uvicorn app.main:app --reload

# Deploy to Railway / Render / Hugging Face Spaces (Dockerfile included):
#   see docs/DEPLOY.md

# With real data:
python predict.py "Arsenal" "Chelsea" --league EPL --data-dir data/raw
python backtest.py --league EPL --data-dir data/raw

# Tune the model + staking hyperparameters on real data (walk-forward OOS):
python scripts/calibrate_model.py --league EPL --xg-dir data/xg --grid quick
python scripts/calibrate_model.py --league EPL --xg-dir data/xg --write-config

# With market odds for a single fixture (residual-vs-market inference):
python predict.py "Arsenal" "Chelsea" --odds 2.1 3.4 3.6
```

Team names must match football-data.co.uk's spelling exactly (e.g. "Man
United" not "Manchester United") — check `model.dc.teams` after loading if
predictions error out on an unknown team. Teams with no matches in the
training window (newly promoted sides) are handled with a league-mean prior:
the state-space and Dixon-Coles models emit neutral ratings and the closing
line carries the fixture, so their matches are still part of the backtest
rather than being dropped.

## Always backtest before trusting a prediction

`backtest.py` runs a **walk-forward** evaluation: train on everything up to
a date, predict the next block of matches, roll forward, repeat. This is the
only honest way to test a time-series model — a random train/test split
leaks future form into the training set and will make the model look better
than it is. It reports log loss and Brier score against a constant-prior
baseline so you can see the actual uplift, not just accuracy (which is a
misleading metric here: always picking the home team already scores ~45-46%
in most leagues).

When odds columns are present it also reports:
- **Residual log loss** (model − market; negative means the model adds
  information beyond the closing line),
- **Edge-win correlation** (does the model's predicted edge predict winning?),
- **Value-bet P&L** and the covariance-adjusted **Kelly staking report**
  (Sharpe, max drawdown, CVaR).

## Tuning

`scripts/calibrate_model.py` runs the same walk-forward backtest over a grid of
model and staking hyperparameters and reports honest out-of-sample metrics per
combo (overall log loss, residual-vs-market log loss, edge correlation, Kelly
Sharpe). Run it on real data and it prints the best combo; add `--write-config`
to write it into `config.yaml`:

```
python scripts/calibrate_model.py --league EPL --xg-dir data/xg --grid quick
```

On the current 5-season EPL set (walk-forward, ~1140 OOS matches) the tuned
combo (`dc_xi=0.0035`, `dc_shrinkage=0.05`, `ss_q_xg=0.01`, `meta_C=0.5`,
`kelly_corr=0.0`) improved overall log loss 1.0053 → 1.0025 and residual log
loss +0.0392 → +0.0364. Note the honest headline: the model still does not
beat the closing line out-of-sample — the residual stays positive — but it
clearly beats the constant-prior baseline and the tuned staking config reports
a better Sharpe than the shipped defaults.

## Sanity-checking the negative result

Before trusting "no edge," run the market-only control through the same
pipeline — a trivial strategy with no model opinion at all:

```
python scripts/market_control.py --results backtest_results.csv
python scripts/market_control.py --league EPL --xg-dir data/xg   # or run a backtest inline
```

On the current EPL set the control returns exactly what a working pipeline
should: flat-betting the **market favorite loses ~-3.6%** (the ~5.5% margin,
favorite-weighted), while the **model favorite loses ~-4.0%** and the model's
value-bet Kelly loses ~-12.7%. Because the trivial control behaves as expected,
the model's worse result is a genuine signal (its "value" calls are wrong, not
a pipeline bug), and the residual against **Pinnacle close** is also positive
(~+0.038) — the model fails the sharper-line test too.

## The pre-registered edge test

The question this pipeline exists to answer — *does the model beat the closing
line out-of-sample?* — is now run under a pre-registered protocol, not an
after-the-fact backtest:

- **Protocol:** `docs/edge_test_preregistration.md` — hypothesis, success
  thresholds (residual log loss ≤ −0.005 on the held-out season, persistence
  across leagues, positive edge correlation), and what is forbidden after
  seeing the result (re-picking thresholds, tuning on the holdout, dropping
  leagues).
- **Holdout:** the most recent season is never used for training, tuning, or
  feature selection. Evaluate it exactly once:
  ```bash
  python backtest.py --league EPL --xg-dir data/xg --holdout-seasons 1 \
      --output backtest_results_holdout.csv
  # tuning must exclude the same season:
  python scripts/calibrate_model.py --league EPL --xg-dir data/xg \
      --exclude-seasons 2025-26 --grid quick
  ```
- **Power:** `scripts/power_analysis.py` computes the per-match residual
  variance and how many matches are needed to detect a given edge. On the
  current EPL set the residual SD is ~0.46: a 1% edge needs ~12,900 OOS
  matches, a 2% edge ~3,200, a 3% edge ~1,400. The observed +0.038 residual is
  a *confident* null for edges ≥ ~3% (≈3.2 SE from zero) — but the sample
  cannot rule out a true 1–2% edge.

**Result to date (pre-registered holdout, 2025-26 EPL, 380 matches):** residual
log loss **+0.034**, edge correlation **−0.19**, value-bet ROI **−8.8%** —
fails all three pre-registered criteria. This is a clean, falsifiable null:
the model is well-calibrated context but not a source of edge against the
closing line.

## Files

```
src/
  state_space.py     dynamic (Kalman-filtered) team strength — primary Poisson model
  dixon_coles.py     Dixon-Coles Poisson model (static baseline, shrinkage, xG fit)
  elo.py             Elo rating engine
  features.py        form + H2H + congestion + league position + PageRank features
  market.py          implied probabilities, line movement, value bets
  xg_loader.py       Understat xG scraper + cache + join helper (all 3 leagues)
  staking.py         covariance-adjusted fractional Kelly + risk metrics
  ensemble.py        stacks everything into calibrated final probabilities
  data_loader.py     football-data.co.uk CSV loader + synthetic data generator
predict.py         CLI: input two teams, get a prediction (--odds, --xg-dir)
backtest.py        walk-forward evaluation harness (--xg-dir)
scripts/scrape_understat.py   fetch match-level xG into data/xg/ (all leagues)
scripts/calibrate_model.py    walk-forward hyperparameter tuning (ss_q, shrinkage, staking)
scripts/market_control.py     sanity-check control: bet-the-market vs model strategies
scripts/power_analysis.py     statistical power / minimum detectable edge for the edge test
app/main.py        FastAPI backend for the web UI (surfaces every metric)
webapp/static/     single-page frontend (index.html, app.js, styles.css)
```

## Web UI

`make web` (or `uvicorn app.main:app --reload`) serves a single-page app at
`http://localhost:8000`:

- **Dashboard** — standings, outcome mix, goals-per-game trend, form, xG badge.
- **Fixtures** — who's playing next (see *New-season fixtures* above); add/remove
  fixtures and run the model on any match or the whole slate.
- **Team Intelligence** — Elo, Dixon-Coles attack/defense **and** dynamic
  (Kalman-filtered) attack/defense ratings per team.
- **Match Predictor** — H/D/A, expected goals, most-likely score, score matrix,
  **component breakdown** (Dixon-Coles / Elo / Dynamic / xG filter), dynamic xG,
  and a **market-vs-model edge** panel from the last meeting's closing line.
- **Backtest Lab** — walk-forward metrics, **residual log loss**, **edge
  correlation**, **model-vs-market log loss**, and the full **staking report**
  (Sharpe, max drawdown, CVaR, ROI, profit units).
- **League Compare** — per-league log loss / Brier plus residual log loss, edge
  correlation and Kelly Sharpe/ROI across leagues.

## Known limitations / future work

- **Lineup-aware ratings** — the single biggest professional-vs-public gap —
  are not yet implemented: they need lineup + player-rating data and are a
  substantial project on their own.
- **In-play / Hawkes-style momentum models** are deliberately out of scope
  for the current pre-match pipeline.
- **Travel distance / European-fixture context**, referee tendencies, weather
  and manager tenure are recognised signal but need data sources not in the
  current free pipeline (venue geography, UEFA fixtures, referee lists,
  weather APIs, coaching changes).
- **Line movement** (`PH/PD/PA` Pinnacle opening odds) is not present in
  recent football-data.co.uk files, so the opening→closing line-movement
  features are dormant; drop in older/higher-tier files that carry those
  columns to activate them.
- **Newly promoted teams** have no training history in the earliest walk-forward
  window; the state-space and Dixon-Coles models fall back to league-mean
  ratings for them, so their fixtures are still predicted (largely off the
  closing line) instead of being skipped.
- **Hyperparameter defaults** (`ss_q`, `dc_shrinkage`, `kelly_corr`, ...) are
  the shipped values; `scripts/calibrate_model.py` tunes them on real data via
  walk-forward CV and can write the winners straight into `config.yaml`.
- xG coverage starts 2014-15; older seasons run goals-only (handled
  automatically by the fallback).
