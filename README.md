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

Match-level xG from **Understat** (2014-15 onward, no API key):
```
python scripts/scrape_understat.py --league EPL --seasons 5   # caches data/xg/EPL/*.csv
python backtest.py --league EPL --xg-dir data/xg              # xG features enabled
python predict.py "Arsenal" "Chelsea" --xg-dir data/xg
```

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

# With market odds for a single fixture (residual-vs-market inference):
python predict.py "Arsenal" "Chelsea" --odds 2.1 3.4 3.6
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

When odds columns are present it also reports:
- **Residual log loss** (model − market; negative means the model adds
  information beyond the closing line),
- **Edge-win correlation** (does the model's predicted edge predict winning?),
- **Value-bet P&L** and the covariance-adjusted **Kelly staking report**
  (Sharpe, max drawdown, CVaR).

## Files

```
src/
  state_space.py     dynamic (Kalman-filtered) team strength — primary Poisson model
  dixon_coles.py     Dixon-Coles Poisson model (static baseline, shrinkage, xG fit)
  elo.py             Elo rating engine
  features.py        form + H2H + congestion + league position + PageRank features
  market.py          implied probabilities, line movement, value bets
  xg_loader.py       Understat xG scraper + cache + join helper
  staking.py         covariance-adjusted fractional Kelly + risk metrics
  ensemble.py        stacks everything into calibrated final probabilities
  data_loader.py     football-data.co.uk CSV loader + synthetic data generator
predict.py         CLI: input two teams, get a prediction (--odds, --xg-dir)
backtest.py        walk-forward evaluation harness (--xg-dir)
scripts/scrape_understat.py   fetch match-level xG into data/xg/
```

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
- xG coverage starts 2014-15; older seasons run goals-only (handled
  automatically by the fallback).
