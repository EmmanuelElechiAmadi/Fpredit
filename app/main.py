"""
Football Predictor Web App — FastAPI backend.

Serves a modern single-page UI on top of the existing prediction pipeline:
  GET  /api/leagues                    — list available leagues + metadata
  GET  /api/dashboard?league=EPL       — standings, form, plots data
  GET  /api/teams?league=EPL           — team intelligence (ratings, form, fixtures)
  GET  /api/fixtures                   — upcoming fixtures + team list (new season)
  POST /api/fixtures                   — add an upcoming fixture
  DELETE /api/fixtures                 — remove an upcoming fixture
  POST /api/fixtures/generate-round-robin — build a placeholder double round-robin
  GET  /api/predict                    — H/D/A probs + expected goals + market
  GET  /api/backtest?league=EPL        — walk-forward metrics vs baselines
  GET  /api/calibrate?league=EPL       — calibrate Elo draw width
  GET  /api/compare                   — model vs market across leagues
  GET  /api/health                     — health check

Run:  uvicorn app.main:app --reload
"""

from __future__ import annotations

import functools
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import load_config
from src.data_loader import LEAGUE_CODES, load_league_csvs
from src.elo import EloEngine
from src.fixtures import (
    add_fixture,
    delete_fixture,
    generate_round_robin,
    load_fixtures,
    write_fixtures_batch,
)
from src.market import add_implied_probabilities
from src.model_factory import build_ensemble
from src.xg_loader import join_xg, load_league_xg

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("webapp")

CACHE_TTL = 60 * 5  # 5 minutes for data-derived endpoint results
_BT_CACHE_TTL = 60 * 30  # backtests are expensive; hold for 30 minutes

# (league, min_train, step, invalidation_hash) -> (computed_at, payload, result_df)
_BT_CACHE: dict[tuple, tuple[float, dict, pd.DataFrame]] = {}
# Walk-forward backtests are expensive (~1-2 min); serialize concurrent
# requests so the warm-up thread and user requests don't each recompute the
# same league simultaneously and thrash the CPU.
_BT_LOCK = threading.Lock()

# Fitted ensemble per (league, data-version) — fitting takes ~15s per league
# and the old code refit on every /api/teams and /api/predict request.
_MODEL_CACHE: dict[tuple, object] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Startup/shutdown hooks (replaces the deprecated ``@app.on_event``)."""
    _warm_caches()
    yield


app = FastAPI(title="Football Predictor API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the built frontend (webapp/static/) at /
STATIC_DIR = Path(__file__).resolve().parent.parent / "webapp" / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


def _cfg():
    return load_config()


def _league_meta(league: str) -> dict:
    return {
        "EPL": {"label": "Premier League", "code": "E0", "country": "England"},
        "LALIGA": {"label": "La Liga", "code": "SP1", "country": "Spain"},
        "SERIEA": {"label": "Serie A", "code": "I1", "country": "Italy"},
    }[league]


def _normalize_metrics(metrics: dict) -> dict:
    """Flatten backtest metrics into the shape the web frontend expects.

    The core backtest module uses its own dict layout (market.log_loss_model,
    market.log_loss_market, separate value_bets dict).  We re-map to a
    UI-friendly contract so the frontend stays simple:

      metrics.market.market_log_loss            — bookmaker implied log loss
      metrics.market.model_minus_market_log_loss — model − market (negative = better)
      metrics.market.value_bets                  — number of flagged value bets
      metrics.market.value_bet_yield             — ROI per bet (profit units / n bets)
      metrics.edge_corr                          — does predicted edge predict winning?
      metrics.staking                            — covariance-adjusted Kelly report
    """
    out = {
        "log_loss": metrics["log_loss"],
        "brier": metrics["brier"],
        "accuracy": metrics["accuracy"],
        "baseline_log_loss": metrics["baseline_log_loss"],
    }
    mkt = metrics.get("market") or {}
    vb = metrics.get("value_bets") or {}
    out["market"] = {
        "market_log_loss": mkt.get("log_loss_market"),
        "model_log_loss": mkt.get("log_loss_model"),
        "residual_log_loss": mkt.get("residual_log_loss"),
        "brier_model": mkt.get("brier_model"),
        "brier_market": mkt.get("brier_market"),
        "n": mkt.get("n", 0),
        "beats_market": mkt.get("beats_market"),
        "calibration": mkt.get("calibration", []),
        "value_bets": int(vb.get("n_bets", 0)),
        "value_bet_yield": vb.get("roi") if vb.get("n_bets") else None,
        "strike_rate": vb.get("strike_rate"),
        "profit_units": vb.get("profit_units"),
    }
    if out["market"]["market_log_loss"] is not None:
        out["market"]["model_minus_market_log_loss"] = round(
            mkt["log_loss_model"] - mkt["log_loss_market"], 4
        )
    else:
        out["market"]["model_minus_market_log_loss"] = None

    # Edge correlation + the covariance-adjusted Kelly staking report
    out["edge_corr"] = metrics.get("edge_corr")
    st = metrics.get("staking") or {}
    out["staking"] = {
        "n": st.get("n", 0),
        "total_staked": st.get("total_staked", 0.0),
        "profit_units": st.get("profit_units", 0.0),
        "roi": st.get("roi"),
        "sharpe": st.get("sharpe"),
        "max_drawdown": st.get("max_drawdown", 0.0),
        "cvar95": st.get("cvar95"),
    }
    return out


@functools.lru_cache(maxsize=16)
def _load_cached(league: str, raw_dir: str) -> pd.DataFrame:
    df = load_league_csvs(raw_dir, league)
    df = add_implied_probabilities(df)
    return df


@functools.lru_cache(maxsize=16)
def _load_xg_cached(league: str, xg_dir: str = "data/xg") -> pd.DataFrame | None:
    """Load cached Understat xG for a league, or None when absent (offline-safe)."""
    try:
        return load_league_xg(xg_dir, league)
    except FileNotFoundError:
        return None


def _xg_available(league: str) -> bool:
    xg = _load_xg_cached(league)
    return xg is not None and not xg.empty


def _get_model_cached(league: str):
    """Fit the ensemble once per (league, data-version) and reuse it.

    Previously every /api/teams and /api/predict request refit the model on the
    full league history (~15s each). Keying on the data fingerprint (config +
    raw data + xG mtimes) keeps the cache valid across data updates.
    """
    cfg = _cfg()
    key = (league, _data_fingerprint(league))
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    df = _load_cached(league, cfg.data.data_dir)
    xg_df = _load_xg_cached(league)
    if xg_df is not None and not xg_df.empty:
        df = join_xg(df, xg_df)
    model = build_ensemble(cfg)
    model.fit(df.sort_values("date"))
    _MODEL_CACHE[key] = model
    return model


def _warm_caches() -> None:
    """Background warm-up so the first user interaction is fast, not a 15s+ spinner."""
    import threading

    def _warm() -> None:
        for league in LEAGUE_CODES:
            try:
                _get_model_cached(league)
                log.info("Warmed model cache for %s", league)
            except Exception as e:  # pragma: no cover - defensive
                log.warning("Warm-up failed for %s: %s", league, e)
        try:
            # Pre-compute the default Backtest Lab result (EPL) too.
            _get_backtest_cached("EPL", None, None)
            log.info("Warmed default backtest for EPL")
        except Exception as e:  # pragma: no cover - defensive
            log.warning("Backtest warm-up failed: %s", e)

    threading.Thread(target=_warm, daemon=True, name="cache-warm").start()


def _compute_backtest(df, xg_df, min_train, step, cfg, bt) -> pd.DataFrame:
    return bt.walk_forward_backtest(
        df,
        min_train_matches=min_train,
        step_matches=step,
        cfg=cfg,
        xg_df=xg_df,
    )


def _get_backtest_cached(
    league: str, min_train_matches: int | None, step_matches: int | None
) -> tuple[pd.DataFrame, dict]:
    """Run (or reuse the cached) walk-forward backtest; return (result_df, payload).

    Used by both /api/backtest and /api/research so the expensive computation
    runs once per (league, window, data-version) key.
    """
    import importlib

    bt = importlib.import_module("backtest")
    cfg = _cfg()
    df = _load_cached(league, cfg.data.data_dir)
    xg_df = _load_xg_cached(league)

    min_train = min_train_matches or cfg.backtest.min_train_matches
    step = step_matches or cfg.backtest.step_matches

    cache_key = (league, min_train, step, _data_fingerprint(league))
    now = time.time()
    hit = _BT_CACHE.get(cache_key)
    if hit and now - hit[0] < _BT_CACHE_TTL:
        return hit[2], hit[1]

    with _BT_LOCK:
        # Re-check under the lock — the warm-up thread or another request may
        # have computed this exact key while we waited.
        now = time.time()
        hit = _BT_CACHE.get(cache_key)
        if hit and now - hit[0] < _BT_CACHE_TTL:
            return hit[2], hit[1]

        result_df = _compute_backtest(df, xg_df, min_train, step, cfg, bt)
        if result_df.empty:
            raise ValueError("Backtest produced no predictions.")

        metrics = _normalize_metrics(bt.evaluate(result_df, with_odds=True))

        # Recent flagged value bets (most interesting for a human reviewer).
        recent_value_bets = []
        if not result_df.empty and "B365H" in result_df.columns:
            from src.market import value_bets

            vb_df = result_df.assign(result=result_df["actual"])
            mp = result_df[["pred_home_win", "pred_draw", "pred_away_win"]].rename(
                columns={
                    "pred_home_win": "home_win",
                    "pred_draw": "draw",
                    "pred_away_win": "away_win",
                }
            )
            vb = value_bets(vb_df, mp)
            if not vb.empty:
                recent_value_bets = (
                    vb.sort_values("date")
                    .tail(25)[
                        [
                            "date",
                            "home_team",
                            "away_team",
                            "market",
                            "model_prob",
                            "implied_prob",
                            "edge",
                            "odds",
                            "pnl",
                        ]
                    ]
                    .assign(
                        date=lambda d: d["date"].dt.strftime("%Y-%m-%d"),
                        model_prob=lambda d: d["model_prob"].round(3),
                        implied_prob=lambda d: d["implied_prob"].round(3),
                        edge=lambda d: d["edge"].round(3),
                        odds=lambda d: d["odds"].round(2),
                        pnl=lambda d: d["pnl"].round(2),
                    )
                    .to_dict("records")
                )

        # Monthly accuracy curve
        result_df = result_df.sort_values("date")
        result_df["month"] = result_df["date"].dt.to_period("M").astype(str)
        monthly_acc = (
            result_df.groupby("month")
            .apply(
                lambda g: float(
                    (
                        g["pred_home_win"].gt(g["pred_draw"])
                        & g["pred_home_win"].gt(g["pred_away_win"])
                    ).mean()
                )
            )
            .round(4)
            .to_dict()
        )

        payload = {
            "metrics": metrics,
            "n_matches": len(result_df),
            "monthly_accuracy": monthly_acc,
            "recent_value_bets": recent_value_bets,
        }
        _BT_CACHE[cache_key] = (now, payload, result_df)
        return result_df, payload


def _data_fingerprint(league: str) -> tuple[str, str, str]:
    """mtime-based invalidation key: config + league raw data + league xG.

    Re-running the expensive walk-forward backtest on every page load wastes
    ~1-2 minutes of the experimentation loop, so repeat calls hit a short-TTL
    cache that is invalidated whenever the config or the underlying data
    changes (file mtimes).
    """

    def _mtime(p: Path) -> str:
        try:
            return str(p.stat().st_mtime)
        except OSError:
            return ""

    return (
        _mtime(Path("config.yaml")),
        _mtime(Path("data/raw") / LEAGUE_CODES[league]),
        _mtime(Path("data/xg") / league),
    )


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "football-predictor"}


@app.get("/api/leagues")
def leagues():
    cfg = _cfg()
    out = []
    for code, folder_code in LEAGUE_CODES.items():
        folder = Path(cfg.data.data_dir) / folder_code
        files = sorted(folder.glob("*.csv"))
        seasons = [f.stem for f in files]
        meta = _league_meta(code)
        out.append(
            {
                "code": code,
                "folder": folder_code,
                "label": meta["label"],
                "country": meta["country"],
                "seasons": seasons,
                "n_seasons": len(seasons),
                "xg_available": _xg_available(code),
                "xg_seasons": (
                    len(list((Path("data/xg") / code).glob("*.csv")))
                    if (Path("data/xg") / code).exists()
                    else 0
                ),
            }
        )
    return out


@app.get("/api/dashboard")
def dashboard(
    league: str = Query("EPL", pattern="^(EPL|LALIGA|SERIEA)$"),
):
    cfg = _cfg()
    try:
        df = _load_cached(league, cfg.data.data_dir)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))

    df = df.sort_values("date")
    latest = df["date"].max()

    # ---- Standings (current in-progress season, full roster) ----
    season_start = _standings_window(df)
    season_df = df[df["date"] >= season_start]

    standings = _standings(season_df, teams=_current_season_teams(league))
    standings_season = f"{season_start.year}/{str(season_start.year + 1)[-2:]}"

    # ---- Form (last 5) for the top teams ----
    top_teams = [s["team"] for s in standings[:8]]
    form = _form(df, top_teams)

    # ---- Recent results ----
    recent = df.tail(15)[
        ["date", "home_team", "home_goals", "away_goals", "away_team", "result"]
    ].to_dict("records")
    for r in recent:
        r["date"] = r["date"].isoformat()

    # ---- Goals over time ----
    monthly = (
        df.groupby(df["date"].dt.to_period("M"))
        .agg(
            total_goals=(
                "home_goals",
                lambda s: s.sum() + df.loc[s.index, "away_goals"].sum(),
            ),
            matches=("home_goals", "size"),
        )
        .reset_index()
    )
    monthly["month"] = monthly["date"].astype(str)
    goals_per_game = (
        (monthly["total_goals"] / monthly["matches"].clip(lower=1)).round(2).tolist()
    )

    # ---- Home advantage ----
    home_win_pct = float((df["result"] == "H").mean())
    draw_pct = float((df["result"] == "D").mean())
    away_win_pct = float((df["result"] == "A").mean())

    # ---- League totals ----
    league_avg = float((df["home_goals"].sum() + df["away_goals"].sum()) / len(df))
    avg_goals_home = float(df["home_goals"].mean())
    avg_goals_away = float(df["away_goals"].mean())

    return {
        "league": league,
        "league_meta": _league_meta(league),
        "n_matches": int(len(df)),
        "first_season": str(df["date"].min().date()),
        "latest_date": str(latest.date()),
        "standings_season": standings_season,
        "standings": standings,
        "form": form,
        "recent": recent,
        "goals_per_game_over_time": {
            "months": monthly["month"].tolist(),
            "goals_per_game": goals_per_game,
        },
        "outcome_distribution": {
            "home_win": round(home_win_pct, 4),
            "draw": round(draw_pct, 4),
            "away_win": round(away_win_pct, 4),
        },
        "averages": {
            "total_goals_per_match": round(league_avg, 2),
            "home_goals_per_match": round(avg_goals_home, 2),
            "away_goals_per_match": round(avg_goals_away, 2),
        },
        "xg_available": _xg_available(league),
    }


def _season_window_start(latest: pd.Timestamp) -> pd.Timestamp:
    """July 1 of the European season containing ``latest`` (seasons run Aug–May).

    Using ``latest - 1 year`` mixes the tail of the previous season into the
    current league table whenever the new season has already started (e.g. La
    Liga in mid-August), so standings were showing a stale/mixed table.
    """
    year = latest.year if latest.month >= 7 else latest.year - 1
    return pd.Timestamp(year=year, month=7, day=1)


def _current_season_start() -> pd.Timestamp:
    """July 1 of the *current* season (anchored to today, not to the latest
    match in the data) — so during the June-July offseason the standings show
    the brand-new season's table instead of the just-finished one."""
    return _season_window_start(pd.Timestamp.now())


def _standings_window(df: pd.DataFrame) -> pd.Timestamp:
    """Start of the season whose standings the dashboard should show.

    Prefers the *current* (today-anchored) season once it has started — even
    before any of its matches are in the data (Serie A in August shows its
    empty 2026-27 table pre-seeded from the roster). Older data that never
    reaches the current season (e.g. synthetic or historical-only sets) shows
    its own latest season instead of an empty table.
    """
    today_start = _current_season_start()
    prev_start = _season_window_start(today_start - pd.Timedelta(days=1))
    latest_start = _season_window_start(df["date"].max())
    if latest_start in (today_start, prev_start):
        return today_start
    return latest_start


def _empty_standing(team: str) -> dict[str, Any]:
    return {
        "team": team,
        "played": 0,
        "won": 0,
        "draw": 0,
        "lost": 0,
        "gf": 0,
        "ga": 0,
        "points": 0,
    }


def _standings(season_df: pd.DataFrame, teams: Optional[list[str]] = None) -> list:
    """Compute a league table (points, GD) from a season's matches.

    ``teams`` optionally pre-seeds clubs that haven't played yet (0 matches) so
    a just-started season shows its full roster instead of only the pairs that
    have kicked off. Ranks use competition ranking — clubs level on
    (points, goal_diff, goals_for) share a rank.
    """
    table: dict[str, dict[str, Any]] = {}
    for t in teams or []:
        table.setdefault(t, _empty_standing(t))

    for _, r in season_df.iterrows():
        for side in ("home", "away"):
            t = r[f"{side}_team"]
            g = table.setdefault(t, _empty_standing(t))
            g["played"] += 1
            g["gf"] += r[f"{side}_goals"]

        # points from the match
        h = table[r["home_team"]]
        a = table[r["away_team"]]
        if r["result"] == "H":
            h["won"] += 1
            a["lost"] += 1
            h["points"] += 3
        elif r["result"] == "D":
            h["draw"] += 1
            a["draw"] += 1
            h["points"] += 1
            a["points"] += 1
        else:
            a["won"] += 1
            h["lost"] += 1
            a["points"] += 3

        h["ga"] += r["away_goals"]
        a["ga"] += r["home_goals"]

    rows = []
    for standing in table.values():
        rows.append(
            {
                **standing,
                "goal_diff": standing["gf"] - standing["ga"],
                "points": standing["points"],
            }
        )
    rows.sort(key=lambda x: (-x["points"], -x["goal_diff"], -x["gf"]))

    # Competition ranking: equal (points, GD, GF) share a rank.
    prev_key, prev_rank = None, 0
    for i, r in enumerate(rows, 1):
        key = (r["points"], r["goal_diff"], r["gf"])
        if key != prev_key:
            prev_key, prev_rank = key, i
        r["rank"] = prev_rank
    return rows


def _form(df: pd.DataFrame, teams: list, n=5) -> dict:
    """Recent form (W/D/L) for each team, most recent last."""
    out = {}
    for team in teams:
        mask = (df["home_team"] == team) | (df["away_team"] == team)
        sub = df[mask].tail(n)
        form_str = ""
        for _, r in sub.iterrows():
            if (r["home_team"] == team and r["result"] == "H") or (
                r["away_team"] == team and r["result"] == "A"
            ):
                form_str += "W"
            elif r["result"] == "D":
                form_str += "D"
            else:
                form_str += "L"
        out[team] = form_str
    return out


@app.get("/api/teams")
def teams(
    league: str = Query("EPL", pattern="^(EPL|LALIGA|SERIEA)$"),
):
    cfg = _cfg()
    try:
        df = _load_cached(league, cfg.data.data_dir)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))

    model = _get_model_cached(league)

    # Elo ratings snapshot
    elo = model.elo
    teams_elo = sorted(
        [{"team": t, "rating": round(r, 1)} for t, r in elo.ratings.items()],
        key=lambda x: -x["rating"],
    )

    # Strength radar-style data (attack/defense from Dixon-Coles)
    dc = model.dc
    attack = {t: round(float(v), 3) for t, v in dc.attack.items()}
    defense = {t: round(float(v), 3) for t, v in dc.defense.items()}

    # Dynamic state-space ratings (Kalman-filtered attack/defense). The state
    # vector is [alpha_0..alpha_{n-1}, delta_0..delta_{n-1}].
    dyn_attack: dict[str, float] = {}
    dyn_defense: dict[str, float] = {}
    dyn = model.dyn
    if dyn.x is not None and dyn.n:
        n = dyn.n
        for i, t in enumerate(dyn.teams):
            dyn_attack[t] = round(float(dyn.x[i]), 3)
            dyn_defense[t] = round(float(dyn.x[n + i]), 3)

    # Form
    all_teams = sorted(set(df["home_team"].unique()) | set(df["away_team"].unique()))
    form = _form(df, all_teams)

    # Standings (current in-progress season, full roster)
    season_df = df[df["date"] >= _standings_window(df)]
    standings = _standings(season_df, teams=_current_season_teams(league))

    xg_available = _xg_available(league)

    # Roster clubs may not have played yet (promoted clubs have no rows), so
    # union the explicit current-season roster into the selectable team list.
    roster = set(_current_season_teams(league))
    team_names = sorted(set(all_teams) | roster)

    return {
        "league": league,
        "xg_available": xg_available,
        "teams": [
            {
                "team": t,
                "elo_rating": next(
                    (x["rating"] for x in teams_elo if x["team"] == t), None
                ),
                "attack": attack.get(t),
                "defense": defense.get(t),
                "dyn_attack": dyn_attack.get(t),
                "dyn_defense": dyn_defense.get(t),
                "form": form.get(t, ""),
            }
            for t in team_names
        ],
        "standings": standings,
    }


@app.get("/api/predict")
def predict(
    home: str = Query(...),
    away: str = Query(...),
    league: str = Query("EPL", pattern="^(EPL|LALIGA|SERIEA)$"),
    odds_home: Optional[float] = Query(None, ge=1.0),
    odds_draw: Optional[float] = Query(None, ge=1.0),
    odds_away: Optional[float] = Query(None, ge=1.0),
):
    cfg = _cfg()
    try:
        df = _load_cached(league, cfg.data.data_dir)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Attach cached Understat xG when present so the dynamic/xG components show.
    xg_df = _load_xg_cached(league)
    if xg_df is not None and not xg_df.empty:
        df = join_xg(df, xg_df)

    # Optional user-supplied closing line. When provided it feeds the
    # meta-learner's market features directly (residual-vs-market inference)
    # and is what the Market vs Model panel compares against. The isinstance
    # guard keeps direct calls (tests) robust when the Query default object
    # leaks through instead of None.
    def _valid_odds(v) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 1.0

    user_odds = (
        (odds_home, odds_draw, odds_away)
        if all(_valid_odds(v) for v in (odds_home, odds_draw, odds_away))
        else None
    )

    model = _get_model_cached(league)
    try:
        p = model.predict(home, away, market_odds=user_odds)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    model_probs = [p["home_win"], p["draw"], p["away_win"]]

    # Market line for the panel: user-supplied odds take precedence; otherwise
    # fall back to the last meeting's closing line from the data.
    market = None
    if user_odds is not None:
        # _valid_odds guaranteed all three are finite numbers >= 1; the cast
        # is purely for mypy (the tuple elements are still Optional[float]).
        odds = cast(tuple[float, float, float], user_odds)
        inv = [1.0 / o for o in odds]
        tot = sum(inv)
        implied = [v / tot for v in inv]
        market = {
            "implied_home": round(implied[0], 4),
            "implied_draw": round(implied[1], 4),
            "implied_away": round(implied[2], 4),
            "edge_home": round(model_probs[0] - implied[0], 4),
            "edge_draw": round(model_probs[1] - implied[1], 4),
            "edge_away": round(model_probs[2] - implied[2], 4),
            "source": "Your odds",
            "odds": list(odds),
        }
    else:
        odds_row = df[
            (df["home_team"] == home)
            & (df["away_team"] == away)
            & df["implied_home"].notna()
        ].tail(1)
        if not odds_row.empty:
            odds_row = odds_row.iloc[0]
            implied = [
                float(odds_row["implied_home"]),
                float(odds_row["implied_draw"]),
                float(odds_row["implied_away"]),
            ]
            market = {
                "implied_home": round(implied[0], 4),
                "implied_draw": round(implied[1], 4),
                "implied_away": round(implied[2], 4),
                "edge_home": round(model_probs[0] - implied[0], 4),
                "edge_draw": round(model_probs[1] - implied[1], 4),
                "edge_away": round(model_probs[2] - implied[2], 4),
                "source": "Last meeting (closing odds)",
            }

    # Dynamic expected goals from the Kalman-filtered state-space model
    dyn_expected = None
    try:
        dyn_expected = model.dyn.expected_goals(home, away)
    except (ValueError, AttributeError):
        pass

    # Score probability matrix for a chart
    mat = model.dc.score_matrix(home, away, max_goals=6)
    score_grid = {
        "home_goals": list(range(7)),
        "away_goals": list(range(7)),
        "probs": mat[:7, :7].round(4).tolist(),
    }

    comps = p["component_probs"]
    return {
        "home": home,
        "away": away,
        "league": league,
        "probabilities": {
            "home_win": round(float(p["home_win"]), 4),
            "draw": round(float(p["draw"]), 4),
            "away_win": round(float(p["away_win"]), 4),
        },
        "predicted_result": p["predicted_result"],
        "most_likely_score": f"{p['most_likely_score'][0]}-{p['most_likely_score'][1]}",
        "expected_goals": {
            "home": round(float(p["expected_goals"][0]), 2),
            "away": round(float(p["expected_goals"][1]), 2),
        },
        "dyn_expected_goals": (
            {
                "home": round(float(dyn_expected[0]), 2),
                "away": round(float(dyn_expected[1]), 2),
            }
            if dyn_expected
            else None
        ),
        "over_2_5_goals": round(float(p["over_2_5_goals"]), 4),
        "btts_yes": round(float(p["btts_yes"]), 4),
        "component_probs": {
            "dixon_coles": [round(x, 4) for x in comps["dixon_coles"]],
            "elo": [round(x, 4) for x in comps["elo"]],
            "dynamic": [round(x, 4) for x in comps["dynamic"]],
            **({"xg": [round(x, 4) for x in comps["xg"]]} if "xg" in comps else {}),
        },
        "xg_enabled": "xg" in comps,
        "market": market,
        "score_matrix": score_grid,
    }


class FixtureIn(BaseModel):
    """Body for POST /api/fixtures — one new fixture."""

    home: str
    away: str
    date: str  # YYYY-MM-DD (or DD/MM/YYYY)
    matchweek: Optional[int] = None


def _fixture_rows(league: str) -> list[dict]:
    """Upcoming fixtures for a league plus the team list used to build them."""
    df = load_fixtures(league=league)
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "date": r["date"].date().isoformat(),
                "home": r["home_team"],
                "away": r["away_team"],
                "matchweek": int(r["matchweek"]) if pd.notna(r["matchweek"]) else None,
                "season": r["season"],
                "source": r["source"],
            }
        )
    return rows


_ROSTER_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "current_season_teams.json"
)


@functools.lru_cache(maxsize=1)
def _load_rosters() -> dict[str, list[str]]:
    """Explicit current-season rosters (football-data.co.uk spellings).

    Optional override in ``data/current_season_teams.json`` — the authoritative
    ˝who plays this season˝ list, which a data-driven derivation cannot know
    before every club has kicked off (promoted clubs, relegated clubs).
    """
    try:
        return json.loads(_ROSTER_PATH.read_text())
    except (OSError, ValueError):
        return {}


def _current_season_teams(league: str) -> list[str]:
    """Team list for the league's *current* season — the clubs the new
    season's fixtures involve — not the union across all seasons, which
    includes relegated clubs.

    Prefers the explicit roster in ``data/current_season_teams.json`` (the real
    promoted/relegated clubs). Without one, falls back to a data-driven list
    from matches since July 1 of the *current* season — and, while that window
    is too sparse to be representative (fewer than a full matchweek's worth of
    matches), the previous season's clubs as well.
    """
    explicit = _load_rosters().get(league)
    if explicit:
        return sorted(set(explicit))

    cfg = _cfg()
    df = _load_cached(league, cfg.data.data_dir).sort_values("date")
    start = _standings_window(df)
    season_df = df[df["date"] >= start]
    teams = set(season_df["home_team"].tolist()) | set(season_df["away_team"].tolist())
    if len(season_df) < 15:  # < 1 full matchweek played -> most clubs unseen yet
        prev_start = _season_window_start(start - pd.Timedelta(days=1))
        prev = df[df["date"] >= prev_start]
        teams |= set(prev["home_team"].tolist()) | set(prev["away_team"].tolist())
    return sorted(teams)


@app.get("/api/fixtures")
def fixtures(league: str = Query("EPL", pattern="^(EPL|LALIGA|SERIEA)$")):
    """Who is playing next in this league — upcoming fixtures plus the team list.

    The fixture rows are editable via POST/DELETE /api/fixtures. Team names
    come from the league's latest season so the add-form only offers teams the
    model actually knows (unknown/promoted teams get a league-mean prior).
    """
    teams = _current_season_teams(league)
    known = set(teams)
    rows = _fixture_rows(league)
    for r in rows:
        r["known"] = r["home"] in known and r["away"] in known

    seasons = sorted({r["season"] for r in rows})
    return {
        "league": league,
        "league_meta": _league_meta(league),
        "has_fixtures": bool(rows),
        "fixtures": rows,
        "teams": teams,
        "seasons": seasons,
        "n_fixtures": len(rows),
        "xg_available": _xg_available(league),
    }


@app.post("/api/fixtures")
def fixtures_add(
    body: FixtureIn, league: str = Query("EPL", pattern="^(EPL|LALIGA|SERIEA)$")
):
    """Add one upcoming fixture to the league's fixtures folder."""
    try:
        if pd.Timestamp(body.date).normalize() < pd.Timestamp.now().normalize():
            raise ValueError("Fixture date is in the past — pick an upcoming date.")
        add_fixture(
            league,
            body.home,
            body.away,
            body.date,
            matchweek=body.matchweek,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "fixtures": _fixture_rows(league)}


@app.delete("/api/fixtures")
def fixtures_delete(
    league: str = Query("EPL", pattern="^(EPL|LALIGA|SERIEA)$"),
    home: str = Query(...),
    away: str = Query(...),
    date: str = Query(...),
):
    """Remove a fixture. Returns the remaining fixture list."""
    deleted = delete_fixture(league, home, away, date)
    if not deleted:
        raise HTTPException(status_code=404, detail="Fixture not found")
    return {"ok": True, "fixtures": _fixture_rows(league)}


@app.post("/api/fixtures/generate-round-robin")
def fixtures_round_robin(
    league: str = Query("EPL", pattern="^(EPL|LALIGA|SERIEA)$"),
):
    """Build a full double round-robin *placeholder* schedule for the league's
    current team list (every pair twice, home and away) and save it to
    data/fixtures. Clearly a stand-in until the official fixtures are released."""
    teams = _current_season_teams(league)
    schedule = generate_round_robin(teams)
    n_added = write_fixtures_batch(league, schedule)
    return {
        "ok": True,
        "n_added": n_added,
        "n_teams": len(teams),
        "n_matchweeks": int(schedule["matchweek"].max()),
        "fixtures": _fixture_rows(league),
    }


@app.get("/api/backtest")
def backtest(
    league: str = Query("EPL", pattern="^(EPL|LALIGA|SERIEA)$"),
    min_train_matches: int = Query(None, ge=100),
    step_matches: int = Query(None, ge=50),
):
    try:
        _, payload = _get_backtest_cached(league, min_train_matches, step_matches)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    return payload


@app.get("/api/calibrate")
def calibrate(
    league: str = Query("EPL", pattern="^(EPL|LALIGA|SERIEA)$"),
):
    """Re-calibrate the Elo draw width on this league's actual data and return the
    suggested value (and confidence interval) without permanently mutating config."""
    cfg = _cfg()
    try:
        df = _load_cached(league, cfg.data.data_dir)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))

    df = df.sort_values("date")
    draws = (df["result"] == "D").astype(float)

    # Elo rating gap at each match (walk-forward style using only past info)
    elo = EloEngine(
        k=cfg.model.elo_k,
        home_advantage=cfg.model.elo_home_advantage,
        draw_width=cfg.model.elo_draw_width,
    )
    gaps_list: list[float] = []
    for _, r in df.iterrows():
        rh = elo.get(r["home_team"]) + elo.home_advantage
        ra = elo.get(r["away_team"])
        gaps_list.append(round(rh - ra, 1))
        elo.update(
            r["date"], r["home_team"], r["away_team"], r["home_goals"], r["away_goals"]
        )

    gaps = np.array(gaps_list)

    # Simple search over draw_width to minimize Brier on the draw outcome
    best_width, best_brier = None, np.inf
    grid = np.round(np.arange(0.05, 1.0, 0.02), 2)
    for dw in grid:
        p_draw = np.clip(dw * (1 - np.minimum(np.abs(gaps) / 800.0, 0.9)), 0.12, 0.34)
        brier = float(np.mean((p_draw - draws) ** 2))
        if brier < best_brier:
            best_brier, best_width = brier, float(dw)

    return {
        "league": league,
        "current_draw_width": cfg.model.elo_draw_width,
        "suggested_draw_width": best_width,
        "current_brier": None,
        "suggested_brier": round(best_brier, 5),
        "note": "Set 'elo_draw_width' in config.yaml to the suggested value to improve calibration.",
    }


@app.get("/api/research")
def research(
    league: str = Query("EPL", pattern="^(EPL|LALIGA|SERIEA)$"),
    min_train_matches: int = Query(None, ge=100),
    step_matches: int = Query(None, ge=50),
):
    """Scientific-context data for the UI: market-only control comparison,
    statistical power, the walk-forward calibration report, and the
    pre-registered edge-test verdict."""
    import scripts.market_control as mc
    import scripts.power_analysis as pa

    try:
        result_df, _ = _get_backtest_cached(league, min_train_matches, step_matches)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))

    # --- Market-only control: bet the market, no model opinion ---
    control_rows = []
    try:
        control = mc.run_control(result_df)
        keep = ["label", "n", "total_staked", "roi", "sharpe", "strike"]
        control_rows = control[keep].fillna("—").to_dict("records")
        avg_margin = control["avg_margin"].iloc[0]
        residual_by_line = {
            k: {
                "model": round(llm, 4),
                "line": round(llk, 4),
                "residual": round(llm - llk, 4),
            }
            for k, (llm, llk) in control.attrs["residual_by_line"].items()
        }
    except Exception as e:
        log.warning("research control failed: %s", e)
        avg_margin, residual_by_line = None, {}

    # --- Statistical power on this sample ---
    power = {}
    try:
        power = pa.analyse(result_df, league)
    except Exception as e:
        log.warning("research power failed: %s", e)

    # --- Calibration report (if a tuning run has been saved) ---
    calibration = []
    cal_path = Path("reports") / "calibration_quick.csv"
    if cal_path.exists():
        try:
            cal = pd.read_csv(cal_path)
            cols = [
                c
                for c in [
                    "combo",
                    "log_loss",
                    "residual_log_loss",
                    "edge_corr",
                    "kelly_sharpe",
                    "seconds",
                ]
                if c in cal.columns
            ]
            calibration = (
                cal.sort_values(
                    "residual_log_loss"
                    if "residual_log_loss" in cal.columns
                    else "log_loss"
                )
                .head(6)[cols]
                .fillna("—")
                .to_dict("records")
            )
        except Exception as e:
            log.warning("research calibration read failed: %s", e)

    # --- Pre-registered protocol + holdout verdict ---
    protocol = {
        "primary_threshold": "residual log loss <= -0.005",
        "secondary_thresholds": "edge correlation > +0.02 AND value-bet ROI > 0",
        "doc": "docs/edge_test_preregistration.md",
    }
    holdout_result = None
    hr_path = Path("reports") / "holdout_result.json"
    if hr_path.exists():
        try:
            import json

            holdout_result = json.loads(hr_path.read_text())
        except Exception as e:
            log.warning("research holdout read failed: %s", e)

    return {
        "league": league,
        "n_matches": len(result_df),
        "control": {
            "avg_margin": avg_margin,
            "rows": control_rows,
            "residual_by_line": residual_by_line,
        },
        "power": power,
        "calibration": calibration,
        "protocol": protocol,
        "holdout_result": holdout_result,
    }


@app.get("/api/compare")
def compare():
    """Model vs market (Brier/log-loss) across leagues."""
    cfg = _cfg()
    import importlib

    bt = importlib.import_module("backtest")
    out = {}
    for league in LEAGUE_CODES:
        try:
            df = _load_cached(league, cfg.data.data_dir)
            xg_df = _load_xg_cached(league)
            result_df = bt.walk_forward_backtest(
                df,
                min_train_matches=cfg.backtest.min_train_matches,
                step_matches=cfg.backtest.step_matches,
                cfg=cfg,
                xg_df=xg_df,
            )
            if result_df.empty:
                continue
            metrics = _normalize_metrics(bt.evaluate(result_df, with_odds=True))
            mkt = metrics.get("market") or {}
            out[league] = {
                "log_loss": metrics["log_loss"],
                "brier": metrics["brier"],
                "accuracy": metrics["accuracy"],
                "baseline_log_loss": metrics["baseline_log_loss"],
                "market": mkt,
                "value_bets": mkt.get("value_bets", 0),
                "edge_corr": metrics.get("edge_corr"),
                "staking": metrics.get("staking"),
                "xg_available": _xg_available(league),
            }
        except Exception as e:
            log.warning("compare: %s failed: %s", league, e)
    return out
