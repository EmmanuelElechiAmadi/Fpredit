"""
Football Predictor Web App — FastAPI backend.

Serves a modern single-page UI on top of the existing prediction pipeline:
  GET  /api/leagues                    — list available leagues + metadata
  GET  /api/dashboard?league=EPL       — standings, form, plots data
  GET  /api/teams?league=EPL           — team intelligence (ratings, form, fixtures)
  GET  /api/predict                    — H/D/A probs + expected goals + market
  GET  /api/backtest?league=EPL        — walk-forward metrics vs baselines
  GET  /api/calibrate?league=EPL       — calibrate Elo draw width
  GET  /api/compare                   — model vs market across leagues
  GET  /api/health                     — health check

Run:  uvicorn app.main:app --reload
"""

import functools
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.config import load_config
from src.data_loader import LEAGUE_CODES, load_league_csvs
from src.elo import EloEngine
from src.market import add_implied_probabilities
from src.model_factory import build_ensemble

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("webapp")

CACHE_TTL = 60 * 5  # 5 minutes for data-derived endpoint results

app = FastAPI(title="Football Predictor API", version="1.0.0")

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
    return out


@functools.lru_cache(maxsize=16)
def _load_cached(league: str, raw_dir: str) -> pd.DataFrame:
    df = load_league_csvs(raw_dir, league)
    df = add_implied_probabilities(df)
    return df


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
            }
        )
    return out


@app.get("/api/dashboard")
def dashboard(
    league: str = Query("EPL", regex="^(EPL|LALIGA|SERIEA)$"),
):
    cfg = _cfg()
    try:
        df = _load_cached(league, cfg.data.data_dir)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))

    df = df.sort_values("date")
    latest = df["date"].max()

    # ---- Standings (latest season) ----
    last_season = df[df["date"] <= latest]
    first_season_date = df.iloc[-1]["date"] - pd.DateOffset(years=1)
    season_df = last_season[last_season["date"] >= first_season_date]

    standings = _standings(season_df)

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
        .agg(total_goals=("home_goals", lambda s: s.sum() + df.loc[s.index, "away_goals"].sum()), matches=("home_goals", "size"))
        .reset_index()
    )
    monthly["month"] = monthly["date"].astype(str)
    goals_per_game = (monthly["total_goals"] / monthly["matches"].clip(lower=1)).round(2).tolist()

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
    }


def _standings(season_df: pd.DataFrame) -> list:
    """Compute a simple league table (points, GD) from a season's matches."""
    teams = {}
    for _, r in season_df.iterrows():
        for side in ("home", "away"):
            t = r[f"{side}_team"]
            g = teams.setdefault(
                t,
                {"team": t, "played": 0, "won": 0, "draw": 0, "lost": 0,
                 "gf": 0, "ga": 0, "points": 0},
            )
            g["played"] += 1
            g["gf"] += r[f"{side}_goals"]
            g["ga"] += r[f"{side}_goals"] if side == "away" else 0

        # points from the match
        h = teams[r["home_team"]]
        a = teams[r["away_team"]]
        h_goal_diff = r["home_goals"] - r["away_goals"]
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
    for t in teams.values():
        rows.append(
            {
                **t,
                "goal_diff": t["gf"] - t["ga"],
                "points": t["points"],
            }
        )
    rows.sort(key=lambda x: (-x["points"], -x["goal_diff"], -x["gf"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
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
    league: str = Query("EPL", regex="^(EPL|LALIGA|SERIEA)$"),
):
    cfg = _cfg()
    try:
        df = _load_cached(league, cfg.data.data_dir)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))

    model = build_ensemble(cfg)
    model.fit(df.sort_values("date"))

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

    # Form
    all_teams = sorted(set(df["home_team"].unique()) | set(df["away_team"].unique()))
    form = _form(df, all_teams)

    # Standings
    latest = df["date"].max()
    first_season_date = latest - pd.DateOffset(years=1)
    season_df = df[df["date"] >= first_season_date]
    standings = _standings(season_df)

    return {
        "league": league,
        "teams": [
            {
                "team": t,
                "elo_rating": next(
                    (x["rating"] for x in teams_elo if x["team"] == t), None
                ),
                "attack": attack.get(t),
                "defense": defense.get(t),
                "form": form.get(t, ""),
            }
            for t in sorted(all_teams)
        ],
        "standings": standings,
    }


@app.get("/api/predict")
def predict(
    home: str = Query(...),
    away: str = Query(...),
    league: str = Query("EPL", regex="^(EPL|LALIGA|SERIEA)$"),
):
    cfg = _cfg()
    try:
        df = _load_cached(league, cfg.data.data_dir)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))

    model = build_ensemble(cfg)
    model.fit(df.sort_values("date"))
    try:
        p = model.predict(home, away)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Market odds for this fixture if available
    market = None
    odds_row = df[
        (df["home_team"] == home)
        & (df["away_team"] == away)
        & df["implied_home"].notna()
    ].tail(1)
    if not odds_row.empty:
        market = {
            "implied_home": round(float(odds_row.iloc[0]["implied_home"]), 4),
            "implied_draw": round(float(odds_row.iloc[0]["implied_draw"]), 4),
            "implied_away": round(float(odds_row.iloc[0]["implied_away"]), 4),
        }

    # Score probability matrix for a chart
    mat = model.dc.score_matrix(home, away, max_goals=6)
    score_grid = {
        "home_goals": list(range(7)),
        "away_goals": list(range(7)),
        "probs": mat[:7, :7].round(4).tolist(),
    }

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
        "over_2_5_goals": round(float(p["over_2_5_goals"]), 4),
        "btts_yes": round(float(p["btts_yes"]), 4),
        "component_probs": {
            "dixon_coles": [round(x, 4) for x in p["component_probs"]["dixon_coles"]],
            "elo": [round(x, 4) for x in p["component_probs"]["elo"]],
        },
        "market": market,
        "score_matrix": score_grid,
    }


@app.get("/api/backtest")
def backtest(
    league: str = Query("EPL", regex="^(EPL|LALIGA|SERIEA)$"),
    min_train_matches: int = Query(None, ge=100),
    step_matches: int = Query(None, ge=50),
):
    import importlib

    bt = importlib.import_module("backtest")
    cfg = _cfg()
    try:
        df = _load_cached(league, cfg.data.data_dir)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))

    min_train = min_train_matches or cfg.backtest.min_train_matches
    step = step_matches or cfg.backtest.step_matches

    result_df = bt.walk_forward_backtest(
        df, min_train_matches=min_train, step_matches=step, cfg=cfg
    )
    if result_df.empty:
        raise HTTPException(status_code=422, detail="Backtest produced no predictions.")

    metrics = _normalize_metrics(bt.evaluate(result_df, with_odds=True))

    # Monthly accuracy curve
    result_df = result_df.sort_values("date")
    result_df["month"] = result_df["date"].dt.to_period("M").astype(str)
    monthly_acc = (
        result_df.groupby("month")
        .apply(lambda g: float((g["pred_home_win"].gt(g["pred_draw"]) & g["pred_home_win"].gt(g["pred_away_win"])).mean()))
        .round(4)
        .to_dict()
    )

    return {"metrics": metrics, "n_matches": len(result_df), "monthly_accuracy": monthly_acc}


@app.get("/api/calibrate")
def calibrate(
    league: str = Query("EPL", regex="^(EPL|LALIGA|SERIEA)$"),
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
    gaps = []
    for _, r in df.iterrows():
        rh = elo.get(r["home_team"]) + elo.home_advantage
        ra = elo.get(r["away_team"])
        gaps.append(round(rh - ra, 1))
        elo.update(r["date"], r["home_team"], r["away_team"], r["home_goals"], r["away_goals"])

    gaps = np.array(gaps)

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
            result_df = bt.walk_forward_backtest(
                df,
                min_train_matches=cfg.backtest.min_train_matches,
                step_matches=cfg.backtest.step_matches,
                cfg=cfg,
            )
            if result_df.empty:
                continue
            metrics = _normalize_metrics(bt.evaluate(result_df, with_odds=True))
            out[league] = {
                "log_loss": metrics["log_loss"],
                "brier": metrics["brier"],
                "accuracy": metrics["accuracy"],
                "baseline_log_loss": metrics["baseline_log_loss"],
                "market": metrics.get("market"),
                "value_bets": metrics.get("market", {}).get("value_bets", 0),
            }
        except Exception as e:
            log.warning("compare: %s failed: %s", league, e)
    return out