"""League Updates orchestrator — the pipeline in Section 2.

    python -m src.update.run --leagues EPL LALIGA SERIEA
    python -m src.update.run --leagues EPL --since 2026-11-01
    python -m src.update.run --dry-run

Each league is processed fetch -> normalise -> validate -> upsert -> merge xG
-> ratings -> refit -> predict, inside a single transaction per league so a
failure rolls back only that league. A single-run lock prevents concurrent runs.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

import pandas as pd

from src.config import load_config
from src.update import config as upconfig
from src.update import db as updb
from src.update import fetch as upfetch
from src.update import normalise as upnormalise
from src.update import predict as uppredict
from src.update import ratings as upratings
from src.update import refit as uprefit
from src.update import report as upreport
from src.update import standings as upstandings
from src.update import store as upstore
from src.update import validate as upvalidate

DATA_RAW = upconfig.PROJECT_ROOT / "data" / "raw"
FIXTURES_CSV = upconfig.PROJECT_ROOT / "data" / "fixtures.csv"
XG_DIR = upconfig.PROJECT_ROOT / "data" / "xg"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _refresh_fixtures(cfg, team_map, conn, snapshots_dir, args, results) -> None:
    """Fetch ``fixtures.csv`` once, normalise, and store the real upcoming
    fixtures (with pre-match odds) so predictions use the real schedule."""
    cache = FIXTURES_CSV
    fixtures_raw = None
    if not args.offline:
        try:
            path = upfetch.fetch_fixtures_csv(
                cfg.sources.fixtures_url, cache.parent, snapshots_dir
            )
            fixtures_raw = pd.read_csv(path, encoding="utf-8-sig")
        except Exception as e:
            results["warnings"].append(f"fixtures source stale ({e})")
    if fixtures_raw is None and cache.exists():
        fixtures_raw = pd.read_csv(cache, encoding="utf-8-sig")
    if fixtures_raw is None:
        results["warnings"].append(
            "no fixtures.csv available (predictions will be empty)"
        )
        results["fixtures"] = {"stored": 0, "unmapped": []}
        return
    fx, unmapped = upnormalise.normalise_fixtures_csv(fixtures_raw, team_map)
    if unmapped:
        results["warnings"].append(
            f"unmapped fixture team(s): {', '.join(sorted(unmapped))}"
        )
    if not args.dry_run and not fx.empty:
        n = upstore.upsert_fixtures(conn, fx)
        conn.commit()
        results["fixtures"] = {"stored": n, "unmapped": sorted(unmapped)}
    else:
        results["fixtures"] = {
            "stored": 0 if fx.empty else len(fx),
            "unmapped": sorted(unmapped),
        }


def _current_roster(league: str) -> list[str]:
    """Current-season 20-team roster (football-data spellings) to pre-seed the
    standings table, so teams that haven't kicked off yet still appear."""
    import json

    path = upconfig.PROJECT_ROOT / "data" / "current_season_teams.json"
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get(league, [])
    except Exception:
        return []


def _merge_xg(conn, league: str, season: str) -> tuple[int, int]:
    """Merge cached Understat xG onto stored matches. Returns (complete, pending)."""
    from src.xg_loader import join_xg, load_league_xg

    try:
        xg_df = load_league_xg(str(XG_DIR), league)
    except Exception:
        return 0, 0
    engine = upstore.load_engine_df(conn, league=league, season=season)
    if engine.empty:
        return 0, 0
    joined = join_xg(engine, xg_df)
    complete = 0
    for _, row in joined.iterrows():
        if pd.notna(row.get("home_xg")) and pd.notna(row.get("away_xg")):
            if upvalidate.validate_xg(row["home_xg"], row["away_xg"]):
                upstore.update_xg(
                    conn, row["match_id"], float(row["home_xg"]), float(row["away_xg"])
                )
                complete += 1
    return complete, len(engine) - complete


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the League Updates pipeline")
    ap.add_argument(
        "--leagues",
        nargs="+",
        default=["EPL", "LALIGA", "SERIEA"],
        help="League keys to update (default: all three).",
    )
    ap.add_argument(
        "--since",
        default=None,
        help="Partial rerun: ingest matches on/after this ISO date.",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="Validate only, write nothing."
    )
    ap.add_argument("--db", default=None, help="SQLite path override.")
    ap.add_argument("--config", default=None, help="league_updates.yaml path override.")
    ap.add_argument(
        "--no-refit",
        action="store_true",
        help="Skip Dixon-Coles refit / walk-forward eval.",
    )
    ap.add_argument(
        "--offline",
        action="store_true",
        help="Use cached data/raw files only (no downloads).",
    )
    args = ap.parse_args(argv)

    cfg = upconfig.load_update_config(args.config)
    team_map = upconfig.load_team_names()
    model_cfg = load_config()

    db_path = upconfig.resolve_project_path(args.db or cfg.database.path)
    snapshots_dir = upconfig.resolve_project_path(cfg.raw_snapshots_dir)
    reports_dir = upconfig.resolve_project_path(cfg.reports_dir)
    season = upfetch.season_label()
    season_code = upfetch.season_short_code()

    results = {
        "started_at": _now(),
        "season": season,
        "dry_run": args.dry_run,
        "leagues": {},
        "halted": [],
        "warnings": [],
    }

    lock = None
    if not args.dry_run:
        lock_path = upconfig.resolve_project_path(cfg.lock.path)
        lock = updb.RunLock(lock_path, cfg.lock.stale_after_seconds)
        try:
            lock.acquire()
        except updb.LockError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2

    conn = updb.connect(db_path)
    updb.init_db(conn)

    try:
        # Real upcoming fixtures from football-data fixtures.csv (one fetch).
        _refresh_fixtures(cfg, team_map, conn, snapshots_dir, args, results)

        for league in args.leagues:
            league_cfg = getattr(cfg.leagues, league, None)
            if league_cfg is None:
                results["warnings"].append(f"Unknown league '{league}' skipped")
                continue
            _run_league(
                league,
                season,
                season_code,
                cfg,
                team_map,
                model_cfg,
                conn,
                snapshots_dir,
                results,
                args,
            )

        results["finished_at"] = _now()
        md = upreport.build_report(results)
        report_path = None
        if args.dry_run:
            print(md)
        else:
            report_path = upreport.write_report(md, reports_dir)
            print(f"Report written to {report_path}")

        conn.execute(
            "INSERT INTO run_log (started_at, finished_at, status, report_path) "
            "VALUES (?, ?, ?, ?)",
            (
                results["started_at"],
                results["finished_at"],
                "halted" if results["halted"] else "ok",
                str(report_path) if report_path else None,
            ),
        )
        conn.commit()
    finally:
        conn.close()
        if lock:
            lock.release()

    return 1 if results["halted"] else 0


def _run_league(
    league,
    season,
    season_code,
    cfg,
    team_map,
    model_cfg,
    conn,
    snapshots_dir,
    results,
    args,
):
    """Run one league through fetch -> normalise -> validate -> upsert -> xG ->
    ratings -> refit -> predict."""
    entry = {
        "status": "ok",
        "added": 0,
        "corrected": 0,
        "unchanged": 0,
        "quarantined": 0,
        "row_count": 0,
        "odds_inserted": 0,
        "latest_match_date": None,
        "xg_complete": 0,
        "xg_pending": 0,
        "walk_forward": {},
        "elo_movers": [],
        "predictions": [],
        "warnings": [],
        "standings": [],
    }
    results["leagues"][league] = entry

    code = getattr(cfg.leagues, league).football_data_code
    m = model_cfg.model
    elo_k = float(getattr(m, "elo_k", 20.0))
    home_adv = float(getattr(m, "elo_home_advantage", 75.0))
    dc_xi = float(getattr(m, "dc_xi", 0.0018))
    dc_shrinkage = float(getattr(m, "dc_shrinkage", 0.0))
    promoted_start = float(getattr(cfg.ratings, "promoted_team_start", 1450.0))
    regression = float(getattr(cfg.ratings, "season_regression", 0.25))

    # 1. Fetch (fall back to a cached file; never write an empty file).
    raw_df = None
    local = DATA_RAW / code / f"{season}.csv"
    if not args.offline:
        try:
            path = upfetch.fetch_league_csv(
                league,
                code,
                season_code,
                cfg.sources.football_data_base,
                DATA_RAW,
                snapshots_dir,
            )
            raw_df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception as e:
            if local.exists():
                raw_df = pd.read_csv(local, encoding="utf-8-sig")
                entry["warnings"].append(f"source stale ({e}); using cached {local}")
            else:
                results["halted"].append(
                    (league, f"source down and no cached file: {e}")
                )
                entry["status"] = "halted"
                return
    else:
        if local.exists():
            raw_df = pd.read_csv(local, encoding="utf-8-sig")
        else:
            results["halted"].append(
                (league, f"no cached file {local} for offline run")
            )
            entry["status"] = "halted"
            return

    # 2. Normalise (unmapped team halts the league).
    matches, odds, unmapped = upnormalise.normalise_season_csv(
        raw_df, league, season, team_map
    )
    if unmapped:
        results["halted"].append((league, f"unmapped team(s): {', '.join(unmapped)}"))
        entry["status"] = "halted"
        entry["warnings"].append("unmapped team(s) — add to config/team_names.yaml")
        return

    if args.since:
        matches = matches[matches["date"] >= pd.Timestamp(args.since)].copy()

    # 3. Validate (quarantine failures, continue with valid rows).
    valid_df, quarantined, missing_rates = upvalidate.validate_matches(matches, cfg)
    valid_odds, bad_odds = upvalidate.validate_odds(
        odds, tuple(cfg.validation.odds_overround_range)
    )
    entry["quarantined"] = len(quarantined) + len(bad_odds)
    if bad_odds:
        entry["warnings"].append(f"{len(bad_odds)} odds snapshot(s) rejected")
    for col, rate in missing_rates.items():
        if (
            col in ("home", "away", "fthg", "ftag", "date")
            and rate > cfg.validation.max_missing_rate
        ):
            entry["warnings"].append(
                f"column '{col}' missing rate {rate:.1%} exceeds threshold"
            )

    # 4. Upsert + quarantine (one transaction per league).
    if not args.dry_run:
        try:
            conn.execute("BEGIN")
            summary = {"added": 0, "corrected": 0, "unchanged": 0, "odds_inserted": 0}
            if len(valid_df):
                summary = upstore.upsert_matches(conn, valid_df, valid_odds)
            upstore.insert_quarantine(conn, quarantined)
            row_count = upstore.check_row_count(conn, league, season)
            conn.commit()
            entry.update(
                {
                    k: summary[k]
                    for k in ("added", "corrected", "unchanged", "odds_inserted")
                }
            )
            entry["row_count"] = row_count
        except upstore.RowCountError as e:
            conn.rollback()
            results["halted"].append((league, str(e)))
            entry["status"] = "halted"
            return
        except Exception as e:
            conn.rollback()
            results["halted"].append((league, f"upsert failed: {e}"))
            entry["status"] = "halted"
            return
    else:
        entry["added"] = len(valid_df)

    # 5. Merge xG.
    if cfg.sources.understat == "enabled" and not args.dry_run:
        entry["xg_complete"], entry["xg_pending"] = _merge_xg(conn, league, season)

    # 6. Ratings (Elo, chronological + season rollover).
    engine_df = upstore.load_engine_df(conn, league)
    engine = upratings.fit_elo(
        engine_df,
        elo_k=elo_k,
        home_advantage=home_adv,
        promoted_start=promoted_start,
        season_regression=regression,
    )
    if not args.dry_run:
        entry["elo_movers"] = upratings.persist_ratings(conn, league, engine)[:10]
    else:
        prev = upratings.load_ratings(conn, league)
        movers = []
        for t, r in engine.ratings.items():
            old = prev.get(t)
            movers.append(
                {
                    "team": t,
                    "old": old,
                    "new": r,
                    "delta": (r - old) if old is not None else 0.0,
                }
            )
        movers.sort(key=lambda x: abs(x["delta"]), reverse=True)
        entry["elo_movers"] = movers[:10]

    # 6b. Standings + form — computed from stored results, not scraped.
    season_df = (
        engine_df[engine_df["season"] == season]
        if "season" in engine_df.columns
        else engine_df
    )
    standings_rows = upstandings.compute_standings(
        season_df, teams=_current_roster(league), form_n=5
    )
    entry["standings"] = standings_rows
    if not args.dry_run:
        upstandings.write_standings(conn, league, season, standings_rows)

    # 7. Refit Dixon-Coles + walk-forward log-loss vs market.
    completed = engine_df.dropna(subset=["home_goals", "away_goals"])
    dc = None
    if not args.no_refit and len(completed) >= 2:
        dc = uprefit.refit_dixon_coles(completed, xi=dc_xi, shrinkage=dc_shrinkage)
        with_odds = upstore.add_odds_columns(conn, engine_df)
        min_train = max(10, min(380, len(completed) // 2))
        entry["walk_forward"] = uprefit.walk_forward_eval(
            with_odds, xi=dc_xi, shrinkage=dc_shrinkage, min_train=min_train
        )
    else:
        entry["walk_forward"] = {}

    # 8. Predict upcoming fixtures.
    if dc is None and len(completed) >= 2:
        dc = uprefit.refit_dixon_coles(completed, xi=dc_xi, shrinkage=dc_shrinkage)
    if dc is None:
        from src.dixon_coles import DixonColes

        dc = DixonColes(xi=dc_xi, shrinkage=dc_shrinkage)
    fixtures = upstore.load_fixtures_for_prediction(conn, league=league, season=season)
    entry["predictions"] = uppredict.generate_predictions(dc, engine, fixtures)

    if len(engine_df):
        entry["latest_match_date"] = str(engine_df["date"].max().date())

    if not args.dry_run:
        conn.commit()


if __name__ == "__main__":
    raise SystemExit(main())
