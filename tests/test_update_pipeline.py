"""End-to-end test for the League Updates pipeline (offline, temp DB)."""

import sqlite3

import src.update.fetch as upfetch
from src.update import run as update_run

RAW_CSV = """Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,HS,AS,HST,AST,B365H,B365D,B365A
E0,14/08/2026,20:00,Arsenal,Coventry,2,1,H,1,0,H,12,8,5,3,1.60,3.90,5.50
E0,21/08/2026,15:00,Coventry,Arsenal,0,3,A,0,1,A,6,15,2,7,2.80,3.40,2.40
"""

FIXTURES_CSV = """Div,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A,AvgH,AvgD,AvgA,MaxH,MaxD,MaxA
E0,21/11/2026,15:00,Chelsea,Leeds,1.80,3.60,4.50,1.75,3.50,4.60,1.90,3.70,4.40
E0,21/11/2026,17:30,Newcastle,Arsenal,3.10,3.40,2.20,3.00,3.30,2.30,3.20,3.50,2.10
"""


def _write_config(tmp_path):
    cfg = tmp_path / "league_updates.yaml"
    cfg.write_text(
        f"""leagues:
  EPL: {{ football_data_code: E0, understat: EPL }}
sources:
  understat: disabled
database:
  path: {tmp_path}/updates.db
raw_snapshots_dir: {tmp_path}/snapshots
reports_dir: {tmp_path}/reports
lock:
  path: {tmp_path}/updates.lock
  stale_after_seconds: 5
"""
    )
    return cfg


def test_end_to_end_and_idempotent(tmp_path, monkeypatch):
    season = upfetch.season_label()
    raw_dir = tmp_path / "data" / "raw" / "E0"
    raw_dir.mkdir(parents=True)
    (raw_dir / f"{season}.csv").write_text(RAW_CSV)

    monkeypatch.setattr(update_run, "DATA_RAW", tmp_path / "data" / "raw")
    monkeypatch.setattr(update_run, "XG_DIR", tmp_path / "data" / "xg")
    fixtures_csv = tmp_path / "fixtures.csv"
    fixtures_csv.write_text(FIXTURES_CSV)
    monkeypatch.setattr(update_run, "FIXTURES_CSV", fixtures_csv)

    cfg_path = _write_config(tmp_path)
    argv = [
        "--offline",
        "--config",
        str(cfg_path),
        "--db",
        str(tmp_path / "updates.db"),
        "--leagues",
        "EPL",
        "--no-refit",
    ]

    assert update_run.main(argv) == 0

    conn = sqlite3.connect(str(tmp_path / "updates.db"))
    conn.row_factory = sqlite3.Row
    assert conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM odds").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM team_ratings").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM quarantine").fetchone()[0] == 0
    # Real fixtures from fixtures.csv are stored (with pre-match odds).
    assert (
        conn.execute("SELECT COUNT(*) FROM fixtures WHERE league = 'EPL'").fetchone()[0]
        == 2
    )

    # Standings are computed from the stored results (not scraped), with the
    # full 20-team roster pre-seeded so teams with no matches still appear.
    standings = conn.execute(
        "SELECT team, points, goal_diff, played FROM standings ORDER BY rank, team"
    ).fetchall()
    by_team = {r["team"]: r for r in standings}
    assert len(standings) == 20
    assert by_team["Arsenal"]["points"] == 6
    assert by_team["Arsenal"]["played"] == 2
    assert by_team["Coventry"]["goal_diff"] == -4
    assert by_team["Coventry"]["played"] == 2
    assert by_team["Man City"]["played"] == 0  # pre-seeded, not played yet

    # Idempotent re-run: nothing added, nothing duplicated.
    assert update_run.main(argv) == 0
    assert conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM odds").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM run_log").fetchone()[0] == 2


def test_unmapped_team_halts_league(tmp_path, monkeypatch):
    season = upfetch.season_label()
    raw_dir = tmp_path / "data" / "raw" / "E0"
    raw_dir.mkdir(parents=True)
    (raw_dir / f"{season}.csv").write_text(
        """Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR
E0,14/08/2026,20:00,Mystery FC,Coventry,2,1,H
"""
    )
    monkeypatch.setattr(update_run, "DATA_RAW", tmp_path / "data" / "raw")
    monkeypatch.setattr(update_run, "XG_DIR", tmp_path / "data" / "xg")
    monkeypatch.setattr(update_run, "FIXTURES_CSV", tmp_path / "missing_fixtures.csv")

    cfg_path = _write_config(tmp_path)
    rc = update_run.main(
        [
            "--offline",
            "--config",
            str(cfg_path),
            "--db",
            str(tmp_path / "updates.db"),
            "--leagues",
            "EPL",
            "--no-refit",
        ]
    )
    assert rc == 1  # halted
    conn = sqlite3.connect(str(tmp_path / "updates.db"))
    assert conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 0
