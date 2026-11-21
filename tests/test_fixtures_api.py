"""API-level tests for the fixtures endpoints (GET/POST/DELETE/generate-round-robin).

Hermetic: the model-fitting warm-up is disabled and the league data + fixture
file I/O are redirected to synthetic/in-memory paths so no real data or model
fits are touched.
"""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app.main as main
from src import fixtures as fmod
from src.data_loader import generate_synthetic_league
from src.market import add_implied_probabilities


@pytest.fixture
def client(monkeypatch, tmp_path):
    league_df = generate_synthetic_league(n_teams=6, n_seasons=1)
    monkeypatch.setattr(main, "_warm_caches", lambda: None)  # skip model warm-up
    monkeypatch.setattr(main, "_load_rosters", lambda: {})  # no explicit roster
    # Pin the wall clock so the add-fixture "no past dates" rule is stable on
    # any real run date: 2026-08-01 is then "past", 2026-08-29 is "upcoming".
    monkeypatch.setattr(main, "_now", lambda: pd.Timestamp("2026-08-15"))
    monkeypatch.setattr(main, "_load_cached", lambda league, raw_dir: league_df)
    monkeypatch.setattr(
        main,
        "_cfg",
        lambda: SimpleNamespace(data=SimpleNamespace(data_dir=str(tmp_path))),
    )
    # Redirect fixture file I/O into the tmp dir so tests never write real fixtures.
    monkeypatch.setattr(
        main,
        "load_fixtures",
        lambda league="EPL": fmod.load_fixtures(
            str(tmp_path), league, today=pd.Timestamp("2026-08-01")
        ),
    )
    monkeypatch.setattr(
        main,
        "add_fixture",
        lambda league, home, away, date, matchweek=None: fmod.add_fixture(
            league, home, away, date, matchweek=matchweek, raw_dir=str(tmp_path)
        ),
    )
    monkeypatch.setattr(
        main,
        "delete_fixture",
        lambda league, home, away, date: fmod.delete_fixture(
            league, home, away, date, raw_dir=str(tmp_path)
        ),
    )
    monkeypatch.setattr(
        main,
        "write_fixtures_batch",
        lambda league, schedule: fmod.write_fixtures_batch(
            league, schedule, raw_dir=str(tmp_path)
        ),
    )
    with TestClient(main.app) as c:
        yield c


class TestFixturesApi:
    def test_get_returns_teams_and_empty_state(self, client):
        r = client.get("/api/fixtures?league=EPL")
        assert r.status_code == 200
        d = r.json()
        assert d["has_fixtures"] is False
        assert d["n_fixtures"] == 0
        assert "Team A" in d["teams"]  # from synthetic league

    def test_add_then_get_reflects_fixture(self, client):
        r = client.post(
            "/api/fixtures?league=EPL",
            json={
                "home": "Team A",
                "away": "Team B",
                "date": "2026-08-29",
                "matchweek": 1,
            },
        )
        assert r.status_code == 200
        fixtures = r.json()["fixtures"]
        assert len(fixtures) == 1
        assert fixtures[0]["home"] == "Team A"
        assert fixtures[0]["away"] == "Team B"
        assert fixtures[0]["matchweek"] == 1

        r = client.get("/api/fixtures?league=EPL")
        d = r.json()
        assert d["n_fixtures"] == 1
        assert d["seasons"] == ["2026-27"]
        assert d["fixtures"][0]["known"] is True  # both teams in the league data

    def test_add_rejects_past_date(self, client):
        r = client.post(
            "/api/fixtures?league=EPL",
            json={"home": "Team A", "away": "Team B", "date": "2026-08-01"},
        )
        assert r.status_code == 400

    def test_add_rejects_self_match(self, client):
        r = client.post(
            "/api/fixtures?league=EPL",
            json={"home": "Team A", "away": "Team A", "date": "2026-08-29"},
        )
        assert r.status_code == 400

    def test_delete_removes_fixture(self, client):
        client.post(
            "/api/fixtures?league=EPL",
            json={"home": "Team A", "away": "Team B", "date": "2026-08-29"},
        )
        r = client.delete(
            "/api/fixtures?league=EPL&home=Team%20A&away=Team%20B&date=2026-08-29"
        )
        assert r.status_code == 200
        assert r.json()["fixtures"] == []

        r = client.delete(
            "/api/fixtures?league=EPL&home=Team%20A&away=Team%20B&date=2026-08-29"
        )
        assert r.status_code == 404

    def test_round_robin_generates_full_schedule(self, client):
        r = client.post("/api/fixtures/generate-round-robin?league=EPL")
        assert r.status_code == 200
        d = r.json()
        # 6 synthetic teams -> 5 rounds per half, 30 matches, 10 matchweeks
        assert d["n_teams"] == 6
        assert d["n_matchweeks"] == 10
        assert d["n_added"] == 30
        assert len(d["fixtures"]) == 30


class TestSeasonWindow:
    """Dashboard/teams standings must be anchored to July 1 of the European
    season containing the latest match — not ``latest - 1 year``, which mixes
    the previous season's tail into the table once the new season has started."""

    def test_august_latest_anchors_to_current_season(self):
        assert main._season_window_start(pd.Timestamp("2026-08-20")) == pd.Timestamp(
            "2026-07-01"
        )
        assert main._season_window_start(pd.Timestamp("2026-08-15")) == pd.Timestamp(
            "2026-07-01"
        )

    def test_may_latest_anchors_to_finished_season(self):
        # End of the season (May) still belongs to the season that started last July.
        assert main._season_window_start(pd.Timestamp("2027-05-24")) == pd.Timestamp(
            "2026-07-01"
        )

    def test_january_latest_uses_previous_july(self):
        assert main._season_window_start(pd.Timestamp("2026-01-15")) == pd.Timestamp(
            "2025-07-01"
        )


class TestCurrentSeasonTeams:
    """The fixtures team list must not collapse to a couple of clubs when the
    new season has only just kicked off (1-2 matches played)."""

    def _df(self, n_prev, n_cur):
        rows = []
        for i in range(n_prev):
            rows.append(
                dict(
                    date=pd.Timestamp("2025-08-16") + pd.Timedelta(days=7 * i),
                    home_team="Team A" if i % 2 else "Team Z",
                    away_team="Team B",
                    home_goals=1,
                    away_goals=0,
                    result="H",
                )
            )
        for i in range(n_cur):
            rows.append(
                dict(
                    date=pd.Timestamp("2026-08-21") + pd.Timedelta(days=7 * i),
                    home_team="Team A" if i % 2 == 0 else "Team C",
                    away_team="Team B" if i % 2 == 0 else "Team D",
                    home_goals=1,
                    away_goals=1,
                    result="D",
                )
            )
        return pd.DataFrame(rows)

    def test_sparse_new_season_falls_back_to_previous_clubs(self, monkeypatch):
        # Only a couple of matches played in 2026-27 -> still list all clubs
        # (current + previous season), not just the pairs that have kicked off.
        monkeypatch.setattr(main, "_load_rosters", lambda: {})
        df = self._df(n_prev=40, n_cur=2)
        monkeypatch.setattr(main, "_load_cached", lambda league, raw_dir: df)
        monkeypatch.setattr(
            main, "_cfg", lambda: SimpleNamespace(data=SimpleNamespace(data_dir="x"))
        )
        teams = main._current_season_teams("EPL")
        assert set(teams) == {"Team A", "Team B", "Team C", "Team D", "Team Z"}

    def test_mature_season_excludes_relegated_clubs(self, monkeypatch):
        # 20 matches played in the new season -> current clubs only; Team Z (a
        # relegated club present only in the previous season) must be dropped.
        monkeypatch.setattr(main, "_load_rosters", lambda: {})
        df = self._df(n_prev=40, n_cur=20)
        monkeypatch.setattr(main, "_load_cached", lambda league, raw_dir: df)
        monkeypatch.setattr(
            main, "_cfg", lambda: SimpleNamespace(data=SimpleNamespace(data_dir="x"))
        )
        teams = main._current_season_teams("EPL")
        assert set(teams) == {"Team A", "Team B", "Team C", "Team D"}

    def test_explicit_roster_takes_precedence(self, monkeypatch):
        # The shipped data/current_season_teams.json is authoritative even
        # before any club has kicked off (promoted clubs have no match rows).
        monkeypatch.setattr(
            main,
            "_load_rosters",
            lambda: {"EPL": ["Team A", "Team B", "Team C", "Team D", "Team New"]},
        )
        teams = main._current_season_teams("EPL")
        assert teams == ["Team A", "Team B", "Team C", "Team D", "Team New"]


class TestStandings:
    """League-table arithmetic (GF/GA/GD) must be correct."""

    def _standings_df(self):
        return pd.DataFrame(
            [
                dict(
                    date=pd.Timestamp("2026-08-15"),
                    home_team="Team A",
                    away_team="Team B",
                    home_goals=2,
                    away_goals=2,
                    result="D",
                ),
                dict(
                    date=pd.Timestamp("2026-08-16"),
                    home_team="Team A",
                    away_team="Team C",
                    home_goals=3,
                    away_goals=0,
                    result="H",
                ),
            ]
        )

    def test_goals_against_not_double_counted(self):
        """Regression: the away side's own goals were added to its GA, so a
        2-2 draw showed GA 4 for the away team (e.g. Villarreal in the
        2026-27 La Liga opener)."""
        rows = {r["team"]: r for r in main._standings(self._standings_df())}
        assert rows["Team B"]["gf"] == 2 and rows["Team B"]["ga"] == 2
        assert rows["Team A"]["gf"] == 5 and rows["Team A"]["ga"] == 2
        assert rows["Team A"]["goal_diff"] == 3

    def test_points_and_order(self):
        rows = main._standings(self._standings_df())
        order = [r["team"] for r in rows]
        assert order == ["Team A", "Team B", "Team C"]
        by = {r["team"]: r for r in rows}
        assert by["Team A"]["points"] == 4  # win + draw
        assert by["Team B"]["points"] == 1
        assert by["Team C"]["points"] == 0

    def test_preseeds_unplayed_roster_clubs(self):
        # A just-started season must show its full roster, not just the clubs
        # that have played (the dashboard table the user expects).
        df = self._standings_df()
        rows = main._standings(df, teams=["Team A", "Team B", "Team C", "Team D"])
        by = {r["team"]: r for r in rows}
        assert by["Team D"]["played"] == 0 and by["Team D"]["points"] == 0
        assert by["Team D"]["goal_diff"] == 0
        assert len(rows) == 4

    def test_competition_ranking_ties_share_rank(self):
        # Two clubs level on (points, GD, GF) must share a rank, like the real
        # league tables (e.g. the pre-season 0-pt group all ranked together).
        df = pd.DataFrame(
            [
                dict(
                    date=pd.Timestamp("2026-08-15"),
                    home_team="Team A",
                    away_team="Team B",
                    home_goals=1,
                    away_goals=1,
                    result="D",
                ),
                dict(
                    date=pd.Timestamp("2026-08-16"),
                    home_team="Team C",
                    away_team="Team D",
                    home_goals=2,
                    away_goals=2,
                    result="D",
                ),
            ]
        )
        rows = main._standings(df)
        by = {r["team"]: r for r in rows}
        # C/D drew 2-2 (GF 2) so they out-rank A/B (GF 1) on goals scored.
        assert by["Team C"]["rank"] == by["Team D"]["rank"] == 1
        assert by["Team A"]["rank"] == by["Team B"]["rank"] == 3


class FakePredictModel:
    """Minimal stand-in for the ensemble's predict() surface."""

    def __init__(self):
        self.dyn = SimpleNamespace(expected_goals=lambda h, a: (1.2, 0.8))
        self.dc = SimpleNamespace(
            # A proper (flat) score distribution so derivative-market pricing
            # has probability mass to work with (7x7 == the API's grid size).
            score_matrix=lambda h, a, max_goals=6: np.full((7, 7), 1.0 / 49.0)
        )

    def predict(self, home, away, market_odds=None):
        return {
            "home_win": 0.5,
            "draw": 0.3,
            "away_win": 0.2,
            "predicted_result": "H",
            "most_likely_score": (1, 0),
            "expected_goals": (1.2, 0.8),
            "over_2_5_goals": 0.4,
            "btts_yes": 0.5,
            "component_probs": {
                "dixon_coles": [0.5, 0.3, 0.2],
                "elo": [0.5, 0.3, 0.2],
                "dynamic": [0.5, 0.3, 0.2],
            },
        }


class TestPredictOdds:
    """User-supplied closing odds enable the residual-vs-market layer for any
    fixture (the old UI told users to 'pass odds to the API' but gave no way
    to do it from the web app)."""

    @pytest.fixture(autouse=True)
    def _patch(self, monkeypatch):
        df = add_implied_probabilities(
            generate_synthetic_league(n_teams=6, n_seasons=1)
        )
        monkeypatch.setattr(main, "_load_cached", lambda league, raw_dir: df)
        monkeypatch.setattr(main, "_load_xg_cached", lambda league: None)
        monkeypatch.setattr(
            main, "_get_model_cached", lambda league: FakePredictModel()
        )
        monkeypatch.setattr(
            main, "_cfg", lambda: SimpleNamespace(data=SimpleNamespace(data_dir="x"))
        )

    def test_odds_without_market_columns_still_produce_market_panel(self):
        # A fixture with no historical odds columns (e.g. two promoted clubs)
        # previously showed "No closing line recorded". With odds supplied the
        # API must build the implied/edge block from them.
        d = main.predict(
            home="Team A",
            away="Team B",
            league="EPL",
            odds_home=2.0,
            odds_draw=3.5,
            odds_away=4.0,
        )
        assert d["market"]["source"] == "Your odds"
        assert d["market"]["odds"] == [2.0, 3.5, 4.0]
        inv = [1 / 2.0, 1 / 3.5, 1 / 4.0]
        tot = sum(inv)
        assert d["market"]["implied_home"] == round(inv[0] / tot, 4)
        assert d["market"]["edge_home"] == round(0.5 - inv[0] / tot, 4)
        # User odds are passed into the model's market_odds path.
        assert d["probabilities"]["home_win"] == 0.5

    def test_no_odds_no_market(self):
        d = main.predict(home="Team A", away="Team B", league="EPL")
        assert d["market"] is None

    def test_derivative_markets_are_exposed(self):
        # The /api/predict payload now carries Over/Under + Asian Handicap lines
        # priced off the Dixon-Coles score matrix (Section C.8).
        d = main.predict(home="Team A", away="Team B", league="EPL")
        dm = d["derivative_markets"]
        assert set(dm) == {"over_under", "asian_handicap"}
        ou25 = next(x for x in dm["over_under"] if x["line"] == 2.5)
        assert ou25["over"] + ou25["under"] == pytest.approx(1.0, abs=1e-6)
        ah0 = next(x for x in dm["asian_handicap"] if x["handicap"] == 0.0)
        assert ah0["home"] + ah0["push"] + ah0["away"] == pytest.approx(1.0, abs=1e-3)

    def test_partial_odds_ignored(self):
        # All three odds must be present & >= 1 to activate the layer.
        d = main.predict(
            home="Team A",
            away="Team B",
            league="EPL",
            odds_home=2.0,
            odds_draw=None,
            odds_away=4.0,
        )
        assert d["market"] is None


class TestStandingsSnapshot:
    """Authoritative league-table snapshots (data/standings/<CODE>/<season>.json)
    override the results-computed table for the leagues/current seasons that
    have them — the web UI shows the official posted table instead of a table
    built from a results CSV that may lag by several gameweeks."""

    def _write_snapshot(self, dir_path, code, rows, as_of="2026-09-06"):
        folder = dir_path / code
        folder.mkdir(parents=True, exist_ok=True)
        payload = {
            "league": "EPL",
            "season": "2026-27",
            "as_of": as_of,
            "rows": rows,
        }
        (folder / "2026-27.json").write_text(json.dumps(payload))

    def test_snapshot_takes_precedence_over_results(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "_STANDINGS_DIR", tmp_path)
        self._write_snapshot(
            tmp_path,
            "E0",
            [
                {
                    "position": 1,
                    "team": "Team A",
                    "played": 3,
                    "won": 3,
                    "draw": 0,
                    "lost": 0,
                    "gf": 7,
                    "ga": 2,
                    "goal_diff": 5,
                    "points": 9,
                },
                {
                    "position": 2,
                    "team": "Team B",
                    "played": 3,
                    "won": 0,
                    "draw": 0,
                    "lost": 3,
                    "gf": 0,
                    "ga": 5,
                    "goal_diff": -5,
                    "points": 0,
                },
            ],
        )
        # A results row that contradicts the snapshot (Team B beat Team A) must
        # NOT leak into the standings while the snapshot is present.
        season_df = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-08-21"),
                    "home_team": "Team B",
                    "away_team": "Team A",
                    "home_goals": 1,
                    "away_goals": 0,
                    "result": "H",
                }
            ]
        )
        rows, meta = main._display_standings(
            "EPL", pd.Timestamp("2026-07-01"), season_df
        )
        assert [r["team"] for r in rows] == ["Team A", "Team B"]
        assert rows[0]["points"] == 9 and rows[0]["rank"] == 1
        assert meta["as_of"] == "2026-09-06"

    def test_falls_back_to_results_without_snapshot(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "_STANDINGS_DIR", tmp_path)  # empty dir
        season_df = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-08-21"),
                    "home_team": "Team A",
                    "away_team": "Team B",
                    "home_goals": 2,
                    "away_goals": 0,
                    "result": "H",
                }
            ]
        )
        rows, meta = main._display_standings(
            "EPL", pd.Timestamp("2026-07-01"), season_df
        )
        assert meta["as_of"] is None and meta["source"] == "results"
        assert rows[0]["team"] == "Team A" and rows[0]["points"] == 3

    def test_other_seasons_do_not_use_snapshot(self, tmp_path, monkeypatch):
        # A snapshot for 2026-27 must not leak into an older season's table.
        monkeypatch.setattr(main, "_STANDINGS_DIR", tmp_path)
        self._write_snapshot(tmp_path, "E0", [{"position": 1, "team": "Team A"}])
        rows, meta = main._display_standings(
            "EPL", pd.Timestamp("2025-07-01"), pd.DataFrame()
        )
        assert meta["source"] == "results"
        assert rows == []

    def test_shipped_snapshots_are_complete(self):
        # The shipped data/standings snapshots parse and cover a full 20-club
        # table in official-table order (top club first).
        for league, code, top in [
            ("EPL", "E0", "Man City"),
            ("LALIGA", "SP1", "Barcelona"),
            ("SERIEA", "I1", "Roma"),
        ]:
            snap = main._load_standings_snapshot(league, pd.Timestamp("2026-07-01"))
            assert snap is not None, f"missing shipped snapshot for {league}"
            assert snap["as_of"] == "2026-09-06"
            assert len(snap["rows"]) == 20
            assert snap["rows"][0]["team"] == top
            assert [r["rank"] for r in snap["rows"]] == list(range(1, 21))
            for r in snap["rows"]:
                assert r["goal_diff"] == r["gf"] - r["ga"]


def _match(date, home, away, hg, ag, result):
    return {
        "date": pd.Timestamp(date),
        "home_team": home,
        "away_team": away,
        "home_goals": hg,
        "away_goals": ag,
        "result": result,
    }


class TestSnapshotReconciliation:
    """When an official snapshot and the results CSV coexist the app must warn
    when one source is older than — or flat-out contradicts — the other. The
    snapshot still wins for display (it is the authoritative posted table); the
    warning exists to tell the user which source is actually driving the panel
    and which clubs disagree."""

    @staticmethod
    def _snapshot(as_of="2026-09-06"):
        return {
            "as_of": as_of,
            "rows": [
                {
                    "position": 1,
                    "team": "Team A",
                    "played": 3,
                    "won": 3,
                    "draw": 0,
                    "lost": 0,
                    "gf": 7,
                    "ga": 2,
                    "goal_diff": 5,
                    "points": 9,
                },
                {
                    "position": 2,
                    "team": "Team B",
                    "played": 3,
                    "won": 1,
                    "draw": 1,
                    "lost": 1,
                    "gf": 4,
                    "ga": 4,
                    "goal_diff": 0,
                    "points": 4,
                },
            ],
        }

    def _display(self, tmp_path, monkeypatch, season_df, as_of="2026-09-06"):
        monkeypatch.setattr(main, "_STANDINGS_DIR", tmp_path)
        folder = tmp_path / "E0"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "2026-27.json").write_text(
            json.dumps({"as_of": as_of, "rows": self._snapshot(as_of)["rows"]})
        )
        return main._display_standings("EPL", pd.Timestamp("2026-07-01"), season_df)

    def test_consistent_sources_no_warning(self, tmp_path, monkeypatch):
        # The CSV fully covers Team A and agrees with the snapshot; Team B is
        # only partly covered (played 1 vs 3) so it is legitimately skipped.
        season_df = pd.DataFrame(
            [
                _match("2026-08-22", "Team A", "Team B", 2, 0, "H"),
                _match("2026-08-29", "Team A", "Team C", 3, 1, "H"),
                _match("2026-09-05", "Team A", "Team D", 2, 1, "H"),
                _match("2026-09-05", "Team B", "Team E", 1, 1, "D"),
            ]
        )
        rows, meta = self._display(tmp_path, monkeypatch, season_df)
        assert meta["as_of"] == "2026-09-06"
        assert meta["results_through"] == "2026-09-05"
        assert meta["warning"] is None
        assert rows[0]["team"] == "Team A"  # snapshot still drives the panel

    def test_snapshot_older_than_results_warns(self, tmp_path, monkeypatch):
        # Results dated after the snapshot's as-of date -> the snapshot is stale.
        season_df = pd.DataFrame(
            [
                _match("2026-08-22", "Team A", "Team B", 2, 0, "H"),
                _match("2026-08-29", "Team A", "Team C", 3, 1, "H"),
                _match("2026-09-05", "Team A", "Team D", 2, 1, "H"),
            ]
        )
        _, meta = self._display(tmp_path, monkeypatch, season_df, as_of="2026-09-01")
        assert meta["warning"] is not None
        assert "2026-09-01" in meta["warning"] and "2026-09-05" in meta["warning"]

    def test_contradicting_results_warn(self, tmp_path, monkeypatch):
        # Team A's three results are all in the CSV and one is a loss — the
        # snapshot says three wins. Fully-covered disagreement must be flagged.
        season_df = pd.DataFrame(
            [
                _match("2026-08-22", "Team A", "Team B", 2, 0, "H"),
                _match("2026-08-29", "Team A", "Team C", 3, 1, "H"),
                _match("2026-09-05", "Team B", "Team A", 1, 0, "H"),
            ]
        )
        _, meta = self._display(tmp_path, monkeypatch, season_df)
        assert meta["warning"] is not None
        assert "Team A" in meta["warning"]

    def test_partial_csv_not_flagged_as_contradiction(self, tmp_path, monkeypatch):
        # Only Team A's first match is in the CSV (played 1 vs 3 in the
        # snapshot): coverage is incomplete, so no false "contradiction" — the
        # results file simply lags the real world.
        season_df = pd.DataFrame([_match("2026-08-22", "Team A", "Team B", 2, 0, "H")])
        _, meta = self._display(tmp_path, monkeypatch, season_df)
        assert meta["warning"] is None

    def test_results_source_meta(self, tmp_path, monkeypatch):
        # No snapshot -> the results-computed meta says so and carries the CSV's
        # newest date so the UI can label the panel accurately.
        monkeypatch.setattr(main, "_STANDINGS_DIR", tmp_path)
        season_df = pd.DataFrame([_match("2026-08-22", "Team A", "Team B", 2, 0, "H")])
        rows, meta = main._display_standings(
            "EPL", pd.Timestamp("2026-07-01"), season_df
        )
        assert meta["source"] == "results" and meta["as_of"] is None
        assert meta["results_through"] == "2026-08-22"
        assert meta["warning"] is None
        assert rows[0]["team"] == "Team A"
