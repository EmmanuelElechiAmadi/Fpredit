"""API-level tests for the fixtures endpoints (GET/POST/DELETE/generate-round-robin).

Hermetic: the model-fitting warm-up is disabled and the league data + fixture
file I/O are redirected to synthetic/in-memory paths so no real data or model
fits are touched.
"""

from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app.main as main
from src import fixtures as fmod
from src.data_loader import generate_synthetic_league


@pytest.fixture
def client(monkeypatch, tmp_path):
    league_df = generate_synthetic_league(n_teams=6, n_seasons=1)
    monkeypatch.setattr(main, "_warm_caches", lambda: None)  # skip model warm-up
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
