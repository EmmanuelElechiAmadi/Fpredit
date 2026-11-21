"""Tests for the weekly data-refresh pipeline (Section B):

- scripts/download_data.download_plan  — current-season refresh semantics
- scripts/refresh_data                 — prune played fixtures, pasted standings
- scripts/fetch_football_data_org      — payload parsing + CSV merge (no network)

Network calls are never made: payload functions get canned dicts and the
orchestrator runs with downloads skipped and tmp dirs.
"""

import json

import pandas as pd

from scripts import download_data as dd
from scripts import fetch_football_data_org as api
from scripts import refresh_data as rd


class TestDownloadPlan:
    def test_refresh_only_overwrites_current_season(self):
        plan = dd.download_plan(current_year=2026, n_seasons=3, refresh=True)
        assert [p["season_start"] for p in plan] == [2026, 2025, 2024]
        assert plan[0]["overwrite"] is True  # current (in-progress) season
        assert plan[1]["overwrite"] is False  # completed seasons stay frozen

    def test_no_refresh_never_overwrites(self):
        plan = dd.download_plan(current_year=2026, n_seasons=2, refresh=False)
        assert all(p["overwrite"] is False for p in plan)


class TestPrunePlayedFixtures:
    def test_played_rows_removed_future_kept(self, tmp_path):
        raw = tmp_path / "raw" / "E0"
        raw.mkdir(parents=True)
        # One played result, dated after a placeholder fixture would have been.
        pd.DataFrame(
            [
                ["E0", "21/08/2026", "20:00", "Arsenal", "Coventry", 3, 0, "H"],
            ],
            columns=[
                "Div",
                "Date",
                "Time",
                "HomeTeam",
                "AwayTeam",
                "FTHG",
                "FTAG",
                "FTR",
            ],
        ).to_csv(raw / "2026-27.csv", index=False)

        fixtures = tmp_path / "fixtures" / "E0"
        fixtures.mkdir(parents=True)
        pd.DataFrame(
            [
                ["21/08/2026", "Arsenal", "Coventry", 1],  # now a result -> prune
                ["22/08/2026", "Arsenal", "Tottenham", 1],  # still to play
                ["12/09/2026", "Chelsea", "Liverpool", 4],
            ],
            columns=["Date", "HomeTeam", "AwayTeam", "Matchweek"],
        ).to_csv(fixtures / "2026-27.csv", index=False)

        pruned = rd.prune_played_fixtures(
            "EPL", tmp_path / "raw", tmp_path / "fixtures"
        )
        assert pruned == 1

        left = pd.read_csv(fixtures / "2026-27.csv", dtype=str)
        pairs = list(zip(left["HomeTeam"], left["AwayTeam"]))
        assert ("Arsenal", "Coventry") not in pairs
        assert ("Arsenal", "Tottenham") in pairs
        assert ("Chelsea", "Liverpool") in pairs

    def test_unmatched_rows_untouched(self, tmp_path):
        raw = tmp_path / "raw" / "I1"
        raw.mkdir(parents=True)
        pd.DataFrame(
            [["I1", "05/09/2026", "", "Roma", "Torino", 2, 0, "H"]],
            columns=[
                "Div",
                "Date",
                "Time",
                "HomeTeam",
                "AwayTeam",
                "FTHG",
                "FTAG",
                "FTR",
            ],
        ).to_csv(raw / "2026-27.csv", index=False)

        fixtures = tmp_path / "fixtures" / "I1"
        fixtures.mkdir(parents=True)
        pd.DataFrame(
            [["12/09/2026", "Inter", "Como", 4]],
            columns=["Date", "HomeTeam", "AwayTeam", "Matchweek"],
        ).to_csv(fixtures / "2026-27.csv", index=False)

        assert (
            rd.prune_played_fixtures("SERIEA", tmp_path / "raw", tmp_path / "fixtures")
            == 0
        )
        assert len(pd.read_csv(fixtures / "2026-27.csv")) == 1


class TestStandingsTableStep:
    def test_import_standings_table_roundtrip(self, tmp_path):
        # A pasted table is written to data/standings/<CODE>/<season>.json.
        table = tmp_path / "seriea.txt"
        table.write_text("Serie A\n1 Roma 3 3 0 0 7 2 +5 9\n2 Inter 3 2 1 0 6 3 +3 7\n")
        out = tmp_path / "standings"
        from datetime import date

        league = rd.import_standings_table(table, out_dir=out, as_of=date(2026, 9, 6))
        assert league == "SERIEA"
        payload = json.loads((out / "I1" / "2026-27.json").read_text())
        assert payload["as_of"] == "2026-09-06"
        assert [r["team"] for r in payload["rows"]] == ["Roma", "Inter"]


class TestFetchPayloadParsing:
    def test_standings_rows_maps_names_and_stats(self):
        payload = {
            "standings": [
                {
                    "type": "TOTAL",
                    "table": [
                        {
                            "position": 1,
                            "team": {
                                "name": "Manchester City FC",
                                "shortName": "Man City",
                            },
                            "playedGames": 3,
                            "won": 3,
                            "draw": 0,
                            "lost": 0,
                            "goalsFor": 7,
                            "goalsAgainst": 2,
                            "goalDifference": 5,
                            "points": 9,
                        },
                        {
                            "position": 2,
                            "team": {"name": "Arsenal FC"},
                            "playedGames": 3,
                            "won": 3,
                            "draw": 0,
                            "lost": 0,
                            "goalsFor": 6,
                            "goalsAgainst": 1,
                            "goalDifference": 5,
                            "points": 9,
                        },
                    ],
                }
            ]
        }
        rows = api.standings_rows(payload, "EPL")
        assert [r["team"] for r in rows] == ["Man City", "Arsenal"]
        assert rows[0]["points"] == 9 and rows[0]["goal_diff"] == 5

    def test_fixture_rows_only_scheduled(self):
        payload = {
            "matches": [
                {
                    "utcDate": "2026-09-12T14:00:00Z",
                    "status": "SCHEDULED",
                    "matchday": 4,
                    "homeTeam": {"name": "Arsenal FC"},
                    "awayTeam": {"name": "Chelsea FC"},
                },
                {
                    "utcDate": "2026-09-10T19:45:00Z",
                    "status": "FINISHED",
                    "homeTeam": {"name": "Everton FC"},
                    "awayTeam": {"name": "Liverpool FC"},
                },
            ]
        }
        rows = api.fixture_rows(payload, "EPL")
        assert len(rows) == 1
        assert rows[0] == {
            "date": "2026-09-12",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "matchweek": 4,
        }

    def test_result_rows_only_finished(self):
        payload = {
            "matches": [
                {
                    "utcDate": "2026-09-10T19:45:00Z",
                    "status": "FINISHED",
                    "homeTeam": {"name": "Everton FC"},
                    "awayTeam": {"name": "Liverpool FC"},
                    "score": {"fullTime": {"home": 2, "away": 2}},
                }
            ]
        }
        rows = api.result_rows(payload, "EPL")
        assert rows[0]["FTR"] == "D"
        assert rows[0]["Date"] == "10/09/2026"
        assert rows[0]["HomeTeam"] == "Everton"

    def test_merge_results_dedup_and_preserves_columns(self, tmp_path):
        data_dir = tmp_path / "raw"
        (data_dir / "E0").mkdir(parents=True)
        pd.DataFrame(
            [["E0", "10/09/2026", "Everton", "Liverpool", 2, 2, "D"]],
            columns=["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"],
        ).to_csv(data_dir / "E0" / "2026-27.csv", index=False)

        rows = [
            {  # duplicate -> skipped
                "Div": "E0",
                "Date": "10/09/2026",
                "HomeTeam": "Everton",
                "AwayTeam": "Liverpool",
                "FTHG": 2,
                "FTAG": 2,
                "FTR": "D",
            },
            {  # new -> appended
                "Div": "E0",
                "Date": "12/09/2026",
                "HomeTeam": "Arsenal",
                "AwayTeam": "Chelsea",
                "FTHG": 1,
                "FTAG": 0,
                "FTR": "H",
            },
        ]
        added = api.merge_results("EPL", rows, str(data_dir), "2026-27")
        assert added == 1
        df = pd.read_csv(data_dir / "E0" / "2026-27.csv", dtype=str)
        assert len(df) == 2
        assert list(df.columns) == [
            "Div",
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
            "FTR",
        ]


class TestRefreshCli:
    def test_offline_run_prunes_and_imports(self, tmp_path, capsys):
        # The scheduled/manual command without network: downloads skipped, a
        # pasted table imported, fixtures pruned, exit code 0.
        table = tmp_path / "laliga.txt"
        table.write_text("Spanish La Liga\n1 Barcelona 4 4 0 0 17 2 +15 12\n")
        code = rd.main(
            [
                "--leagues",
                "LALIGA",
                "--no-download",
                "--data-dir",
                str(tmp_path / "raw"),
                "--fixtures-dir",
                str(tmp_path / "fixtures"),
                "--standings-dir",
                str(tmp_path / "standings"),
                "--standings-table",
                str(table),
                "--as-of",
                "2026-09-06",
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "Pruned 0 played fixture row(s)" in out
        assert (tmp_path / "standings" / "SP1" / "2026-27.json").exists()


    def test_sync_fixtures_replace_drops_placeholders(self, tmp_path, monkeypatch):
        # The automatic-refresh path replaces the season fixture file with the
        # API's official list rather than merging beside placeholder rows.
        folder = tmp_path / "I1"
        folder.mkdir(parents=True)
        (folder / "2026-27.csv").write_text(
            "Date,HomeTeam,AwayTeam,Matchweek\n22/08/2026,Roma,Torino,1\n"
        )

        def fake_request(path, api_key, query=None):
            return {
                "matches": [
                    {
                        "utcDate": "2026-09-12T16:00:00Z",
                        "status": "SCHEDULED",
                        "matchday": 4,
                        "homeTeam": {"name": "Inter"},
                        "awayTeam": {"name": "Como"},
                    }
                ]
            }

        monkeypatch.setattr(api, "_request", fake_request)
        monkeypatch.setattr(api, "_season_param", lambda today: "2026")
        monkeypatch.setattr(api, "season_for_date", lambda d: "2026-27")
        added = api.sync_fixtures(
            "SERIEA", "test-key", fixtures_dir=tmp_path, replace=True
        )
        assert added == 1
        df = pd.read_csv(folder / "2026-27.csv", dtype=str)
        assert len(df) == 1
        assert df.iloc[0]["HomeTeam"] == "Inter"

