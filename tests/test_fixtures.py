"""Tests for the fixtures loader/editor (new-season previews).
Uses tmp dirs — never touches the real data/fixtures folder.
"""

import pandas as pd
import pytest

from src.fixtures import (
    add_fixture,
    delete_fixture,
    generate_round_robin,
    load_fixtures,
    season_for_date,
    write_fixtures_batch,
)


class TestSeasonForDate:
    def test_autumn_date(self):
        assert season_for_date("2026-08-15") == "2026-27"

    def test_spring_date_belongs_to_previous_season(self):
        assert season_for_date("2027-05-01") == "2026-27"

    def test_july_boundary(self):
        assert season_for_date("2027-07-01") == "2027-28"


class TestAddLoadDelete:
    def test_round_trip(self, tmp_path):
        add_fixture(
            "EPL",
            "Arsenal",
            "Chelsea",
            "2026-08-15",
            matchweek=1,
            raw_dir=str(tmp_path),
        )
        add_fixture(
            "EPL",
            "Liverpool",
            "Man City",
            "2026-08-15",
            matchweek=1,
            raw_dir=str(tmp_path),
        )
        df = load_fixtures(str(tmp_path), "EPL", today=pd.Timestamp("2026-08-01"))
        assert len(df) == 2
        assert list(df.columns) == [
            "date",
            "home_team",
            "away_team",
            "matchweek",
            "season",
            "source",
        ]
        assert df["season"].tolist() == ["2026-27", "2026-27"]
        assert df.iloc[0]["home_team"] == "Arsenal"
        assert df.iloc[0]["matchweek"] == 1

    def test_append_preserves_existing_rows(self, tmp_path):
        add_fixture("EPL", "Arsenal", "Chelsea", "2026-08-15", raw_dir=str(tmp_path))
        add_fixture("EPL", "Arsenal", "Man City", "2026-08-22", raw_dir=str(tmp_path))
        df = load_fixtures(str(tmp_path), "EPL", today=pd.Timestamp("2026-08-01"))
        assert len(df) == 2
        # Same season file holds both rows
        file = tmp_path / "E0" / "2026-27.csv"
        assert file.exists()
        raw = pd.read_csv(file)
        assert len(raw) == 2

    def test_add_same_fixture_twice_is_idempotent(self, tmp_path):
        add_fixture("EPL", "Arsenal", "Chelsea", "2026-08-15", raw_dir=str(tmp_path))
        add_fixture("EPL", "Arsenal", "Chelsea", "2026-08-15", raw_dir=str(tmp_path))
        df = load_fixtures(str(tmp_path), "EPL", today=pd.Timestamp("2026-08-01"))
        assert len(df) == 1

    def test_past_fixtures_filtered_out(self, tmp_path):
        add_fixture("EPL", "Arsenal", "Chelsea", "2026-08-01", raw_dir=str(tmp_path))
        df = load_fixtures(str(tmp_path), "EPL", today=pd.Timestamp("2026-08-10"))
        assert len(df) == 0

    def test_delete_removes_one_row(self, tmp_path):
        add_fixture(
            "EPL",
            "Arsenal",
            "Chelsea",
            "2026-08-15",
            matchweek=1,
            raw_dir=str(tmp_path),
        )
        add_fixture(
            "EPL",
            "Liverpool",
            "Man City",
            "2026-08-15",
            matchweek=1,
            raw_dir=str(tmp_path),
        )
        assert delete_fixture(
            "EPL", "Arsenal", "Chelsea", "2026-08-15", raw_dir=str(tmp_path)
        )
        df = load_fixtures(str(tmp_path), "EPL", today=pd.Timestamp("2026-08-01"))
        assert len(df) == 1
        assert df.iloc[0]["home_team"] == "Liverpool"
        # Deleting the last row removes the file entirely
        assert delete_fixture(
            "EPL", "Liverpool", "Man City", "2026-08-15", raw_dir=str(tmp_path)
        )
        assert not (tmp_path / "E0" / "2026-27.csv").exists()

    def test_delete_missing_returns_false(self, tmp_path):
        assert not delete_fixture(
            "EPL", "Arsenal", "Chelsea", "2026-08-15", raw_dir=str(tmp_path)
        )

    def test_rejects_invalid(self, tmp_path):
        with pytest.raises(ValueError):
            add_fixture(
                "EPL", "Arsenal", "Arsenal", "2026-08-15", raw_dir=str(tmp_path)
            )
        with pytest.raises(ValueError):
            add_fixture("EPL", "", "Chelsea", "2026-08-15", raw_dir=str(tmp_path))


class TestLoadFormats:
    def test_minimal_format_csv(self, tmp_path):
        folder = tmp_path / "E0"
        folder.mkdir(parents=True)
        (folder / "2026-27.csv").write_text(
            "date,home_team,away_team,matchweek\n"
            "2026-08-15,Arsenal,Chelsea,1\n"
            "2026-08-15,Liverpool,Man City,1\n"
        )
        df = load_fixtures(str(tmp_path), "EPL", today=pd.Timestamp("2026-08-01"))
        assert len(df) == 2

    def test_footballdata_style_csv(self, tmp_path):
        folder = tmp_path / "E0"
        folder.mkdir(parents=True)
        (folder / "2026-27.csv").write_text(
            "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG\n"
            "E0,14/08/2026,20:00,Arsenal,Chelsea,,\n"
            "E0,15/08/2026,12:30,Liverpool,Man City,,\n"
        )
        df = load_fixtures(str(tmp_path), "EPL", today=pd.Timestamp("2026-08-01"))
        assert len(df) == 2
        assert df.iloc[0]["home_team"] == "Arsenal"
        assert df.iloc[1]["away_team"] == "Man City"


class TestWriteFixturesBatch:
    def test_batch_merges_and_dedupes(self, tmp_path):
        add_fixture(
            "EPL",
            "Arsenal",
            "Chelsea",
            "2026-08-15",
            matchweek=1,
            raw_dir=str(tmp_path),
        )
        batch = pd.DataFrame(
            {
                "date": ["2026-08-15", "2026-08-22", "2026-08-29"],
                "home_team": ["Arsenal", "Liverpool", "Man City"],
                "away_team": ["Chelsea", "Everton", "Man United"],
                "matchweek": [1, 2, 3],
            }
        )
        added = write_fixtures_batch("EPL", batch, raw_dir=str(tmp_path))
        # Existing Arsenal-Chelsea is a duplicate -> skipped; 2 new rows added.
        assert added == 2
        df = load_fixtures(str(tmp_path), "EPL", today=pd.Timestamp("2026-08-01"))
        assert len(df) == 3

    def test_batch_all_duplicates_adds_nothing(self, tmp_path):
        add_fixture("EPL", "Arsenal", "Chelsea", "2026-08-15", raw_dir=str(tmp_path))
        batch = pd.DataFrame(
            {
                "date": ["2026-08-15"],
                "home_team": ["Arsenal"],
                "away_team": ["Chelsea"],
            }
        )
        assert write_fixtures_batch("EPL", batch, raw_dir=str(tmp_path)) == 0


class TestGenerateRoundRobin:
    TEAMS = [
        "Arsenal",
        "Chelsea",
        "Liverpool",
        "Man City",
        "Man United",
        "Tottenham",
    ]

    def test_even_teams_full_schedule(self):
        rr = generate_round_robin(self.TEAMS)
        # 6 teams -> 5 rounds per half -> 30 matches, matchweeks 1..10
        assert len(rr) == 30
        assert rr["matchweek"].nunique() == 10
        assert rr["matchweek"].max() == 10
        for t in self.TEAMS:
            assert len(rr[rr["home_team"] == t]) == 5
            assert len(rr[rr["away_team"] == t]) == 5

    def test_every_pair_meets_twice(self):
        rr = generate_round_robin(self.TEAMS)
        from collections import Counter

        pairs = Counter(
            tuple(sorted([r["home_team"], r["away_team"]])) for _, r in rr.iterrows()
        )
        assert all(v == 2 for v in pairs.values())

    def test_odd_teams_adds_bye(self):
        rr = generate_round_robin(self.TEAMS[:5])
        # 5 teams -> 5 rounds per half, one bye each round -> 20 matches
        assert len(rr) == 20
        assert "(bye)" not in set(rr["home_team"]) | set(rr["away_team"])
        for t in self.TEAMS[:5]:
            assert len(rr[rr["home_team"] == t]) == 4

    def test_too_few_teams(self):
        with pytest.raises(ValueError):
            generate_round_robin(["A", "B", "C"])
