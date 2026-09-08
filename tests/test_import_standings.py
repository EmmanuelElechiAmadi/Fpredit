"""Tests for scripts/import_standings.py — pasting a website league table into
data/standings/<CODE>/<season>.json.

The parser must handle the copy-pasted BBC-style layout (each row split over
three lines: position / team name / stats) and map display names onto the
football-data spellings the rest of the project uses.
"""

from datetime import date

import pytest

from scripts import import_standings as imp

SPLIT_EPL = """Premier League
Last updated: 6 September, 6:28pm
Position\tTeam Team\tPl Played\tW Won\tD Drawn\tL Lost\tF Goals For\tA Goals Against\tGD Goal Difference\tPts Points
1\t \nManchester City\n3\t3\t0\t0\t7\t2\t+5\t9
2\t \nArsenal\n3\t3\t0\t0\t6\t1\t+5\t9
3\t \nHull City\n3\t2\t1\t0\t3\t0\t+3\t7
"""

ONE_LINE_LALIGA = """Spanish La Liga
1 Barcelona 4 4 0 0 17 2 +15 12
2 Real Betis 4 3 0 1 5 5 0 9
3 Athletic Club 4 2 0 2 6 5 +1 6
"""


class TestImportStandings:
    def test_parse_split_layout_epl(self):
        rows = imp.parse_table(SPLIT_EPL, "EPL")
        assert [r["team"] for r in rows] == ["Man City", "Arsenal", "Hull"]
        assert rows[0] == {
            "position": 1,
            "team": "Man City",
            "played": 3,
            "won": 3,
            "draw": 0,
            "lost": 0,
            "gf": 7,
            "ga": 2,
            "goal_diff": 5,
            "points": 9,
        }

    def test_parse_one_line_layout_laliga(self):
        rows = imp.parse_table(ONE_LINE_LALIGA, "LALIGA")
        assert [r["team"] for r in rows] == ["Barcelona", "Betis", "Ath Bilbao"]
        assert rows[1]["goal_diff"] == 0 and rows[1]["points"] == 9
        assert rows[2]["team"] == "Ath Bilbao"

    def test_detect_league_from_title(self):
        assert imp.detect_league(SPLIT_EPL) == "EPL"
        assert imp.detect_league(ONE_LINE_LALIGA) == "LALIGA"
        with pytest.raises(ValueError):
            imp.detect_league("some random text")

    def test_season_for_date(self):
        assert imp.season_for_date(date(2026, 9, 6)) == "2026-27"
        assert imp.season_for_date(date(2027, 5, 1)) == "2026-27"
        assert imp.season_for_date(date(2026, 1, 15)) == "2025-26"

    def test_write_and_read_roundtrip(self, tmp_path):
        rows = imp.parse_table(SPLIT_EPL, "EPL")
        path = imp.write_snapshot(
            "EPL", rows, date(2026, 9, 6), "2026-27", out_dir=tmp_path
        )
        assert path == tmp_path / "E0" / "2026-27.json"
        data = __import__("json").loads(path.read_text())
        assert data["as_of"] == "2026-09-06"
        assert data["rows"][0]["team"] == "Man City"
