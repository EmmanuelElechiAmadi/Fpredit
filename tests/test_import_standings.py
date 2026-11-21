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

VERTICAL_SERIEA = """Serie A
1
Roma
Giocate: 3
Vittorie: 3
Pareggi: 0
Sconfitte: 0
Gol fatti: 7
Gol subiti: 2
Differenza reti: +5
Punti: 9
2
Inter
Giocate: 3
Vittorie: 2
Pareggi: 1
Sconfitte: 0
Gol fatti: 6
Gol subiti: 3
Differenza reti: +3
Punti: 7
"""

VERTICAL_ENGLISH = """Serie A table
1) Roma
Played: 3
Won: 3
Drawn: 0
Lost: 0
Goals For: 7
Goals Against: 2
Goal Difference: +5
Points: 9
2) AC Milan
Played: 3
Won: 2
Drawn: 1
Lost: 0
Goals For: 5
Goals Against: 3
Goal Difference: +2
Points: 7
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

    def test_parse_vertical_italian_layout(self):
        # Official-Serie-A-style vertical cards: position + name lines followed
        # by labelled stat lines (Italian), clubs stacked without blank lines.
        rows = imp.parse_table(VERTICAL_SERIEA, "SERIEA")
        assert [r["team"] for r in rows] == ["Roma", "Inter"]
        assert rows[0] == {
            "position": 1,
            "team": "Roma",
            "played": 3,
            "won": 3,
            "draw": 0,
            "lost": 0,
            "gf": 7,
            "ga": 2,
            "goal_diff": 5,
            "points": 9,
        }
        assert rows[1]["team"] == "Inter" and rows[1]["points"] == 7

    def test_parse_vertical_english_layout_with_pos_name(self):
        # English labels with position + team on one line ("1) Roma").
        rows = imp.parse_table(VERTICAL_ENGLISH, "SERIEA")
        assert [r["team"] for r in rows] == ["Roma", "Milan"]  # AC Milan -> Milan
        assert rows[0]["goal_diff"] == 5
        assert rows[1]["won"] == 2 and rows[1]["draw"] == 1

    def test_vertical_labels_in_any_order(self):
        # Labels can arrive out of the P/W/D/L order — values bind to their
        # label, not their position in the block.
        text = """Serie A
1
Napoli
Punti: 9
Gol subiti: 2
Vittorie: 3
Giocate: 3
Gol fatti: 7
Pareggi: 0
Sconfitte: 0
"""
        rows = imp.parse_table(text, "SERIEA")
        assert rows[0]["team"] == "Napoli"
        assert rows[0]["points"] == 9 and rows[0]["gf"] == 7
        assert rows[0]["goal_diff"] == 5
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
