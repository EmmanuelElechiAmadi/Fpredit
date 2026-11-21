"""Tests for src.update.normalise — dates, team mapping, types, odds."""

import pandas as pd

from src.update.normalise import normalise_season_csv

TEAM_MAP = {
    "EPL": {
        "Manchester United": "Man United",
        "Arsenal": "Arsenal",
        "Chelsea": "Chelsea",
        "Man United": "Man United",
    }
}


def _raw():
    return pd.DataFrame(
        {
            "Div": ["E0", "E0"],
            "Date": ["14/08/2026", "21/08/2026"],
            "Time": ["20:00", "15:00"],
            "HomeTeam": ["Manchester United", "Arsenal"],
            "AwayTeam": ["Arsenal", "Chelsea"],
            "FTHG": ["2", "1"],
            "FTAG": ["1", "1"],
            "FTR": ["H", "D"],
            "HTHG": ["1", "0"],
            "HTAG": ["0", "1"],
            "HTR": ["H", "A"],
            "HS": ["12", "8"],
            "AS": ["8", "12"],
            "HST": ["5", "3"],
            "AST": ["3", "5"],
            "B365H": [1.6, 2.5],
            "B365D": [3.9, 3.4],
            "B365A": [5.5, 2.6],
        }
    )


def test_maps_teams_and_dates():
    out, odds, unmapped = normalise_season_csv(_raw(), "EPL", "2026-27", TEAM_MAP)
    assert unmapped == set()
    assert out.loc[0, "home"] == "Man United"
    assert out.loc[0, "away"] == "Arsenal"
    assert out.loc[0, "kickoff_utc"] == pd.Timestamp("2026-08-14 20:00")
    assert out.loc[0, "date"] == pd.Timestamp("2026-08-14")


def test_coerces_int_types():
    out, _, _ = normalise_season_csv(_raw(), "EPL", "2026-27", TEAM_MAP)
    assert int(out.loc[0, "fthg"]) == 2
    assert int(out.loc[0, "hst"]) == 5
    assert int(out.loc[1, "as"]) == 12


def test_extracts_1x2_odds():
    out, odds, _ = normalise_season_csv(_raw(), "EPL", "2026-27", TEAM_MAP)
    b365 = [o for o in odds if o["bookmaker"] == "B365" and o["market"] == "1X2"]
    assert len(b365) == 2
    assert b365[0]["home"] == 1.6
    assert b365[0]["is_closing"] == 1
    # odds are keyed back to the match natural key
    assert b365[0]["_key"].endswith("|Man United|Arsenal")


def test_derives_ftr_when_missing():
    raw = _raw()
    raw["FTR"] = [None, None]
    out, _, _ = normalise_season_csv(raw, "EPL", "2026-27", TEAM_MAP)
    assert out.loc[0, "ftr"] == "H"
    assert out.loc[1, "ftr"] == "D"


def test_unmapped_team_halts():
    raw = _raw()
    raw.loc[0, "HomeTeam"] = "Unknown FC"
    _, _, unmapped = normalise_season_csv(raw, "EPL", "2026-27", TEAM_MAP)
    assert "Unknown FC" in unmapped
