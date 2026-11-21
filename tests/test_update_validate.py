"""Tests for src.update.validate — Section 5 checks and quarantine."""

from datetime import date

import pandas as pd

from src.update import config as upconfig
from src.update.validate import validate_matches, validate_odds


def _cfg():
    return upconfig.load_update_config()


def _row(**overrides):
    row = {
        "league": "EPL",
        "season": "2026-27",
        "date": pd.Timestamp("2026-08-14"),
        "home": "Arsenal",
        "away": "Chelsea",
        "fthg": 2,
        "ftag": 1,
        "ftr": "H",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_valid_row_passes():
    valid, quarantined, _ = validate_matches(_row(), _cfg(), today=date(2026, 12, 1))
    assert len(valid) == 1
    assert quarantined == []


def test_result_disagrees_with_goals_quarantined():
    valid, q, _ = validate_matches(_row(ftr="A"), _cfg(), today=date(2026, 12, 1))
    assert len(valid) == 0
    assert q[0]["reason"] == "result disagrees with goals"


def test_future_date_quarantined():
    valid, q, _ = validate_matches(
        _row(date=pd.Timestamp("2026-12-15")), _cfg(), today=date(2026, 12, 1)
    )
    assert len(valid) == 0
    assert q[0]["reason"] == "match date in the future"


def test_negative_goals_quarantined():
    valid, q, _ = validate_matches(_row(fthg=-1), _cfg(), today=date(2026, 12, 1))
    assert len(valid) == 0
    assert q[0]["reason"] == "fthg negative"


def test_duplicate_key_quarantined():
    df = pd.concat([_row(), _row()], ignore_index=True)
    valid, q, _ = validate_matches(df, _cfg(), today=date(2026, 12, 1))
    assert len(valid) == 1
    assert q[0]["reason"] == "duplicate match key"


def test_shots_on_target_not_more_than_shots():
    df = _row(hs=5, hst=6)
    valid, q, _ = validate_matches(df, _cfg(), today=date(2026, 12, 1))
    assert len(valid) == 0
    assert q[0]["reason"] == "hst > hs"


def test_odds_overround_rejected():
    good = {"market": "1X2", "home": 2.0, "draw": 3.5, "away": 4.0}
    bad = {"market": "1X2", "home": 1.05, "draw": 1.05, "away": 1.05}
    valid, rejected = validate_odds([good, bad])
    assert len(valid) == 1
    assert len(rejected) == 1
    assert "overround" in rejected[0]["reason"]
