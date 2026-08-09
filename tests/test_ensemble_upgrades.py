"""
Integration tests for the upgraded ensemble: market-residual features and
xG-derived features, plus the dynamic model's presence in the output.
"""

import numpy as np
import pytest

from src.data_loader import generate_synthetic_league
from src.ensemble import FootballEnsemble
from src.xg_loader import generate_synthetic_xg


@pytest.fixture
def demo_data():
    return generate_synthetic_league(n_teams=8, n_seasons=2)


@pytest.fixture
def demo_data_with_odds(demo_data):
    df = demo_data.reset_index(drop=True)
    rng = np.random.default_rng(42)
    df["B365H"] = rng.uniform(1.4, 2.2, size=len(df))
    df["B365D"] = rng.uniform(3.0, 4.2, size=len(df))
    df["B365A"] = rng.uniform(2.8, 5.0, size=len(df))
    return df


@pytest.fixture
def demo_data_with_xg(demo_data):
    return generate_synthetic_xg(demo_data.reset_index(drop=True))


class TestMarketFeatures:
    def test_market_columns_in_feature_set(self, demo_data_with_odds):
        model = FootballEnsemble().fit(demo_data_with_odds)
        assert "market_home" in model.feature_cols
        assert "market_draw" in model.feature_cols
        assert "market_away" in model.feature_cols

    def test_market_absent_without_odds(self, demo_data):
        model = FootballEnsemble().fit(demo_data)
        assert "market_home" not in model.feature_cols

    def test_predict_with_market_odds(self, demo_data_with_odds):
        model = FootballEnsemble().fit(demo_data_with_odds)
        result = model.predict("Team A", "Team B", market_odds=(2.0, 3.4, 3.8))
        assert result["home_win"] + result["draw"] + result[
            "away_win"
        ] == pytest.approx(1.0, abs=1e-6)

    def test_predict_without_market_odds_neutralizes(self, demo_data_with_odds):
        model = FootballEnsemble().fit(demo_data_with_odds)
        r_no_market = model.predict("Team A", "Team B")
        r_market = model.predict("Team A", "Team B", market_odds=(2.0, 3.4, 3.8))
        # Different market signal -> different output (model is market-aware)
        assert r_no_market["home_win"] != pytest.approx(r_market["home_win"], abs=1e-9)

    def test_use_market_features_false(self, demo_data_with_odds):
        model = FootballEnsemble(use_market_features=False).fit(demo_data_with_odds)
        assert "market_home" not in model.feature_cols

    def test_line_movement_features_when_present(self, demo_data_with_odds):
        df = demo_data_with_odds.copy()
        df["PH"], df["PD"], df["PA"] = 2.0, 3.5, 4.0
        df["PSH"], df["PSD"], df["PSA"] = 1.9, 3.6, 4.2
        model = FootballEnsemble().fit(df)
        assert "line_mv_home" in model.feature_cols
        assert "line_mv_abs" in model.feature_cols


class TestXGFeatures:
    def test_xg_columns_in_feature_set(self, demo_data_with_xg):
        model = FootballEnsemble().fit(demo_data_with_xg)
        assert model.has_xg
        assert "xg_home" in model.feature_cols

    def test_fit_with_separate_xg_frame(self, demo_data, demo_data_with_xg):
        model = FootballEnsemble().fit(demo_data, xg_df=demo_data_with_xg)
        assert model.has_xg
        assert "xg_home" in model.feature_cols

    def test_predict_valid_with_xg(self, demo_data_with_xg):
        model = FootballEnsemble().fit(demo_data_with_xg)
        result = model.predict("Team A", "Team B")
        assert "xg" in result["component_probs"]
        assert result["home_win"] + result["draw"] + result[
            "away_win"
        ] == pytest.approx(1.0, abs=1e-6)

    def test_no_xg_when_absent(self, demo_data):
        model = FootballEnsemble().fit(demo_data)
        assert not model.has_xg
        assert "xg_home" not in model.feature_cols
        result = model.predict("Team A", "Team B")
        assert "xg" not in result["component_probs"]


class TestDynamicComponent:
    def test_dynamic_in_component_probs(self, demo_data):
        model = FootballEnsemble().fit(demo_data)
        result = model.predict("Team A", "Team B")
        assert "dynamic" in result["component_probs"]
        assert sum(result["component_probs"]["dynamic"]) == pytest.approx(1.0, abs=1e-6)

    def test_dynamic_feature_cols_present(self, demo_data):
        model = FootballEnsemble().fit(demo_data)
        assert {"dyn_home", "dyn_draw", "dyn_away"}.issubset(model.feature_cols)

    def test_advanced_form_features_present(self, demo_data):
        model = FootballEnsemble().fit(demo_data)
        for col in ("load_diff", "congestion_diff", "pos_diff", "pagerank_diff"):
            assert col in model.feature_cols

    def test_deterministic_with_xg(self, demo_data_with_xg):
        e1 = FootballEnsemble().fit(demo_data_with_xg)
        r1 = e1.predict("Team A", "Team B")
        e2 = FootballEnsemble().fit(demo_data_with_xg)
        r2 = e2.predict("Team A", "Team B")
        assert r1["home_win"] == pytest.approx(r2["home_win"], abs=1e-6)
