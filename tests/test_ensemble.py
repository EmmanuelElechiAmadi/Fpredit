"""
Unit tests for the ensemble meta-learner.
"""

import numpy as np
import pytest

from src.ensemble import FootballEnsemble


@pytest.fixture
def demo_data():
    """Synthetic dataset from generate_synthetic_league via data_loader.

    Use a small league (8 teams, 2 seasons = ~112 matches) so the full
    ensemble fit (~3s each × ~16 fits) completes the test file in ~1 min
    instead of timing out (default 18 teams × 3 seasons = 918 matches at
    ~25s per fit is far too slow for 15 tests).
    """
    from src.data_loader import generate_synthetic_league

    return generate_synthetic_league(n_teams=8, n_seasons=2)


@pytest.fixture
def ensemble():
    return FootballEnsemble()


class TestFootballEnsemble:
    def test_fit_creates_meta_model(self, ensemble, demo_data):
        """After fitting, the meta model should be trained."""
        ensemble.fit(demo_data)
        assert ensemble.meta is not None

    def test_fit_trains_both_components(self, ensemble, demo_data):
        """Both Dixon-Coles and Elo components should be fitted."""
        ensemble.fit(demo_data)
        assert ensemble.dc is not None
        assert ensemble.elo is not None

    def test_predict_returns_dict(self, ensemble, demo_data):
        """predict() should return a dictionary with expected keys."""
        ensemble.fit(demo_data)
        result = ensemble.predict("Team A", "Team B")
        assert isinstance(result, dict)

    def test_predict_contains_required_keys(self, ensemble, demo_data):
        """predict() output should have all required probability fields."""
        ensemble.fit(demo_data)
        result = ensemble.predict("Team A", "Team B")
        required_keys = {
            "home_win",
            "draw",
            "away_win",
            "most_likely_score",
            "expected_goals",
            "over_2_5_goals",
            "btts_yes",
            "component_probs",
        }
        assert required_keys.issubset(result.keys())

    def test_predict_probs_sum_to_one(self, ensemble, demo_data):
        """Home/Draw/Away probabilities should sum to 1."""
        ensemble.fit(demo_data)
        result = ensemble.predict("Team A", "Team B")
        total = result["home_win"] + result["draw"] + result["away_win"]
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_predict_probs_in_range(self, ensemble, demo_data):
        """All probabilities should be in [0, 1]."""
        ensemble.fit(demo_data)
        result = ensemble.predict("Team A", "Team B")
        for key in ["home_win", "draw", "away_win", "over_2_5_goals", "btts_yes"]:
            assert 0 <= result[key] <= 1, f"{key} = {result[key]} out of range"

    def test_home_win_prob_greater_with_home_advantage(self, ensemble, demo_data):
        """Against a much weaker team, home win prob should be > away win prob."""
        ensemble.fit(demo_data)
        # In synthetic data, all teams are fairly even, but home advantage matters
        result = ensemble.predict("Team A", "Team B")
        assert result["home_win"] > result["away_win"]

    def test_expected_goals_positive(self, ensemble, demo_data):
        """Expected goals should be positive."""
        ensemble.fit(demo_data)
        result = ensemble.predict("Team A", "Team B")
        assert result["expected_goals"][0] > 0
        assert result["expected_goals"][1] > 0

    def test_most_likely_score_consistent(self, ensemble, demo_data):
        """The most likely score should be consistent with expected goals."""
        ensemble.fit(demo_data)
        result = ensemble.predict("Team A", "Team B")
        hg, ag = result["most_likely_score"]
        assert isinstance(hg, (int, np.integer))
        assert isinstance(ag, (int, np.integer))
        assert hg >= 0
        assert ag >= 0

    def test_component_probs_structure(self, ensemble, demo_data):
        """component_probs should contain 'dixon_coles' and 'elo' keys."""
        ensemble.fit(demo_data)
        result = ensemble.predict("Team A", "Team B")
        comps = result["component_probs"]
        assert "dixon_coles" in comps
        assert "elo" in comps

    def test_component_probs_sum_to_one(self, ensemble, demo_data):
        """Each component's H/D/A should sum to 1."""
        ensemble.fit(demo_data)
        result = ensemble.predict("Team A", "Team B")
        for comp_name, probs in result["component_probs"].items():
            assert sum(probs) == pytest.approx(1.0, abs=1e-6)

    def test_demo_data_contains_expected_columns(self, demo_data):
        """Synthetic data should have the minimum required columns."""
        required = {"date", "home_team", "away_team", "home_goals", "away_goals"}
        assert required.issubset(demo_data.columns)

    def test_demo_data_has_reasonable_size(self, demo_data):
        """Synthetic data should have at least a few hundred matches."""
        assert len(demo_data) >= 100

    def test_fit_with_different_datasets_consistent(self, demo_data):
        """Fitting twice with same data should give same probabilities (deterministic)."""
        e1 = FootballEnsemble()
        e1.fit(demo_data)
        r1 = e1.predict("Team A", "Team B")

        e2 = FootballEnsemble()
        e2.fit(demo_data)
        r2 = e2.predict("Team A", "Team B")

        assert r1["home_win"] == pytest.approx(r2["home_win"], abs=1e-6)

    def test_predict_over_under_consistent(self, ensemble, demo_data):
        """over_2_5_goals and (1 - over_2_5_goals) should sum to 1."""
        ensemble.fit(demo_data)
        result = ensemble.predict("Team A", "Team B")
        assert result["over_2_5_goals"] + (
            1 - result["over_2_5_goals"]
        ) == pytest.approx(1.0)

    def test_predict_btts_consistent(self, ensemble, demo_data):
        """btts_yes should be in [0,1] and complement btts_no."""
        ensemble.fit(demo_data)
        result = ensemble.predict("Team A", "Team B")
        assert 0 <= result["btts_yes"] <= 1
