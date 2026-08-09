"""
Unit tests for the Dixon-Coles model.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.dixon_coles import DixonColes, _tau


class TestTauFunction:
    """The low-score correlation adjustment function."""

    def test_tau_no_correction_for_high_scores(self):
        """Non low-score results should have tau=1.0 (no adjustment)."""
        assert _tau(3, 2, 1.5, 1.2, -0.05) == 1.0
        assert _tau(0, 3, 1.5, 1.2, -0.05) == 1.0
        assert _tau(2, 0, 1.5, 1.2, -0.05) == 1.0

    def test_tau_00(self):
        """Tau for 0-0: 1 - lam*mu*rho"""
        result = _tau(0, 0, 2.0, 1.0, -0.1)
        expected = 1 - (2.0 * 1.0 * (-0.1))
        assert result == pytest.approx(expected)

    def test_tau_01(self):
        """Tau for 0-1: 1 + lam*rho"""
        result = _tau(0, 1, 2.0, 1.0, -0.1)
        expected = 1 + (2.0 * (-0.1))
        assert result == pytest.approx(expected)

    def test_tau_10(self):
        """Tau for 1-0: 1 + mu*rho"""
        result = _tau(1, 0, 2.0, 1.0, -0.1)
        expected = 1 + (1.0 * (-0.1))
        assert result == pytest.approx(expected)

    def test_tau_11(self):
        """Tau for 1-1: 1 - rho"""
        result = _tau(1, 1, 2.0, 1.0, -0.1)
        expected = 1 - (-0.1)
        assert result == pytest.approx(expected)


class TestDixonColes:
    """Full Dixon-Coles model tests."""

    @pytest.fixture
    def simple_matches(self):
        """A minimal dataset with 2 teams and several matches."""
        base = datetime(2024, 1, 1)
        return [
            {
                "date": base + timedelta(days=i * 7),
                "home_team": "A",
                "away_team": "B",
                "home_goals": 2,
                "away_goals": 1,
            }
            for i in range(10)
        ]

    @pytest.fixture
    def balanced_matches(self):
        """20 matches where home and away teams trade wins."""
        base = datetime(2024, 1, 1)
        matches = []
        for i in range(10):
            matches.append(
                {
                    "date": base + timedelta(days=i * 7),
                    "home_team": "A",
                    "away_team": "B",
                    "home_goals": 2,
                    "away_goals": 1,
                }
            )
            matches.append(
                {
                    "date": base + timedelta(days=i * 7 + 3),
                    "home_team": "B",
                    "away_team": "A",
                    "home_goals": 1,
                    "away_goals": 0,
                }
            )
        return matches

    def test_fit_creates_team_ratings(self, simple_matches):
        """After fitting, all teams should have attack/defense parameters."""
        dc = DixonColes(xi=0.01)
        dc.fit(simple_matches)
        assert set(dc.teams) == {"A", "B"}
        assert "A" in dc.attack
        assert "B" in dc.defense

    def test_fit_attack_sum_zero(self, simple_matches):
        """Post-fit re-centering should make sum of attacks ~= 0."""
        dc = DixonColes(xi=0.01)
        dc.fit(simple_matches)
        attack_values = np.array(list(dc.attack.values()))
        assert attack_values.sum() == pytest.approx(0.0, abs=1e-10)

    def test_expected_goals_home_advantage(self, balanced_matches):
        """Home team should have higher expected goals on average."""
        dc = DixonColes(xi=0.01)
        dc.fit(balanced_matches)
        lam, mu = dc.expected_goals("A", "B")
        assert lam > mu  # home advantage + A is slightly stronger

    def test_expected_goals_unknown_team_raises(self, simple_matches):
        """Asking for an unknown team should raise ValueError."""
        dc = DixonColes(xi=0.01)
        dc.fit(simple_matches)
        with pytest.raises(ValueError, match="Unknown team"):
            dc.expected_goals("A", "Unknown")

    def test_score_matrix_sum_to_one(self, simple_matches):
        """The score matrix should be a valid probability distribution."""
        dc = DixonColes(xi=0.01)
        dc.fit(simple_matches)
        mat = dc.score_matrix("A", "B")
        assert mat.sum() == pytest.approx(1.0, abs=1e-6)

    def test_score_matrix_shape(self, simple_matches):
        """Default max_goals=10 gives an 11x11 matrix."""
        dc = DixonColes(xi=0.01)
        dc.fit(simple_matches)
        mat = dc.score_matrix("A", "B")
        assert mat.shape == (11, 11)

    def test_match_probabilities_valid(self, simple_matches):
        """Home/Draw/Away probabilities should sum to 1 and be in [0,1]."""
        dc = DixonColes(xi=0.01)
        dc.fit(simple_matches)
        p = dc.match_probabilities("A", "B")
        total = p["home_win"] + p["draw"] + p["away_win"]
        assert total == pytest.approx(1.0, abs=1e-6)
        assert 0 <= p["home_win"] <= 1
        assert 0 <= p["draw"] <= 1
        assert 0 <= p["away_win"] <= 1

    def test_match_probabilities_over_under_btts(self, simple_matches):
        """Over/under and BTTS probabilities should be consistent."""
        dc = DixonColes(xi=0.01)
        dc.fit(simple_matches)
        p = dc.match_probabilities("A", "B")
        assert p["over_2_5"] + p["under_2_5"] == pytest.approx(1.0, abs=1e-6)
        assert p["btts_yes"] + p["btts_no"] == pytest.approx(1.0, abs=1e-6)

    def test_empty_matches_raises(self):
        """Fitting with no matches should raise ValueError."""
        dc = DixonColes()
        with pytest.raises(ValueError):
            dc.fit([])

    def test_convergence_flag(self, simple_matches):
        """The fit should report convergence."""
        dc = DixonColes(xi=0.01)
        dc.fit(simple_matches)
        assert dc.converged

    def test_weights_decay_with_time(self):
        """Older matches should receive lower weights."""
        dc = DixonColes(xi=0.005)
        ref = datetime(2024, 6, 1)
        dates = [datetime(2024, 1, 1), datetime(2024, 3, 1), datetime(2024, 5, 1)]
        w = dc._weights(dates, ref)
        assert w[0] < w[1] < w[2]  # older = less weight

    def test_same_teams_symmetric(self):
        """A vs B and B vs A should satisfy the home-advantage relationship."""
        base = datetime(2024, 1, 1)
        matches = [
            {
                "date": base + timedelta(days=i * 7),
                "home_team": "A",
                "away_team": "B",
                "home_goals": 2,
                "away_goals": 1,
            }
            for i in range(5)
        ] + [
            {
                "date": base + timedelta(days=i * 7 + 3),
                "home_team": "B",
                "away_team": "A",
                "home_goals": 1,
                "away_goals": 2,
            }
            for i in range(5)
        ]
        dc = DixonColes(xi=0.01)
        dc.fit(matches)
        lam_ab, mu_ab = dc.expected_goals("A", "B")
        lam_ba, mu_ba = dc.expected_goals("B", "A")
        # Home advantage means lam_ab > mu_ba (home team's expected goals > same team's
        # expected goals when away) and vice versa. The relationship is:
        # lam_ab = exp(atk_A - def_B + home_adv) vs mu_ba = exp(atk_A - def_B)
        # So home_adv should account for the ratio.
        adv_ratio = lam_ab / mu_ba if mu_ba > 0 else 1.0
        adv_ratio_ba = lam_ba / mu_ab if mu_ab > 0 else 1.0
        assert adv_ratio == pytest.approx(np.exp(dc.home_adv), abs=0.1)
        assert adv_ratio_ba == pytest.approx(np.exp(dc.home_adv), abs=0.1)

    def test_home_win_prob_greater_than_away(self, balanced_matches):
        """Home team should be favored due to home advantage."""
        dc = DixonColes(xi=0.01)
        dc.fit(balanced_matches)
        p = dc.match_probabilities("A", "B")
        assert p["home_win"] > p["away_win"]
