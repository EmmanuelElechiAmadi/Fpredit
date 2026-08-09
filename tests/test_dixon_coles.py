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

    def test_expected_goals_unknown_team_falls_back(self, simple_matches):
        """Asking for an unknown team (newly promoted, no history) should fall
        back to league-mean ratings rather than raising."""
        dc = DixonColes(xi=0.01)
        dc.fit(simple_matches)
        lam, mu = dc.expected_goals("A", "Unknown")
        assert lam > 0 and mu > 0
        probs = dc.match_probabilities("Unknown", "A")
        assert sum(
            [probs["home_win"], probs["draw"], probs["away_win"]]
        ) == pytest.approx(1.0, abs=1e-6)

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


class TestShrinkage:
    def test_shrinkage_pulls_ratings_toward_zero(self):
        """With very little data, ridge shrinkage should pull a team's rating
        closer to the league mean (0) than the unregularized fit."""
        base = datetime(2024, 1, 1)
        # Team A plays 8 matches and dominates; Team B plays only 1 match
        matches = [
            {
                "date": base + timedelta(days=i * 7),
                "home_team": "A",
                "away_team": "B",
                "home_goals": 4,
                "away_goals": 0,
            }
            for i in range(8)
        ]
        dc_full = DixonColes(xi=0.01, shrinkage=0.0)
        dc_full.fit(matches)
        dc_shrunk = DixonColes(xi=0.01, shrinkage=2.0)
        dc_shrunk.fit(matches)
        # The 1-match team's rating is strongly shrunk toward 0
        assert abs(dc_shrunk.attack["B"]) < abs(dc_full.attack["B"])
        assert abs(dc_shrunk.defense["B"]) < abs(dc_full.defense["B"])

    def test_zero_shrinkage_matches_classical(self):
        base = datetime(2024, 1, 1)
        matches = [
            {
                "date": base + timedelta(days=i * 7),
                "home_team": "A",
                "away_team": "B",
                "home_goals": 2,
                "away_goals": 1,
            }
            for i in range(6)
        ]
        dc_a = DixonColes(xi=0.01, shrinkage=0.0).fit(matches)
        dc_b = DixonColes(xi=0.01).fit(matches)  # default = no shrinkage
        assert dc_a.attack["A"] == pytest.approx(dc_b.attack["A"], abs=1e-6)


class TestXGTarget:
    def test_fit_xg_valid(self):
        """Fitting on continuous xG values should not crash and must produce
        valid probabilities (unlike a Poisson pmf, which can't score xG)."""
        base = datetime(2024, 1, 1)
        matches = [
            {
                "date": base + timedelta(days=i * 7),
                "home_team": "A",
                "away_team": "B",
                "home_goals": 2,
                "away_goals": 1,
                "home_xg": 2.1,
                "away_xg": 0.8,
            }
            for i in range(6)
        ]
        dc = DixonColes(xi=0.01)
        dc.fit(matches, target="xg")
        p = dc.match_probabilities("A", "B")
        assert p["home_win"] + p["draw"] + p["away_win"] == pytest.approx(1.0, abs=1e-6)
        assert dc.expected_goals("A", "B")[0] > 0
        assert dc.expected_goals("A", "B")[1] > 0

    def test_requires_xg_columns(self):
        base = datetime(2024, 1, 1)
        matches = [
            {
                "date": base + timedelta(days=i * 7),
                "home_team": "A",
                "away_team": "B",
                "home_goals": 2,
                "away_goals": 1,
            }
            for i in range(4)
        ]
        with pytest.raises(KeyError):
            DixonColes().fit(matches, target="xg")

    def test_invalid_target_raises(self):
        with pytest.raises(ValueError):
            DixonColes().fit([], target="nope")
