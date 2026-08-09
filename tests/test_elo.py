"""
Unit tests for the Elo rating engine.
"""

from datetime import datetime, timedelta

import pytest

from src.elo import EloEngine


def make_match(date, home, away, hg, ag):
    return {
        "date": date,
        "home_team": home,
        "away_team": away,
        "home_goals": hg,
        "away_goals": ag,
    }


class TestEloEngine:
    @pytest.fixture
    def elo(self):
        return EloEngine(k=20.0, home_advantage=75.0)

    def test_initial_rating_default(self, elo):
        """Unknown teams start at initial_rating."""
        assert elo.get("Unknown Team") == 1500.0

    def test_expected_score_home_advantage(self, elo):
        """Home team should have >0.5 expected score against equal opponent."""
        # Both teams at 1500, but home gets +75
        exp = elo.expected_score("Team A", "Team B")
        assert exp > 0.5

    def test_expected_score_symmetric(self, elo):
        """Both teams get home advantage when called, so both > 0.5 on neutral ground."""
        exp_home = elo.expected_score("A", "B")
        exp_away = elo.expected_score("B", "A")
        # Both equal because ratings are equal, and each gets +home_advantage
        assert exp_home == pytest.approx(exp_away, abs=1e-6)
        assert exp_home > 0.5
        assert exp_away > 0.5

    def test_home_win_increases_rating(self, elo):
        """Home win should increase home rating and decrease away rating."""
        elo.ratings["A"] = 1500.0
        elo.ratings["B"] = 1500.0
        elo.update("2024-01-01", "A", "B", 2, 0)
        assert elo.ratings["A"] > 1500.0
        assert elo.ratings["B"] < 1500.0

    def test_away_win_increases_away_rating(self, elo):
        """Away win should increase away rating more than home loss."""
        elo.ratings["A"] = 1500.0
        elo.ratings["B"] = 1500.0
        elo.update("2024-01-01", "A", "B", 0, 2)
        assert elo.ratings["A"] < 1500.0
        assert elo.ratings["B"] > 1500.0

    def test_draw_updates_ratings(self, elo):
        """Draw should move both ratings toward each other (slightly)."""
        elo.ratings["A"] = 1600.0  # stronger
        elo.ratings["B"] = 1400.0  # weaker
        elo.update("2024-01-01", "A", "B", 1, 1)
        # Stronger team loses rating, weaker gains (draw was better than expected for B)
        assert elo.ratings["A"] < 1600.0
        assert elo.ratings["B"] > 1400.0

    def test_margin_multiplier(self, elo):
        """Larger win margins produce larger rating changes."""
        elo.ratings["A"] = 1500.0
        elo.ratings["B"] = 1500.0
        elo.update("2024-01-01", "A", "B", 5, 0)
        big_margin_delta = elo.ratings["A"] - 1500.0

        elo2 = EloEngine(k=20.0, home_advantage=75.0)
        elo2.ratings["A"] = 1500.0
        elo2.ratings["B"] = 1500.0
        elo2.update("2024-01-01", "A", "B", 1, 0)
        small_margin_delta = elo2.ratings["A"] - 1500.0

        assert abs(big_margin_delta) > abs(small_margin_delta)

    def test_margin_multiplier_disabled(self):
        """When goal_diff_multiplier=False, margin should not affect delta."""
        elo = EloEngine(k=20.0, home_advantage=75.0, goal_diff_multiplier=False)
        elo.ratings["A"] = 1500.0
        elo.ratings["B"] = 1500.0
        elo.update("2024-01-01", "A", "B", 5, 0)
        delta_5 = elo.ratings["A"] - 1500.0

        elo2 = EloEngine(k=20.0, home_advantage=75.0, goal_diff_multiplier=False)
        elo2.ratings["A"] = 1500.0
        elo2.ratings["B"] = 1500.0
        elo2.update("2024-01-01", "A", "B", 1, 0)
        delta_1 = elo2.ratings["A"] - 1500.0

        assert delta_5 == pytest.approx(delta_1)

    def test_fit_processes_all_matches(self, elo):
        """fit() should update ratings for all matches sequentially."""
        base = datetime(2024, 1, 1)
        matches = [
            make_match(
                base + timedelta(days=i * 7),
                f"Team {chr(65+i)}",
                f"Team {chr(66+i)}",
                2,
                1,
            )
            for i in range(3)
        ]
        elo.fit(matches)
        assert len(elo.history) == 3

    def test_history_records_pre_match_ratings(self, elo):
        """Each update should append to history with pre-match ratings."""
        elo.ratings["A"] = 1500.0
        elo.ratings["B"] = 1500.0
        elo.update("2024-01-01", "A", "B", 2, 0)
        assert len(elo.history) == 1
        date, home, away, rh, ra = elo.history[0]
        assert rh == 1500.0
        assert ra == 1500.0

    def test_win_draw_loss_probs_sum_to_one(self, elo):
        """The three probabilities should always sum to 1."""
        elo.ratings["A"] = 1500.0
        elo.ratings["B"] = 1500.0
        ph, pd_, pa = elo.win_draw_loss_prob("A", "B")
        assert ph + pd_ + pa == pytest.approx(1.0, abs=1e-6)

    def test_win_draw_loss_home_favored_equal_teams(self, elo):
        """With equal ratings but home advantage, home should be favored."""
        elo.ratings["A"] = 1500.0
        elo.ratings["B"] = 1500.0
        ph, pd_, pa = elo.win_draw_loss_prob("A", "B")
        assert ph > pa

    def test_strong_away_team_can_be_favored(self, elo):
        """A much stronger away team can overcome home advantage."""
        elo.ratings["A"] = 1300.0
        elo.ratings["B"] = 1700.0
        ph, pd_, pa = elo.win_draw_loss_prob("A", "B")
        assert pa > ph

    def test_draw_probability_highest_when_even(self, elo):
        """Draw probability should be higher for evenly matched teams."""
        elo.ratings["A"] = 1500.0
        elo.ratings["B"] = 1500.0
        _, pd_even, _ = elo.win_draw_loss_prob("A", "B")

        elo.ratings["A"] = 1800.0
        elo.ratings["B"] = 1200.0
        _, pd_uneven, _ = elo.win_draw_loss_prob("A", "B")
        assert pd_even > pd_uneven

    def test_k_higher_more_reactive(self):
        """Higher K should produce larger rating changes."""
        elo_high = EloEngine(k=40.0, home_advantage=75.0)
        elo_low = EloEngine(k=10.0, home_advantage=75.0)

        for elo in [elo_high, elo_low]:
            elo.ratings["A"] = 1500.0
            elo.ratings["B"] = 1500.0
            elo.update("2024-01-01", "A", "B", 2, 0)

        assert abs(elo_high.ratings["A"] - 1500) > abs(elo_low.ratings["A"] - 1500)

    def test_historical_match_preserves_pre_ratings(self, elo):
        """Updating a match should preserve pre-match ratings in history."""
        elo.ratings["A"] = 1500.0
        elo.ratings["B"] = 1500.0
        pre_a = elo.get("A")
        pre_b = elo.get("B")
        elo.update("2024-01-01", "A", "B", 2, 0)
        assert elo.history[0][3] == pre_a
        assert elo.history[0][4] == pre_b
