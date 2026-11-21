"""Tests for derivative market pricing (Section C.8): Over/Under totals and
Asian Handicap priced straight off a Dixon-Coles score matrix."""

import numpy as np
import pytest

from src.market_lines import (
    DEFAULT_AH_LINES,
    DEFAULT_OU_LINES,
    asian_handicap_prices,
    market_lines_from_matrix,
    over_under_prices,
    total_goals_distribution,
)


def _square_matrix() -> np.ndarray:
    """A 4x4 score matrix with a strong diagonal (draw-heavy fixture)."""
    mat = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            mat[i, j] = 1.0 + 3.0 * abs(i - j) * (0 if i == j else 1)
            if i == j:
                mat[i, j] = 4.0  # draws most likely
    mat /= mat.sum()
    return mat


class TestOverUnder:
    def test_total_distribution_sums_to_one(self):
        dist = total_goals_distribution(_square_matrix())
        assert sum(dist.values()) == pytest.approx(1.0)

    def test_over_under_are_complements(self):
        for row in over_under_prices(_square_matrix()):
            assert row["over"] + row["under"] == pytest.approx(1.0, abs=1e-6)

    def test_more_goals_needed_makes_over_harder(self):
        prices = over_under_prices(_square_matrix())
        p25 = next(p for p in prices if p["line"] == 2.5)
        p45 = next(p for p in prices if p["line"] == 4.5)
        assert p45["over"] < p25["over"]

    def test_odds_are_inverse_of_probability(self):
        for row in over_under_prices(_square_matrix()):
            if row["over_odds"] is not None:
                assert row["over_odds"] * row["over"] == pytest.approx(1.0, rel=0.02)


class TestAsianHandicap:
    def test_level_ball_push_equals_draw_mass(self):
        mat = _square_matrix()
        prices = asian_handicap_prices(mat, handicaps=[0.0])
        row = prices[0]
        # A level ball: home covers a straight win, push = the draw, away wins.
        assert row["push"] == pytest.approx(float(np.trace(mat)), abs=1e-3)
        assert row["home"] + row["push"] + row["away"] == pytest.approx(1.0)

    def test_quarter_line_averages_adjacent_half_lines(self):
        mat = _square_matrix()
        row = next(p for p in asian_handicap_prices(mat) if p["handicap"] == -0.25)
        half_0 = next(p for p in asian_handicap_prices(mat, handicaps=[0.0]))
        half_05 = next(p for p in asian_handicap_prices(mat, handicaps=[-0.5]))
        expected_home = 0.5 * (half_0["home"] + half_05["home"])
        assert row["home"] == pytest.approx(expected_home, abs=1e-6)

    def test_probabilities_sum_to_one_including_push(self):
        for row in asian_handicap_prices(_square_matrix()):
            assert row["home"] + row["push"] + row["away"] == pytest.approx(
                1.0, abs=1e-6
            )


class TestBundle:
    def test_bundle_contains_both_markets(self):
        mat = _square_matrix()
        bundle = market_lines_from_matrix(mat)
        assert len(bundle["over_under"]) == len(DEFAULT_OU_LINES)
        assert len(bundle["asian_handicap"]) == len(DEFAULT_AH_LINES)
        assert bundle["asian_handicap"][0]["handicap"] == DEFAULT_AH_LINES[0]
