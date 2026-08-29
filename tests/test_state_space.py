"""
Tests for the dynamic state-space (Kalman-filtered) strength model.

The key property we verify beyond basic sanity is LEAK-FREENESS: the filtered
probability for match i must depend only on matches strictly before it, so a
model's in-sample features can never contain the outcome they predict.
"""

import numpy as np
import pytest

from src.data_loader import generate_synthetic_league
from src.state_space import StateSpaceModel


@pytest.fixture
def matches():
    df = generate_synthetic_league(n_teams=8, n_seasons=2)
    return df.to_dict("records")


def _strength_matches(n_games=20, seed=7):
    """A is genuinely stronger than B regardless of venue."""
    rng = np.random.default_rng(seed)
    base = np.datetime64("2024-01-01")
    rows = []
    for i in range(n_games):
        date = base + np.timedelta64(i * 7, "D")
        home, away = ("A", "B") if i % 2 == 0 else ("B", "A")
        if home == "A":
            lam, mu = 2.5, 0.6
        else:
            lam, mu = 1.0, 2.0  # B at home still loses the xG battle to A
        hg, ag = int(rng.poisson(lam)), int(rng.poisson(mu))
        rows.append(
            {
                "date": date,
                "home_team": home,
                "away_team": away,
                "home_goals": hg,
                "away_goals": ag,
            }
        )
    return rows


class TestFiltering:
    def test_returns_probs_array(self, matches):
        model = StateSpaceModel(q=0.01)
        probs = model.filter_matches(matches)
        assert probs.shape == (len(matches), 3)

    def test_probs_sum_to_one(self, matches):
        model = StateSpaceModel(q=0.01)
        probs = model.filter_matches(matches)
        sums = probs.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-9)

    def test_probs_in_range(self, matches):
        model = StateSpaceModel(q=0.01)
        probs = model.filter_matches(matches)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_handles_unknown_team_gracefully(self, matches):
        """Unknown teams (newly promoted, no history) get a league-mean prior
        prediction instead of raising."""
        model = StateSpaceModel(q=0.01)
        model.filter_matches(matches)
        lam, mu = model.expected_goals("Team A", "Team NOT REAL")
        assert lam > 0 and mu > 0
        probs = model.match_probabilities("Team NOT REAL", "Team A")
        assert 0 <= probs["home_win"] <= 1

    def test_empty_matches_raises(self):
        with pytest.raises(ValueError):
            StateSpaceModel().filter_matches([])

    def test_recovers_known_strength(self):
        """A dominates B -> after filtering, A's overall quality is higher.

        Note: for this parametrisation the identified quantity is
        quality_i = attack_i + defense_i (attack-minus-defense is a gauge
        direction that never appears in any match rate).
        """
        model = StateSpaceModel(q=0.08, prior_var=0.5, mu=0.5, home_adv=0.25)
        model.filter_matches(_strength_matches())
        a_atk = model.x[model._idx["A"]]
        b_atk = model.x[model._idx["B"]]
        a_q = a_atk + model.x[model.n + model._idx["A"]]
        b_q = b_atk + model.x[model.n + model._idx["B"]]
        assert a_atk > b_atk
        assert a_q > b_q + 0.05

    def test_leak_free_filtered_probs(self, matches):
        """The filtered probability for match k must not depend on match k's
        own outcome, nor on any later match — only strictly prior matches.
        Scale parameters are fixed so the comparison isolates the state path."""
        kw = dict(mu=0.5, home_adv=0.25)
        model_full = StateSpaceModel(q=0.01, **kw)
        probs_full = model_full.filter_matches(matches)

        k = len(matches) - 5
        model_prefix = StateSpaceModel(q=0.01, **kw)
        probs_prefix = model_prefix.filter_matches(matches[:k])
        assert np.allclose(probs_full[0], probs_prefix[0], atol=1e-12)
        assert np.allclose(probs_full[k - 1], probs_prefix[k - 1], atol=1e-12)

    def test_first_match_uses_prior_only(self):
        """Perturbing a match's own result must not change its filtered prob."""
        kw = dict(mu=0.5, home_adv=0.25)
        m = _strength_matches(6)
        model = StateSpaceModel(q=0.01, **kw)
        p1 = model.filter_matches(m)
        m2 = [dict(mm) for mm in m]
        m2[2]["home_goals"], m2[2]["away_goals"] = 0, 9
        model2 = StateSpaceModel(q=0.01, **kw)
        p2 = model2.filter_matches(m2)
        assert np.allclose(p1[:3], p2[:3], atol=1e-12)
        assert not np.allclose(p1[3:], p2[3:], atol=1e-9)

    def test_filters_on_xg_when_requested(self, matches):
        m = [dict(mm) for mm in matches]
        for row in m:
            row["home_xg"] = max(row["home_goals"], 0.3)
            row["away_xg"] = max(row["away_goals"], 0.3)
        model = StateSpaceModel(q=0.005, obs_var_scale=0.5)
        probs = model.filter_matches(m, use_xg=True)
        assert probs.shape == (len(m), 3)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-9)

    def test_nan_xg_observations_fall_back_to_goals(self, matches):
        """Current-season rows joined without xG keep NaN in the xG columns;
        the filter must fall back to the goals observation instead of
        NaN-poisoning the whole state vector (regression: new-season rows
        made the xG filter emit all-NaN probabilities)."""
        m = [dict(mm) for mm in matches]
        for i, row in enumerate(m):
            row["home_xg"] = max(row["home_goals"], 0.3)
            row["away_xg"] = max(row["away_goals"], 0.3)
        # xG columns exist (joined) but are NaN for some matches
        m[0]["home_xg"] = m[0]["away_xg"] = float("nan")
        m[3]["home_xg"] = m[3]["away_xg"] = np.nan
        m[6]["home_xg"] = m[6]["away_xg"] = None

        model = StateSpaceModel(q=0.005, obs_var_scale=0.5)
        probs = model.filter_matches(m, use_xg=True)
        assert np.isfinite(model.x).all()
        assert np.isfinite(probs).all()
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-9)

    def test_missing_goals_rows_skip_update(self):
        """Unplayed fixture rows (blank/NaN scores) must not corrupt the filter."""
        m = _strength_matches(8)
        m[2]["home_goals"] = m[2]["away_goals"] = float("nan")
        model = StateSpaceModel(q=0.01)
        probs = model.filter_matches(m)
        assert np.isfinite(model.x).all()
        assert np.isfinite(probs).all()
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-9)


class TestPrediction:
    def test_expected_goals_positive(self, matches):
        model = StateSpaceModel(q=0.01)
        model.filter_matches(matches)
        lam, mu = model.expected_goals("Team A", "Team B")
        assert lam > 0 and mu > 0

    def test_match_probabilities_valid(self, matches):
        model = StateSpaceModel(q=0.01)
        model.filter_matches(matches)
        p = model.match_probabilities("Team A", "Team B")
        total = p["home_win"] + p["draw"] + p["away_win"]
        assert total == pytest.approx(1.0, abs=1e-6)
        assert 0 <= p["home_win"] <= 1
        assert p["over_2_5"] + (1 - p["over_2_5"]) == pytest.approx(1.0, abs=1e-6)
        assert p["btts_yes"] + (1 - p["btts_yes"]) == pytest.approx(1.0, abs=1e-6)

    def test_home_advantage_respected(self, matches):
        model = StateSpaceModel(q=0.01)
        model.filter_matches(matches)
        p_ab = model.match_probabilities("Team A", "Team B")
        assert p_ab["home_win"] >= p_ab["away_win"]

    def test_requires_fit_first(self, matches):
        model = StateSpaceModel(q=0.01)
        with pytest.raises(ValueError):
            model.expected_goals("Team A", "Team B")
