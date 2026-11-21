"""Tests for the configurable meta-learner (Section C.9): gradient-boosted tree
and isotonic/Platt-calibrated variants of the stack."""

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src.ensemble import FootballEnsemble


@pytest.fixture
def demo_data():
    from src.data_loader import generate_synthetic_league

    return generate_synthetic_league(n_teams=8, n_seasons=2)


def _assert_predictions_valid(ensemble):
    out = ensemble.predict("Team A", "Team B")
    probs = [out["home_win"], out["draw"], out["away_win"]]
    assert all(np.isfinite(p) for p in probs)
    assert sum(probs) == pytest.approx(1.0, abs=1e-6)


class TestMetaLearnerVariants:
    def test_gbm_meta_learner(self, demo_data):
        ens = FootballEnsemble(meta_learner="gbm", meta_gbm_max_iter=50)
        ens.fit(demo_data)
        assert isinstance(ens.meta, HistGradientBoostingClassifier)
        _assert_predictions_valid(ens)

    def test_logistic_with_platt_calibration(self, demo_data):
        ens = FootballEnsemble(meta_calibration="sigmoid")
        ens.fit(demo_data)
        assert isinstance(ens.meta, CalibratedClassifierCV)
        _assert_predictions_valid(ens)

    def test_gbm_with_isotonic_calibration(self, demo_data):
        ens = FootballEnsemble(
            meta_learner="gbm", meta_calibration="isotonic", meta_gbm_max_iter=50
        )
        ens.fit(demo_data)
        assert isinstance(ens.meta, CalibratedClassifierCV)
        _assert_predictions_valid(ens)

    def test_default_stays_logistic_uncalibrated(self):
        ens = FootballEnsemble()
        assert isinstance(ens.meta, LogisticRegression)
        assert ens.meta_learner == "logistic"
        assert ens.meta_calibration is None
