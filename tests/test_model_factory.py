"""
Tests for the model factory — building an ensemble from config.yaml
so hyperparameters live in one place.
"""

from types import SimpleNamespace

from src.ensemble import FootballEnsemble
from src.model_factory import _ns_or_default, build_ensemble


class TestBuildEnsemble:
    def test_returns_ensemble_without_config(self):
        model = build_ensemble()
        assert isinstance(model, FootballEnsemble)
        assert model.elo.k == 20.0
        assert model.dc.xi == 0.0018

    def test_uses_config_values(self):
        cfg = SimpleNamespace(
            model=SimpleNamespace(
                dc_xi=0.0025,
                elo_k=15.0,
                elo_home_advantage=50.0,
                elo_initial_rating=1400.0,
                elo_goal_diff_multiplier=False,
                elo_draw_width=0.5,
                meta_max_iter=1000,
                meta_C=0.8,
            )
        )
        model = build_ensemble(cfg)
        assert model.elo.k == 15.0
        assert model.elo.home_advantage == 50.0
        assert model.elo.initial_rating == 1400.0
        assert model.dc.xi == 0.0025
        assert model.meta.max_iter == 1000
        assert model.meta.C == 0.8

    def test_missing_section_falls_back_to_defaults(self):
        cfg = SimpleNamespace(other=SimpleNamespace(foo=1))
        model = build_ensemble(cfg)
        assert model.elo.k == 20.0
        assert model.dc.xi == 0.0018

    def test_partial_config_uses_defaults_for_missing_keys(self):
        cfg = SimpleNamespace(model=SimpleNamespace(elo_k=30.0))
        model = build_ensemble(cfg)
        assert model.elo.k == 30.0
        # Missing values fall back to defaults
        assert model.elo.home_advantage == 75.0
        assert model.dc.xi == 0.0018


class TestNsOrDefault:
    def test_returns_section_when_present(self):
        cfg = SimpleNamespace(model=SimpleNamespace(a=1))
        ns = _ns_or_default(cfg, "model", {"a": 2, "b": 3})
        assert ns.a == 1
        # _ns_or_default does NOT merge defaults into an existing section;
        # build_ensemble fills missing keys via getattr(..., default).
        assert not hasattr(ns, "b")

    def test_returns_defaults_when_section_missing(self):
        ns = _ns_or_default(None, "model", {"a": 1, "b": 2})
        assert ns.a == 1
        assert ns.b == 2

    def test_returns_defaults_when_cfg_is_none(self):
        ns = _ns_or_default(None, "anything", {"x": 9})
        assert ns.x == 9
