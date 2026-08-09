"""
Model factory — builds an ensemble from config.yaml so scripts stay thin
and every hyperparameter lives in one place.

Usage:
    from src.config import load_config
    from src.model_factory import build_ensemble
    model = build_ensemble(load_config())
"""

from __future__ import annotations

from types import SimpleNamespace

from src.ensemble import FootballEnsemble


def _ns_or_default(cfg, section: str, default: dict) -> SimpleNamespace:
    if cfg is not None and hasattr(cfg, section):
        return getattr(cfg, section)
    return SimpleNamespace(**default)


def build_ensemble(cfg=None) -> FootballEnsemble:
    """Construct a FootballEnsemble using hyperparameters from config.yaml
    (falls back to hardcoded defaults when no config is present)."""
    model_cfg = _ns_or_default(
        cfg,
        "model",
        {
            "dc_xi": 0.0018,
            "elo_k": 20.0,
            "elo_home_advantage": 75.0,
            "elo_initial_rating": 1500.0,
            "elo_goal_diff_multiplier": True,
            "elo_draw_width": 0.44,
            "meta_max_iter": 2000,
            "meta_C": 1.0,
        },
    )

    elo_kwargs = {
        "k": getattr(model_cfg, "elo_k", 20.0),
        "home_advantage": getattr(model_cfg, "elo_home_advantage", 75.0),
        "initial_rating": getattr(model_cfg, "elo_initial_rating", 1500.0),
        "goal_diff_multiplier": getattr(model_cfg, "elo_goal_diff_multiplier", True),
        "draw_width": getattr(model_cfg, "elo_draw_width", 0.44),
    }
    dc_kwargs = {"xi": getattr(model_cfg, "dc_xi", 0.0018)}

    return FootballEnsemble(
        elo_kwargs=elo_kwargs,
        dc_kwargs=dc_kwargs,
        meta_max_iter=getattr(model_cfg, "meta_max_iter", 2000),
        meta_C=getattr(model_cfg, "meta_C", 1.0),
    )