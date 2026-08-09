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
            "dc_shrinkage": 0.0,
            "elo_k": 20.0,
            "elo_home_advantage": 75.0,
            "elo_initial_rating": 1500.0,
            "elo_goal_diff_multiplier": True,
            "elo_draw_width": 0.44,
            "ss_q": 0.01,
            "ss_q_xg": 0.005,
            "ss_prior_var": 0.25,
            "ss_obs_var_scale_xg": 0.5,
            "meta_max_iter": 2000,
            "meta_C": 1.0,
            "use_market_features": True,
            "form_window": 5,
            "h2h_lookback": 5,
            "congestion_days": 8,
            "load_days": 14,
            "position_reset_days": 100,
            "kelly_fraction": 0.25,
            "kelly_max_stake": 0.10,
            "kelly_cov_shrinkage": 0.9,
            "kelly_corr": 0.05,
        },
    )

    elo_kwargs = {
        "k": getattr(model_cfg, "elo_k", 20.0),
        "home_advantage": getattr(model_cfg, "elo_home_advantage", 75.0),
        "initial_rating": getattr(model_cfg, "elo_initial_rating", 1500.0),
        "goal_diff_multiplier": getattr(model_cfg, "elo_goal_diff_multiplier", True),
        "draw_width": getattr(model_cfg, "elo_draw_width", 0.44),
    }
    dc_kwargs = {
        "xi": getattr(model_cfg, "dc_xi", 0.0018),
        "shrinkage": getattr(model_cfg, "dc_shrinkage", 0.0),
    }

    return FootballEnsemble(
        elo_kwargs=elo_kwargs,
        dc_kwargs=dc_kwargs,
        meta_max_iter=getattr(model_cfg, "meta_max_iter", 2000),
        meta_C=getattr(model_cfg, "meta_C", 1.0),
        ss_q=getattr(model_cfg, "ss_q", 0.01),
        ss_q_xg=getattr(model_cfg, "ss_q_xg", 0.005),
        ss_prior_var=getattr(model_cfg, "ss_prior_var", 0.25),
        ss_obs_var_scale_xg=getattr(model_cfg, "ss_obs_var_scale_xg", 0.5),
        use_market_features=getattr(model_cfg, "use_market_features", True),
        form_window=getattr(model_cfg, "form_window", 5),
        h2h_lookback=getattr(model_cfg, "h2h_lookback", 5),
        congestion_days=getattr(model_cfg, "congestion_days", 8),
        load_days=getattr(model_cfg, "load_days", 14),
        position_reset_days=getattr(model_cfg, "position_reset_days", 100),
    )
