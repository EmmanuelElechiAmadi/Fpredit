"""
Centralized configuration loader.

Reads config.yaml and exposes settings as a simple namespace.
Environment variables override YAML values: FOOTBALL_PREDICTOR__model__elo_k=30.0
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import yaml  # type: ignore[import-untyped]

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

# Sentinel for unset env-var overrides
_UNSET = object()


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base dict."""
    result = dict(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _env_overrides() -> dict:
    """Parse FOOTBALL_PREDICTOR__<section>__<key> env vars into a nested dict."""
    prefix = "FOOTBALL_PREDICTOR__"
    overrides: dict = {}
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        parts = env_key[len(prefix) :].lower().split("__")
        target = overrides
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        # Try to cast to int/float if applicable
        try:
            target[parts[-1]] = int(env_val)
        except ValueError:
            try:
                target[parts[-1]] = float(env_val)
            except ValueError:
                target[parts[-1]] = env_val
    return overrides


def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert dicts to SimpleNamespace for dotted access."""
    ns = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(ns, key, _dict_to_namespace(value))
        elif isinstance(value, list):
            setattr(ns, key, value)
        else:
            setattr(ns, key, value)
    return ns


def load_config(config_path: str | Path | None = None) -> SimpleNamespace:
    """Load config from YAML, merge env overrides, return as SimpleNamespace."""
    path = Path(config_path) if config_path else _CONFIG_PATH

    if not path.exists():
        # Return defaults if no config file
        return _dict_to_namespace(_get_defaults())

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    overrides = _env_overrides()
    merged = _deep_merge(raw, overrides)
    return _dict_to_namespace(merged)


def _get_defaults() -> dict:
    """Hardcoded defaults when no config.yaml is present."""
    return {
        "model": {
            "dc_xi": 0.0018,
            "elo_k": 20.0,
            "elo_home_advantage": 75.0,
            "elo_initial_rating": 1500.0,
            "elo_goal_diff_multiplier": True,
            "elo_draw_width": 0.44,
            "meta_max_iter": 2000,
            "meta_C": 1.0,
            "form_window": 5,
            "h2h_lookback": 5,
        },
        "data": {
            "allowed_leagues": ["EPL", "LALIGA", "SERIEA"],
            "data_dir": "data/raw",
            "cache_dir": "models",
        },
        "backtest": {
            "min_train_matches": 380,
            "step_matches": 190,
        },
    }
