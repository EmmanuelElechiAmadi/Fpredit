"""
Football match prediction engine.

Combines:
  - Dixon-Coles bivariate Poisson model (statistical baseline)
  - Elo ratings (fast-reacting team strength)
  - Rolling form + head-to-head features
  - Logistic regression meta-learner stacking them all

Usage:
    from src.ensemble import FootballEnsemble
    from src.data_loader import load_league_csvs, generate_synthetic_league

    df = load_league_csvs("data/raw", "EPL")
    model = FootballEnsemble().fit(df)
    result = model.predict("Arsenal", "Chelsea")
    print(result["home_win"])
"""

from .config import load_config
from .data_loader import generate_synthetic_league, load_league_csvs
from .dixon_coles import DixonColes
from .elo import EloEngine
from .ensemble import FootballEnsemble
from .features import head_to_head, rolling_form
from .split import train_val_test_split, walk_forward_windows

__all__ = [
    "FootballEnsemble",
    "DixonColes",
    "EloEngine",
    "rolling_form",
    "head_to_head",
    "load_league_csvs",
    "generate_synthetic_league",
    "train_val_test_split",
    "walk_forward_windows",
    "load_config",
]
