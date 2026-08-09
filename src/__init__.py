"""
Football match prediction engine.

Combines:
  - Dynamic state-space (Kalman-filtered) team strength
  - Dixon-Coles bivariate Poisson model (static baseline / diagnostics)
  - Elo ratings (fast-reacting team strength)
  - Rolling form, congestion, league position, PageRank + H2H features
  - Market implied probabilities (residual-vs-market meta-model)
  - Optional xG features (Understat)
  - Logistic regression meta-learner stacking everything
  - Covariance-adjusted fractional Kelly staking

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
from .features import (
    fixture_congestion,
    head_to_head,
    league_position,
    pagerank_strength,
    rolling_form,
)
from .market import (
    add_implied_probabilities,
    add_market_features,
    implied_probabilities,
    market_comparison,
    value_bets,
)
from .split import train_val_test_split, walk_forward_windows
from .staking import covariance_adjusted_stakes, kelly_fraction, portfolio_report
from .state_space import StateSpaceModel
from .xg_loader import fetch_league_xg, generate_synthetic_xg, join_xg, load_league_xg

__all__ = [
    "FootballEnsemble",
    "DixonColes",
    "EloEngine",
    "StateSpaceModel",
    "rolling_form",
    "head_to_head",
    "fixture_congestion",
    "league_position",
    "pagerank_strength",
    "load_league_csvs",
    "generate_synthetic_league",
    "train_val_test_split",
    "walk_forward_windows",
    "load_config",
    "add_implied_probabilities",
    "add_market_features",
    "implied_probabilities",
    "market_comparison",
    "value_bets",
    "covariance_adjusted_stakes",
    "kelly_fraction",
    "portfolio_report",
    "fetch_league_xg",
    "load_league_xg",
    "join_xg",
    "generate_synthetic_xg",
]
