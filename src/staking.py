"""
Covariance-adjusted fractional Kelly staking.

A full weekend slate of matches is not a set of independent bets: form,
referee-assignment clusters and weather are correlated within a matchday.
Sizing each bet independently with Kelly overstates your effective
diversification. This module treats the slate as a portfolio:

    w = inv(Sigma) * mu

(mean-variance approximation to log-utility / Kelly portfolio), where mu is
the per-bet expected excess return and Sigma is the covariance of the per-bet
Bernoulli returns: diagonal variance p_i(1-p_i)*odds_i^2 plus a constant
cross-match correlation rho. The empirical cross-match correlation is barely
estimable from ~1 slate per week, so the covariance is shrunk between the
full correlation structure and the independent-bets model.

Metrics reported are the portfolio-native ones used in trading: Sharpe,
maximum drawdown and CVaR (expected shortfall), plus plain ROI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_SEASONS_PER_YEAR = 38.0  # ~38 matchdays per season, used for annualizing


def kelly_fraction(p: float, odds: float) -> float:
    """Full-Kelly fraction for a single binary bet: f* = (p*odds - 1)/(odds - 1)."""
    b = odds - 1.0
    if b <= 0:
        return 0.0
    return max((p * odds - 1.0) / b, 0.0)


def fractional_kelly_stake(
    p: float, odds: float, fraction: float = 0.25, max_stake: float = 0.10
) -> float:
    """Fractional-Kelly stake as a fraction of bankroll, capped at max_stake."""
    return min(kelly_fraction(p, odds) * fraction, max_stake)


def covariance_adjusted_stakes(
    bets: pd.DataFrame,
    kelly_fraction: float = 0.25,
    max_stake: float = 0.10,
    corr: float = 0.05,
    cov_shrinkage: float = 0.9,
) -> pd.Series:
    """Portfolio Kelly sizing over a slate of bets.

    Parameters
    ----------
    bets : pd.DataFrame
        One row per bet with at least `model_prob` and `odds`.
    kelly_fraction : float
        Fraction of the portfolio-Kelly vector to actually stake.
    max_stake : float
        Per-bet cap as a fraction of bankroll.
    corr : float
        Structural cross-match outcome correlation assumption.
    cov_shrinkage : float
        Weight on the correlated (full-rho) covariance vs the independent
        model. 1.0 = full correlation structure, 0.0 = independent bets.

    Returns
    -------
    pd.Series of stakes (0 for non-positive-edge rows).
    """
    if bets.empty:
        return pd.Series(dtype=float)

    p = bets["model_prob"].to_numpy(dtype=float)
    odds = bets["odds"].to_numpy(dtype=float)
    n = len(p)

    # Expected net return per unit staked (the edge): p*odds - 1
    mu = np.maximum(p * odds - 1.0, 0.0)

    # Single-bet case reduces to scalar fractional Kelly
    if n == 1:
        b = odds[0] - 1.0
        full = max((p[0] * odds[0] - 1.0) / b, 0.0) if b > 0 else 0.0
        return pd.Series([min(full * kelly_fraction, max_stake)], index=bets.index)

    # Structural covariance: Bernoulli variance scaled by odds, plus a single
    # cross-match correlation rho.
    var = p * (1.0 - p) * odds**2
    std = np.sqrt(var)
    s_corr = np.outer(std, std) * corr
    np.fill_diagonal(s_corr, var)
    s_indep = np.diag(var)
    sigma = cov_shrinkage * s_corr + (1.0 - cov_shrinkage) * s_indep

    # Ridge-regularize the covariance. A full weekend slate of correlated bets
    # is near-singular (0.05 correlation over hundreds of matches), and solving
    # the raw matrix produces degenerate alternating weights that collapse to a
    # handful of huge long positions once negative legs are clipped to zero.
    # Adding ~1% of the median variance to the diagonal keeps the solve stable
    # while leaving the correlation structure intact.
    sigma = sigma + np.eye(n) * max(1e-6, 0.01 * float(np.median(var)))

    # Portfolio Kelly (log-utility): w = inv(Sigma) mu, then clip and scale.
    try:
        w = np.linalg.solve(sigma + np.eye(n) * 1e-9, mu)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(sigma) @ mu
    w = np.clip(w, 0.0, None)
    total = w.sum()
    if total > 0:
        w = w / total * kelly_fraction
    w = np.clip(w, 0.0, max_stake)
    return pd.Series(w, index=bets.index)


def portfolio_report(pnl: pd.Series, stakes: pd.Series) -> dict:
    """Risk metrics for a realized PnL series (historical simulation).

    Parameters
    ----------
    pnl : pd.Series of net P&L per bet (in units of the stake).
    stakes : pd.Series of stakes (bankroll fraction) per bet.

    Returns
    -------
    dict with n, total_staked, profit_units, roi, sharpe, max_drawdown, cvar95.
    """
    pnl = np.asarray(pnl, dtype=float)
    stakes = np.asarray(stakes, dtype=float)
    mask = stakes > 0
    if not mask.any():
        return {
            "n": 0,
            "total_staked": 0.0,
            "profit_units": 0.0,
            "roi": None,
            "sharpe": None,
            "max_drawdown": 0.0,
            "cvar95": None,
        }

    pnl, stakes = pnl[mask], stakes[mask]
    returns = pnl / stakes  # per-bet ROI on staked capital

    total_staked = float(stakes.sum())
    profit = float(pnl.sum())
    roi = profit / total_staked if total_staked > 0 else 0.0

    std = float(returns.std(ddof=1))
    sharpe = (
        float(returns.mean()) / std * np.sqrt(_SEASONS_PER_YEAR) if std > 0 else None
    )

    cum = np.cumsum(pnl)
    max_dd = float(np.min(cum - np.maximum.accumulate(cum))) if len(cum) else 0.0

    alpha = 0.05
    if len(returns) >= 5:
        k = max(1, int(alpha * len(returns)))
        cvar = float(np.mean(np.sort(returns)[:k]))
    else:
        cvar = None

    return {
        "n": int(len(pnl)),
        "total_staked": total_staked,
        "profit_units": profit,
        "roi": roi,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "cvar95": cvar,
    }
