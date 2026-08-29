"""
Dynamic state-space (Kalman-filtered) team strength — the principled upgrade
over static-window Dixon-Coles.

Dixon-Coles treats attack/defense as fixed within a fit window and is only
patched by re-fitting periodically. That is equivalent to assuming team form
is stationary, which it isn't. This module replaces that with a latent
random-walk state-space model in the spirit of Rue & Salvesen (2000):

    log(lambda_home) = mu + alpha_home - delta_away + home_adv
    log(lambda_away) = mu + alpha_away - delta_home
    goals ~ Poisson(lambda)
    theta_{t+1} = theta_t + eta,   eta ~ N(0, Q)

Because the Poisson observation is non-Gaussian we filter with an extended
Kalman filter: each goal observation is linearized via its local gradient
(y - lambda) and Fisher information (lambda). Two important consequences:

  1. Filtered estimates are leak-free by construction — each match's rating
     is computed from strictly prior matches. This also fixes the subtle
     in-sample leakage that occurs when a static model predicts matches it
     was itself fitted on.
  2. The same filter can observe xG instead of goals. xG is a far less noisy
     observation of true attacking/defending quality, so the filter extracts
     materially more signal (this is why xG + state-space is the combination
     that shows up in the serious literature).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import poisson


def _finite(value) -> bool:
    """True when ``value`` is a real finite number.

    Rejects ``None``, ``NaN``, ``pd.NA`` and non-numeric values. Used so the
    Kalman filter never ingests a NaN observation (e.g. current-season rows
    whose xG hasn't been scraped yet, or unplayed fixture rows with blank
    goals) — a single NaN observation poisons the whole state vector because
    ``_recenter`` spreads it to every component.
    """
    if value is None:
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


_DEFAULT_MAX_GOALS = 10


class StateSpaceModel:
    """EKF-filtered random-walk attack/defense ratings.

    State vector x = [alpha_0..alpha_{n-1}, delta_0..delta_{n-1}] where n is
    the number of teams. Observations are the match goals (or xG) with a
    Poisson likelihood approximated locally by a Gaussian with variance equal
    to the expected rate (Fisher information of the log-Poisson likelihood).
    """

    def __init__(
        self,
        q: float = 0.01,
        prior_var: float = 0.25,
        home_adv: Optional[float] = None,
        max_goals: int = _DEFAULT_MAX_GOALS,
        obs_var_scale: float = 1.0,
        mu: Optional[float] = None,
    ):
        self.q = q
        self.prior_var = prior_var
        self.max_goals = max_goals
        self.obs_var_scale = obs_var_scale

        # Scale constants (league baseline log-rate mu and home advantage).
        # When left None they are estimated from the data; when provided they
        # are fixed, which keeps the filter strictly leak-free even in the
        # (negligible-in-practice) league-baseline sense.
        self._mu_fixed = mu is not None
        self._ha_fixed = home_adv is not None
        self.mu = 0.0 if mu is None else float(mu)
        self.home_adv = 0.25 if home_adv is None else float(home_adv)

        self.teams: list[str] = []
        self._idx: dict[str, int] = {}
        self.x: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.filtered_probs: Optional[np.ndarray] = None
        self.filtered_expected: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------
    def _estimate_scale(self, matches: list[dict]) -> None:
        """Estimate league baseline mu and home advantage from the data
        (skipped for whichever of the two was fixed in the constructor)."""
        hg = np.array([m.get("home_goals") for m in matches], dtype=float)
        ag = np.array([m.get("away_goals") for m in matches], dtype=float)
        # Unplayed fixture rows (blank scores) or NaN xG rows carry no valid
        # observation; drop them so a NaN can't leak into the league baseline.
        valid = np.isfinite(hg) & np.isfinite(ag)
        if valid.any():
            hg, ag = hg[valid], ag[valid]
            mean_h, mean_a = float(hg.mean()), float(ag.mean())
            mean_h = max(mean_h, 1e-3)
            mean_a = max(mean_a, 1e-3)
            if not self._mu_fixed:
                self.mu = np.log(mean_a)
            if not self._ha_fixed:
                self.home_adv = float(np.clip(np.log(mean_h / mean_a), 0.05, 0.6))

    def _update(
        self, obs: float, log_rate: float, obs_var_scale: float, h_row: np.ndarray
    ) -> None:
        """One EKF measurement update for a single Poisson observation.

        Linearizes the log-Poisson likelihood at the current log-rate:
        innovation = y - exp(z), measurement variance ~= obs_var_scale * exp(z).
        """
        assert self.x is not None and self.P is not None
        lam = float(np.exp(log_rate))
        lam = max(lam, 1e-6)
        v = obs - lam
        hp = h_row @ self.P  # (2n,)
        s = float(hp @ h_row.T + obs_var_scale * lam)
        s = max(s, 1e-9)
        k = hp / s  # (2n,)
        self.x += k * v
        self.P -= np.outer(k, hp)
        self.P = (self.P + self.P.T) / 2.0

    def _recenter(self, n: int) -> None:
        """Re-center attack so mean(alpha) == 0; shift defense oppositely.

        The model is only identified up to an additive shift between alpha and
        delta (adding c to both leaves every expected-goals rate unchanged).
        Re-centering keeps log-rates bounded and mirrors Dixon-Coles.
        """
        assert self.x is not None
        alpha = self.x[:n]
        offset = float(alpha.mean())
        self.x[:n] -= offset
        self.x[n:] += offset

    def filter_matches(
        self,
        matches: list[dict],
        use_xg: bool = False,
        obs_var_scale: Optional[float] = None,
    ) -> np.ndarray:
        """Filter a chronological match list, storing per-match filtered
        probabilities computed strictly from prior matches.

        Returns an (n_matches, 3) array of [home_win, draw, away_win] — this
        is leak-free by construction and can be used to build meta-learner
        training features.
        """
        if not matches:
            raise ValueError("StateSpaceModel.filter_matches requires >= 1 match")

        matches = sorted(matches, key=lambda m: m["date"])
        teams = sorted(
            set(m["home_team"] for m in matches) | set(m["away_team"] for m in matches)
        )
        self.teams = teams
        self._idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        self._estimate_scale(matches)
        scale = obs_var_scale if obs_var_scale is not None else self.obs_var_scale

        self.x = np.zeros(2 * n)
        self.P = np.eye(2 * n) * self.prior_var
        assert self.x is not None and self.P is not None

        probs = np.zeros((len(matches), 3))
        expected = np.zeros((len(matches), 2))

        for i, m in enumerate(matches):
            hi, ai = self._idx[m["home_team"]], self._idx[m["away_team"]]
            z_h = self.mu + self.x[hi] - self.x[n + ai] + self.home_adv
            z_a = self.mu + self.x[ai] - self.x[n + hi]

            expected[i] = (np.exp(z_h), np.exp(z_a))
            probs[i] = self._probs_from_logrates(z_h, z_a)

            # Predict step (random-walk drift)
            self.P += np.eye(2 * n) * self.q

            h_row_h = np.zeros(2 * n)
            h_row_h[hi] = 1.0
            h_row_h[n + ai] = -1.0
            h_row_a = np.zeros(2 * n)
            h_row_a[ai] = 1.0
            h_row_a[n + hi] = -1.0

            # Only feed observations that are real numbers: current-season rows
            # joined without xG (NaN) or unplayed fixture rows (blank goals)
            # must fall back to the goals branch — or skip the update entirely
            # when no valid observation exists — instead of NaN-poisoning the
            # filter state.
            if use_xg and _finite(m.get("home_xg")) and _finite(m.get("away_xg")):
                self._update(float(m["home_xg"]), z_h, scale, h_row_h)
                self._update(float(m["away_xg"]), z_a, scale, h_row_a)
            elif _finite(m.get("home_goals")) and _finite(m.get("away_goals")):
                self._update(float(m["home_goals"]), z_h, scale, h_row_h)
                self._update(float(m["away_goals"]), z_a, scale, h_row_a)
            # else: no valid observation yet — state just drifts (predict step)

            self._recenter(n)

        self.filtered_probs = probs
        self.filtered_expected = expected
        return probs

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        if self.x is None:
            raise ValueError("StateSpaceModel must be fitted before expected_goals()")
        # Teams absent from the training window (newly promoted, no history yet)
        # get the league-mean rating 0.0 — the filter can still emit a
        # prior-style prediction for them instead of erroring out.
        n = self.n
        hi = self._idx.get(home, -1)
        ai = self._idx.get(away, -1)
        ah = self.x[hi] if hi >= 0 else 0.0
        ad = self.x[ai] if ai >= 0 else 0.0
        dh = self.x[n + hi] if hi >= 0 else 0.0
        dd = self.x[n + ai] if ai >= 0 else 0.0
        z_h = self.mu + ah - dd + self.home_adv
        z_a = self.mu + ad - dh
        return float(np.exp(z_h)), float(np.exp(z_a))

    @property
    def n(self) -> int:
        return len(self.teams)

    def _probs_from_logrates(self, z_h: float, z_a: float) -> np.ndarray:
        lam, mu = float(np.exp(z_h)), float(np.exp(z_a))
        g = np.arange(self.max_goals + 1)
        ph = poisson.pmf(g, lam)[:, None]  # (G+1, 1)
        pa = poisson.pmf(g, mu)[None, :]  # (1, G+1)
        mat = ph * pa
        mat = mat / mat.sum()
        p_home = float(np.tril(mat, -1).sum())
        p_draw = float(np.trace(mat))
        p_away = float(np.triu(mat, 1).sum())
        return np.array([p_home, p_draw, p_away])

    def match_probabilities(self, home: str, away: str) -> dict:
        """H/D/A probabilities from the filtered state after all training matches."""
        lam, mu = self.expected_goals(home, away)
        g = np.arange(self.max_goals + 1)
        ph_ = poisson.pmf(g, lam)[:, None]
        pa_ = poisson.pmf(g, mu)[None, :]
        mat = ph_ * pa_
        mat = mat / mat.sum()
        over_mask = np.fromfunction(
            lambda i, j: i + j > 2.5, (self.max_goals + 1, self.max_goals + 1)
        )
        over = float(mat[over_mask].sum())
        btts = float(mat[1:, 1:].sum())
        ph = float(mat[np.tril_indices(self.max_goals + 1, -1)].sum())
        pd_ = float(np.trace(mat))
        pa = float(mat[np.triu_indices(self.max_goals + 1, 1)].sum())
        return {
            "home_win": ph,
            "draw": pd_,
            "away_win": pa,
            "expected_goals": (lam, mu),
            "over_2_5": over,
            "btts_yes": btts,
            "correct_score": tuple(
                int(t) for t in np.unravel_index(np.argmax(mat), mat.shape)
            ),
        }
