"""
Dixon-Coles model (Dixon & Coles, 1997) — the standard statistical baseline
for football match prediction, still competitive today and the backbone of
most serious betting-market and academic models.

Each team gets an ATTACK strength and DEFENSE strength. Expected goals:
    lambda_home = exp(attack_home - defense_away + home_adv)
    lambda_away = exp(attack_away - defense_home)

Goals are modelled as (near-)independent Poisson variables, with a small
correlation correction (rho) for low-scoring results (0-0, 1-0, 0-1, 1-1),
since plain independent Poisson underestimates how often those happen.

We also apply exponential time-decay weighting so recent matches count more
than old ones (form matters more than a result from 3 seasons ago) — this is
the "xi" (half-life) parameter in the original paper.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson


def _tau(x, y, lam, mu, rho):
    """Dixon-Coles low-score correlation adjustment."""
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    elif x == 0 and y == 1:
        return 1 + lam * rho
    elif x == 1 and y == 0:
        return 1 + mu * rho
    elif x == 1 and y == 1:
        return 1 - rho
    return 1.0


class DixonColes:
    def __init__(self, xi: float = 0.0018):
        """xi: time-decay rate. 0.0018/day ~= results from 1 year ago carry ~55% weight,
        2 years ago ~30%. Tune via cross-validation; higher xi = more reactive to recent form.
        """
        self.xi = xi
        self.teams: list[str] = []
        self.attack: dict[str, float] = {}
        self.defense: dict[str, float] = {}
        self.home_adv = 0.25
        self.rho = -0.05

    def _weights(self, dates, ref_date):
        days = np.array([(ref_date - d).days for d in dates], dtype=float)
        days = np.clip(days, 0, None)
        return np.exp(-self.xi * days)

    def fit(self, matches, ref_date=None):
        """matches: list of dicts with date, home_team, away_team, home_goals, away_goals"""
        self.teams = sorted(
            set([m["home_team"] for m in matches] + [m["away_team"] for m in matches])
        )
        n = len(self.teams)
        idx = {t: i for i, t in enumerate(self.teams)}
        ref_date = ref_date or max(m["date"] for m in matches)

        home_idx = np.array([idx[m["home_team"]] for m in matches])
        away_idx = np.array([idx[m["away_team"]] for m in matches])
        hg = np.array([m["home_goals"] for m in matches])
        ag = np.array([m["away_goals"] for m in matches])
        dates = [m["date"] for m in matches]
        w = self._weights(dates, ref_date)

        # params: [attack_0..n-1, defense_0..n-1, home_adv, rho]
        # Constraint: mean(attack) = 0 for identifiability.
        # We enforce this *post-fit* by re-centering, which avoids biasing the
        # likelihood with a penalty term. (Dixon & Coles 1997, Section 2.2)
        x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25], [-0.05]])

        def _nll(params, out_lam=None, out_mu=None):
            atk = params[:n]
            dfn = params[n : 2 * n]
            home_adv = params[2 * n]
            rho = params[2 * n + 1]

            lam = np.exp(atk[home_idx] - dfn[away_idx] + home_adv)
            mu = np.exp(atk[away_idx] - dfn[home_idx])

            if out_lam is not None:
                out_lam[:] = lam
            if out_mu is not None:
                out_mu[:] = mu

            ll = poisson.logpmf(hg, lam) + poisson.logpmf(ag, mu)
            tau_vals = np.array(
                [
                    _tau(h, a, lam_i, mu_i, rho)
                    for h, a, lam_i, mu_i in zip(hg, ag, lam, mu)
                ]
            )
            tau_vals = np.clip(tau_vals, 1e-10, None)
            ll = ll + np.log(tau_vals)

            return -np.sum(w * ll)

        bounds = [(-3, 3)] * n + [(-3, 3)] * n + [(-1, 1)] + [(-0.3, 0.3)]

        # Try L-BFGS-B first (fast, good for smooth likelihoods)
        res = minimize(
            _nll, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 1000}
        )

        # If L-BFGS-B didn't converge, fall back to Nelder-Mead (slower but more robust)
        if not res.success and res.status != 0:
            res = minimize(
                _nll,
                res.x,
                method="Nelder-Mead",
                bounds=bounds,
                options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-6},
            )

        params = res.x

        # Post-fit re-centering: subtract mean(attack) so sum(attack) = 0.
        attack_offset = np.mean(params[:n])
        params[:n] -= attack_offset
        params[
            n : 2 * n
        ] += attack_offset  # defense shifts oppositely to keep expected goals unchanged

        self.attack = {t: params[idx[t]] for t in self.teams}
        self.defense = {t: params[n + idx[t]] for t in self.teams}
        self.home_adv = params[2 * n]
        self.rho = params[2 * n + 1]
        self.converged = res.success
        if not self.converged:
            import warnings

            warnings.warn(
                f"Dixon-Coles optimizer did not converge (status={res.status}): {res.message}"
            )
        return self

    def expected_goals(self, home, away):
        if home not in self.attack or away not in self.attack:
            raise ValueError(
                f"Unknown team(s): check '{home}' / '{away}' against self.teams"
            )
        lam = np.exp(self.attack[home] - self.defense[away] + self.home_adv)
        mu = np.exp(self.attack[away] - self.defense[home])
        return lam, mu

    def score_matrix(self, home, away, max_goals=10):
        lam, mu = self.expected_goals(home, away)
        mat = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                p = (
                    poisson.pmf(i, lam)
                    * poisson.pmf(j, mu)
                    * _tau(i, j, lam, mu, self.rho)
                )
                mat[i, j] = max(p, 0)
        mat /= mat.sum()
        return mat

    def match_probabilities(self, home, away, max_goals=10):
        mat = self.score_matrix(home, away, max_goals)
        p_home = np.tril(mat, -1).sum()
        p_draw = np.trace(mat)
        p_away = np.triu(mat, 1).sum()
        most_likely_idx = np.unravel_index(np.argmax(mat), mat.shape)
        over_2_5 = sum(
            mat[i, j]
            for i in range(max_goals + 1)
            for j in range(max_goals + 1)
            if i + j > 2.5
        )
        btts = sum(
            mat[i, j] for i in range(1, max_goals + 1) for j in range(1, max_goals + 1)
        )
        return {
            "home_win": p_home,
            "draw": p_draw,
            "away_win": p_away,
            "correct_score": most_likely_idx,
            "over_2_5": over_2_5,
            "under_2_5": 1 - over_2_5,
            "btts_yes": btts,
            "btts_no": 1 - btts,
            "expected_goals": self.expected_goals(home, away),
        }
