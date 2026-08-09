"""
Stacked ensemble: combines
  1. Dynamic state-space (Kalman-filtered) Poisson strengths — leak-free
     filtered probabilities that evolve match-by-match (the principled
     replacement for a static-window Dixon-Coles fit, and immune to the
     in-sample leakage a static fit produces when predicting its own data)
  2. Elo win/draw/loss probabilities (long-run team strength, reacts fast)
  3. Rolling form + H2H + congestion + league-position + PageRank features
  4. Market implied probabilities when odds are present — the residual-vs-
     market meta-model: with the closing line as a feature, the meta-learner
     can only contribute signal the market has not yet priced
  5. xG-derived probabilities when xG data is available (a second dynamic
     model filtered on xG, which is a far less noisy observation than goals)

A multinomial logistic regression meta-learner is trained on top of these
signals to output final calibrated Home/Draw/Away probabilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .dixon_coles import DixonColes
from .elo import EloEngine
from .features import (
    fixture_congestion,
    head_to_head,
    league_position,
    pagerank_strength,
    rolling_form,
)
from .market import add_market_features, odds_columns_available
from .state_space import StateSpaceModel
from .xg_loader import join_xg

_NEUTRAL = 1.0 / 3.0


class FootballEnsemble:
    def __init__(
        self,
        elo_kwargs=None,
        dc_kwargs=None,
        meta_max_iter: int = 2000,
        meta_C: float = 1.0,
        ss_q: float = 0.01,
        ss_q_xg: float = 0.005,
        ss_prior_var: float = 0.25,
        ss_obs_var_scale_xg: float = 0.5,
        use_market_features: bool = True,
        form_window: int = 5,
        h2h_lookback: int = 5,
        congestion_days: int = 8,
        load_days: int = 14,
        position_reset_days: int = 100,
    ):
        self._elo_kwargs = dict(elo_kwargs or {})
        self._dc_kwargs = dict(dc_kwargs or {})
        self.elo = EloEngine(**self._elo_kwargs)
        self.dc = DixonColes(**self._dc_kwargs)

        self.ss_q = ss_q
        self.ss_q_xg = ss_q_xg
        self.ss_prior_var = ss_prior_var
        self.ss_obs_var_scale_xg = ss_obs_var_scale_xg
        self.use_market_features = use_market_features
        self.form_window = form_window
        self.h2h_lookback = h2h_lookback
        self.congestion_days = congestion_days
        self.load_days = load_days
        self.position_reset_days = position_reset_days

        self.scaler = StandardScaler()
        self.meta = LogisticRegression(max_iter=meta_max_iter, C=meta_C)

        self.dyn = StateSpaceModel(q=ss_q, prior_var=ss_prior_var)
        self.dyn_xg: StateSpaceModel | None = None
        self.has_xg = False
        self.feature_cols: list[str] = []
        self.train_df: pd.DataFrame | None = None

    def _base_feature_cols(self) -> list[str]:
        return [
            "dyn_home",
            "dyn_draw",
            "dyn_away",
            "elo_home",
            "elo_draw",
            "elo_away",
            "form_ppg_diff",
            "form_goal_diff",
            "rest_diff",
            "h2h_home_ppg_norm",
            "load_diff",
            "congestion_diff",
            "pos_diff",
            "pagerank_diff",
        ]

    def _build_feature_frame(
        self, df: pd.DataFrame, fit_elo=False, fit_dc=False
    ) -> pd.DataFrame:
        """Build the meta-learner feature matrix. All features are strictly
        pre-match / leak-free: the dynamic-model probabilities come from
        filtered estimates that use only prior matches."""
        if fit_dc:
            self.dc.fit(df.to_dict("records"))

        df = df.sort_values("date").reset_index(drop=True)
        df = rolling_form(df, window=self.form_window)
        df["h2h_home_ppg_norm"] = head_to_head(
            df, lookback_matches=self.h2h_lookback
        ).values
        df = fixture_congestion(df, self.congestion_days, self.load_days)
        df = league_position(df, self.position_reset_days)
        df = pagerank_strength(df)

        # Dynamic (Kalman) filtered probabilities — leak-free by construction
        if self.dyn.filtered_probs is not None and len(self.dyn.filtered_probs) == len(
            df
        ):
            df["dyn_home"], df["dyn_draw"], df["dyn_away"] = (
                self.dyn.filtered_probs[:, 0],
                self.dyn.filtered_probs[:, 1],
                self.dyn.filtered_probs[:, 2],
            )
        else:
            df["dyn_home"] = df["dyn_draw"] = df["dyn_away"] = _NEUTRAL

        # xG dynamic model (fitted on the same chronological matches)
        if self.dyn_xg is not None and self.dyn_xg.filtered_probs is not None:
            df["xg_home"], df["xg_draw"], df["xg_away"] = (
                self.dyn_xg.filtered_probs[:, 0],
                self.dyn_xg.filtered_probs[:, 1],
                self.dyn_xg.filtered_probs[:, 2],
            )

        # Elo, updated chronologically so each row sees only prior matches
        elo_rows = []
        for m in df.to_dict("records"):
            eh, ed, ea = self.elo.win_draw_loss_prob(m["home_team"], m["away_team"])
            elo_rows.append([eh, ed, ea])
            if fit_elo:
                self.elo.update(
                    m["date"],
                    m["home_team"],
                    m["away_team"],
                    m["home_goals"],
                    m["away_goals"],
                )
        elo_arr = np.array(elo_rows)
        df["elo_home"], df["elo_draw"], df["elo_away"] = (
            elo_arr[:, 0],
            elo_arr[:, 1],
            elo_arr[:, 2],
        )

        # Market implied probabilities + line movement (residual-vs-market)
        if self.use_market_features and odds_columns_available(df):
            df = add_market_features(df)
            for c in ("market_home", "market_draw", "market_away"):
                df[c] = df[c].fillna(_NEUTRAL)
            for c in ("line_mv_home", "line_mv_draw", "line_mv_away", "line_mv_abs"):
                df[c] = df[c].fillna(0.0)
        return df

    def fit(self, df: pd.DataFrame, xg_df: pd.DataFrame | None = None):
        """df must be sorted ascending by date with columns:
        date, home_team, away_team, home_goals, away_goals, result (H/D/A).

        xg_df: optional Understat-style frame with date, home_team, away_team,
        home_xg, away_xg — when provided, xG features are added automatically.
        """
        df = df.sort_values("date").reset_index(drop=True)
        if xg_df is not None and not xg_df.empty:
            df = join_xg(df, xg_df)
        self.train_df = df
        matches = df.to_dict("records")
        self.has_xg = "home_xg" in df.columns and df["home_xg"].notna().any()

        # Static Dixon-Coles on goals — used for scoreline/EG diagnostics and
        # as the standalone component reported to the user.
        self.dc = DixonColes(**self._dc_kwargs)
        self.dc.fit(matches)

        # Dynamic state-space model on goals: leak-free filtered probabilities
        self.dyn = StateSpaceModel(q=self.ss_q, prior_var=self.ss_prior_var)
        self.dyn.filter_matches(matches)

        if self.has_xg:
            self.dyn_xg = StateSpaceModel(
                q=self.ss_q_xg,
                prior_var=self.ss_prior_var,
                obs_var_scale=self.ss_obs_var_scale_xg,
            )
            self.dyn_xg.filter_matches(matches, use_xg=True)
        else:
            self.dyn_xg = None

        # Elo must be fit chronologically (walk forward) so ratings reflect only past info
        self.elo = EloEngine(**self._elo_kwargs)

        # Determine the meta-learner feature set from what the data provides
        self.feature_cols = self._base_feature_cols()
        if self.has_xg:
            self.feature_cols += ["xg_home", "xg_draw", "xg_away"]
        if self.use_market_features and odds_columns_available(df):
            self.feature_cols += [
                "market_home",
                "market_draw",
                "market_away",
            ]
            from .market import opening_odds_columns_available

            if opening_odds_columns_available(df):
                self.feature_cols += [
                    "line_mv_home",
                    "line_mv_draw",
                    "line_mv_away",
                    "line_mv_abs",
                ]

        feat_df = self._build_feature_frame(df, fit_elo=True)
        feat_df = feat_df.dropna(subset=self.feature_cols)
        if len(feat_df) < 30:
            raise ValueError(
                f"Too few rows with complete features ({len(feat_df)}) to fit the meta-learner"
            )
        X = self.scaler.fit_transform(feat_df[self.feature_cols])
        y = feat_df["result"]
        self.meta.fit(X, y)
        return self

    def predict(
        self,
        home_team: str,
        away_team: str,
        as_of_date=None,
        market_odds: tuple | None = None,
    ) -> dict:
        """Predict H/D/A probabilities.

        market_odds: optional (home_odds, draw_odds, away_odds) decimal odds
        for the fixture. When provided, the meta-learner's market features use
        them (residual-vs-market inference). When omitted, market features are
        neutralized to 1/3 — the model then predicts with no market signal.
        """
        if self.train_df is None:
            raise RuntimeError("FootballEnsemble must be fitted before predict()")
        if as_of_date is None:
            as_of_date = self.train_df["date"].max() + pd.Timedelta(days=7)

        dc_p = self.dc.match_probabilities(home_team, away_team)
        dyn_p = self.dyn.match_probabilities(home_team, away_team)
        eh, ed, ea = self.elo.win_draw_loss_prob(home_team, away_team)

        recent = self.train_df[
            (self.train_df["home_team"].isin([home_team, away_team]))
            | (self.train_df["away_team"].isin([home_team, away_team]))
        ]
        pseudo_row = pd.DataFrame(
            [
                {
                    "date": as_of_date,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_goals": np.nan,
                    "away_goals": np.nan,
                    "result": None,
                }
            ]
        )
        combined = (
            pd.concat([recent, pseudo_row], ignore_index=True)
            .sort_values("date")
            .reset_index(drop=True)
        )
        combined = combined.assign(
            home_goals=combined["home_goals"].fillna(0),
            away_goals=combined["away_goals"].fillna(0),
        )
        combined = rolling_form(combined, window=self.form_window)
        combined = fixture_congestion(combined, self.congestion_days, self.load_days)
        combined = league_position(combined, self.position_reset_days)
        combined = pagerank_strength(combined)
        last_row = combined.iloc[-1]
        h2h = head_to_head(
            pd.concat([recent, pseudo_row], ignore_index=True),
            lookback_matches=self.h2h_lookback,
        ).iloc[-1]

        # Market features: use provided odds, or neutral 1/3 when unknown
        if "market_home" in self.feature_cols:
            if market_odds is not None:
                o_h, o_d, o_a = market_odds
                inv = [1.0 / o_h, 1.0 / o_d, 1.0 / o_a]
                tot = sum(inv)
                m_h, m_d, m_a = [p / tot for p in inv]
            else:
                m_h = m_d = m_a = _NEUTRAL
            line_h = line_d = line_a = line_abs = 0.0
        else:
            m_h = m_d = m_a = _NEUTRAL
            line_h = line_d = line_a = line_abs = 0.0

        # xG features (if the model was trained with them)
        xg_p = None
        if "xg_home" in self.feature_cols and self.dyn_xg is not None:
            xg_p = self.dyn_xg.match_probabilities(home_team, away_team)

        values = {
            "dyn_home": dyn_p["home_win"],
            "dyn_draw": dyn_p["draw"],
            "dyn_away": dyn_p["away_win"],
            "elo_home": eh,
            "elo_draw": ed,
            "elo_away": ea,
            "form_ppg_diff": last_row["form_ppg_diff"],
            "form_goal_diff": last_row["form_goal_diff"],
            "rest_diff": last_row["rest_diff"],
            "h2h_home_ppg_norm": h2h,
            "load_diff": last_row["load_diff"],
            "congestion_diff": last_row["congestion_diff"],
            "pos_diff": last_row["pos_diff"],
            "pagerank_diff": last_row["pagerank_diff"],
            "xg_home": xg_p["home_win"] if xg_p is not None else _NEUTRAL,
            "xg_draw": xg_p["draw"] if xg_p is not None else _NEUTRAL,
            "xg_away": xg_p["away_win"] if xg_p is not None else _NEUTRAL,
            "market_home": m_h,
            "market_draw": m_d,
            "market_away": m_a,
            "line_mv_home": line_h,
            "line_mv_draw": line_d,
            "line_mv_away": line_a,
            "line_mv_abs": line_abs,
        }
        x = pd.DataFrame([{c: values[c] for c in self.feature_cols}])
        x = x.fillna(0.0)
        x_scaled = self.scaler.transform(x)
        proba = self.meta.predict_proba(x_scaled)[0]
        classes = list(self.meta.classes_)

        result = {c: proba[classes.index(c)] for c in ["H", "D", "A"]}
        comps = {
            "dixon_coles": [dc_p["home_win"], dc_p["draw"], dc_p["away_win"]],
            "dynamic": [dyn_p["home_win"], dyn_p["draw"], dyn_p["away_win"]],
            "elo": [eh, ed, ea],
        }
        if xg_p is not None:
            comps["xg"] = [xg_p["home_win"], xg_p["draw"], xg_p["away_win"]]
        return {
            "home_win": result["H"],
            "draw": result["D"],
            "away_win": result["A"],
            "predicted_result": max(result, key=lambda k: result[k]),
            "most_likely_score": dc_p["correct_score"],
            "expected_goals": dc_p["expected_goals"],
            "over_2_5_goals": dc_p["over_2_5"],
            "btts_yes": dc_p["btts_yes"],
            "component_probs": comps,
        }
