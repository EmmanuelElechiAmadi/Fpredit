"""
Stacked ensemble: combines
  1. Dixon-Coles match probabilities (principled goal-based model)
  2. Elo win/draw/loss probabilities (long-run team strength, reacts fast to form)
  3. Rolling form + H2H features (captures short-term trend Dixon-Coles smooths away)

A multinomial logistic regression meta-learner is trained on top of these three
probability triplets (+ raw form features) to output final calibrated
Home/Draw/Away probabilities. This is standard stacking: no single model wins
every case (Dixon-Coles is steadier long-run, Elo reacts faster to hot/cold
streaks), so let the data decide how much to trust each one.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .dixon_coles import DixonColes
from .elo import EloEngine
from .features import head_to_head, rolling_form


class FootballEnsemble:
    def __init__(
        self,
        elo_kwargs=None,
        dc_kwargs=None,
        meta_max_iter: int = 2000,
        meta_C: float = 1.0,
    ):
        self.elo = EloEngine(**(elo_kwargs or {}))
        self.dc = DixonColes(**(dc_kwargs or {}))
        self.scaler = StandardScaler()
        self.meta = LogisticRegression(max_iter=meta_max_iter, C=meta_C)
        self.feature_cols = [
            "dc_home",
            "dc_draw",
            "dc_away",
            "elo_home",
            "elo_draw",
            "elo_away",
            "form_ppg_diff",
            "form_goal_diff",
            "rest_diff",
            "h2h_home_ppg_norm",
        ]

    def _build_feature_frame(
        self, df: pd.DataFrame, fit_elo=False, fit_dc=False
    ) -> pd.DataFrame:
        matches = df.to_dict("records")

        if fit_dc:
            self.dc.fit(matches)

        df = rolling_form(df)
        df["h2h_home_ppg_norm"] = head_to_head(df).values

        dc_rows, elo_rows = [], []
        for m in matches:
            try:
                p = self.dc.match_probabilities(m["home_team"], m["away_team"])
                dc_rows.append([p["home_win"], p["draw"], p["away_win"]])
            except Exception:
                dc_rows.append([1 / 3, 1 / 3, 1 / 3])

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

        dc_arr, elo_arr = np.array(dc_rows), np.array(elo_rows)
        df["dc_home"], df["dc_draw"], df["dc_away"] = (
            dc_arr[:, 0],
            dc_arr[:, 1],
            dc_arr[:, 2],
        )
        df["elo_home"], df["elo_draw"], df["elo_away"] = (
            elo_arr[:, 0],
            elo_arr[:, 1],
            elo_arr[:, 2],
        )
        return df

    def fit(self, df: pd.DataFrame):
        """df must be sorted ascending by date with columns:
        date, home_team, away_team, home_goals, away_goals, result (H/D/A)."""
        df = df.sort_values("date").reset_index(drop=True)

        # Fit Dixon-Coles once on the full training window (it internally time-decays)
        self.dc.fit(df.to_dict("records"))

        # Elo must be fit chronologically (walk forward) so ratings reflect only past info
        self.elo = EloEngine(
            **{
                "k": self.elo.k,
                "home_advantage": self.elo.home_advantage,
                "initial_rating": self.elo.initial_rating,
                "goal_diff_multiplier": self.elo.goal_diff_multiplier,
                "draw_width": self.elo.draw_width,
                "draw_min": self.elo.draw_min,
                "draw_max": self.elo.draw_max,
            }
        )
        feat_df = self._build_feature_frame(df, fit_elo=True, fit_dc=False)

        feat_df = feat_df.dropna(subset=self.feature_cols)
        X = self.scaler.fit_transform(feat_df[self.feature_cols])
        y = feat_df["result"]
        self.meta.fit(X, y)
        self.train_df = df  # keep for form/H2H lookups at predict time
        return self

    def predict(self, home_team: str, away_team: str, as_of_date=None) -> dict:
        if as_of_date is None:
            as_of_date = self.train_df["date"].max() + pd.Timedelta(days=7)

        dc_p = self.dc.match_probabilities(home_team, away_team)
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
                }
            ]
        )
        combined = pd.concat([recent, pseudo_row], ignore_index=True).sort_values(
            "date"
        )
        combined = rolling_form(
            combined.assign(
                home_goals=combined["home_goals"].fillna(0),
                away_goals=combined["away_goals"].fillna(0),
            )
        )
        last_row = combined.iloc[-1]
        h2h = head_to_head(pd.concat([recent, pseudo_row], ignore_index=True)).iloc[-1]

        x = pd.DataFrame(
            [
                [
                    dc_p["home_win"],
                    dc_p["draw"],
                    dc_p["away_win"],
                    eh,
                    ed,
                    ea,
                    last_row["form_ppg_diff"],
                    last_row["form_goal_diff"],
                    last_row["rest_diff"],
                    h2h,
                ]
            ],
            columns=self.feature_cols,
        )
        x = x.fillna(0.0)
        x_scaled = self.scaler.transform(x)
        proba = self.meta.predict_proba(x_scaled)[0]
        classes = list(self.meta.classes_)

        result = {c: proba[classes.index(c)] for c in ["H", "D", "A"]}
        return {
            "home_win": result["H"],
            "draw": result["D"],
            "away_win": result["A"],
            "predicted_result": max(result, key=lambda k: result[k]),
            "most_likely_score": dc_p["correct_score"],
            "expected_goals": dc_p["expected_goals"],
            "over_2_5_goals": dc_p["over_2_5"],
            "btts_yes": dc_p["btts_yes"],
            "component_probs": {
                "dixon_coles": [dc_p["home_win"], dc_p["draw"], dc_p["away_win"]],
                "elo": [eh, ed, ea],
            },
        }
