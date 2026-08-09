"""
Feature engineering for the ML meta-layer. These features capture things
Dixon-Coles and Elo don't directly model: recent form trend, rest/fatigue,
and head-to-head history.

IMPORTANT: every feature here must be computable using ONLY information
available strictly BEFORE kickoff. This is the #1 place people leak future
information into sports models (e.g. using full-season stats to predict a
match from mid-season) and get unrealistically good backtest results.
"""

import numpy as np
import pandas as pd


def rolling_form(
    df: pd.DataFrame,
    team_col_home="home_team",
    team_col_away="away_team",
    window: int = 5,
) -> pd.DataFrame:
    """
    Adds pre-match rolling features for both teams:
      - points per game (last `window` matches)
      - goals scored / conceded per game
      - days since last match (fatigue proxy)
    df must be sorted by date ascending and have columns:
      date, home_team, away_team, home_goals, away_goals
    """
    df = df.sort_values("date").reset_index(drop=True)
    records = []
    for team in pd.concat([df[team_col_home], df[team_col_away]]).unique():
        team_matches = df[
            (df[team_col_home] == team) | (df[team_col_away] == team)
        ].copy()
        team_matches = team_matches.sort_values("date")
        for _, row in team_matches.iterrows():
            is_home = row[team_col_home] == team
            gf = row["home_goals"] if is_home else row["away_goals"]
            ga = row["away_goals"] if is_home else row["home_goals"]
            if gf > ga:
                pts = 3
            elif gf == ga:
                pts = 1
            else:
                pts = 0
            records.append(
                {"team": team, "date": row["date"], "pts": pts, "gf": gf, "ga": ga}
            )

    form_df = pd.DataFrame(records).sort_values(["team", "date"])
    form_df["ppg"] = form_df.groupby("team")["pts"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    form_df["gf_avg"] = form_df.groupby("team")["gf"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    form_df["ga_avg"] = form_df.groupby("team")["ga"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    form_df["days_since_last"] = form_df.groupby("team")["date"].diff().dt.days

    lookup = form_df.drop_duplicates(subset=["team", "date"], keep="last")[
        ["team", "date", "ppg", "gf_avg", "ga_avg", "days_since_last"]
    ]

    out = df.copy()
    for prefix, col in [("home", team_col_home), ("away", team_col_away)]:
        renamed = lookup.rename(
            columns={
                "team": col,
                "ppg": f"{prefix}_ppg",
                "gf_avg": f"{prefix}_gf_avg",
                "ga_avg": f"{prefix}_ga_avg",
                "days_since_last": f"{prefix}_days_since_last",
            }
        )
        out = out.merge(renamed, on=[col, "date"], how="left")

    out["form_ppg_diff"] = out["home_ppg"] - out["away_ppg"]
    out["form_goal_diff"] = (out["home_gf_avg"] - out["home_ga_avg"]) - (
        out["away_gf_avg"] - out["away_ga_avg"]
    )
    out["rest_diff"] = out["home_days_since_last"] - out["away_days_since_last"]
    return out


def head_to_head(df: pd.DataFrame, lookback_matches: int = 5) -> pd.Series:
    """Pre-match H2H points-per-game for the home team over the last N meetings, computed
    strictly from matches before the current row's date."""
    df = df.sort_values("date").reset_index(drop=True)
    # meetings[pair] = chronological list of
    # (past_home_team, past_away_team, home_pts, away_pts)
    # for prior meetings between those two teams (either orientation).
    meetings: dict[tuple[str, str], list[tuple[str, str, float, float]]] = {}
    result = [0.5] * len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        pair = tuple(sorted([row["home_team"], row["away_team"]]))
        hist = meetings.get(pair)
        if hist:
            recent = hist[-lookback_matches:]
            pts = [
                (home_pts if past_home == row["home_team"] else away_pts)
                for past_home, _, home_pts, away_pts in recent
            ]
            result[i] = float(np.mean(pts)) / 3.0
        # Record this match so it only influences strictly later rows.
        # (Skip NaN-goal pseudo rows used for upcoming-match predictions —
        # they must never count as a past meeting, same as before.)
        hg, ag = row["home_goals"], row["away_goals"]
        if not (pd.isna(hg) or pd.isna(ag)):
            home_pts = 3.0 if hg > ag else 1.0 if hg == ag else 0.0
            away_pts = 0.0 if hg > ag else 1.0 if hg == ag else 3.0
            meetings.setdefault(pair, []).append(
                (row["home_team"], row["away_team"], home_pts, away_pts)
            )
    return pd.Series(result, index=df.index, name="h2h_home_ppg_norm")
