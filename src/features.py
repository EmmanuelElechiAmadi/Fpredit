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


def fixture_congestion(
    df: pd.DataFrame,
    congestion_days: int = 8,
    load_days: int = 14,
) -> pd.DataFrame:
    """Fixture congestion and cumulative-load features.

    A team playing its third match in eight days — especially away, after a
    midweek European trip — is measurably weaker than its season-average form
    suggests. Computed strictly from matches BEFORE each kickoff:

      home_matches_3in8 / away_matches_3in8 : 1 if the team has >= 3 matches
                                              in the trailing congestion_days
      home_load / away_load                : number of matches in the trailing
                                              load_days window (cumulative load)
      load_diff                            : home_load - away_load
      congestion_diff                      : 3in8 flags difference

    Rows with NaN goals (pseudo rows for upcoming fixtures) still get feature
    values but never contribute to any team's load.
    """
    df = df.sort_values("date").reset_index(drop=True)

    def _team_counts(team_matches: pd.DataFrame, days: int):
        dates = team_matches["date"].to_numpy()
        counts = np.zeros(len(dates), dtype=int)
        lo = 0
        for i in range(len(dates)):
            while lo < i and dates[lo] < dates[i] - np.timedelta64(days, "D"):
                lo += 1
            counts[i] = i - lo
        return pd.Series(counts, index=team_matches.index)

    played = df[df["home_goals"].notna() & df["away_goals"].notna()]
    out = df.copy()
    for prefix, col in [("home", "home_team"), ("away", "away_team")]:
        sub_rows = []
        for _team, tm in played.groupby(col, sort=False):
            tm = tm.sort_values("date")
            sub_rows.append(
                pd.DataFrame(
                    {
                        col: tm[col].values,
                        "date": tm["date"].values,
                        f"{prefix}_matches_3in8": _team_counts(
                            tm, congestion_days
                        ).values,
                        f"{prefix}_load": _team_counts(tm, load_days).values,
                    }
                )
            )
        if sub_rows:
            sub = pd.concat(sub_rows, ignore_index=True)
        else:
            sub = pd.DataFrame(
                columns=[col, "date", f"{prefix}_matches_3in8", f"{prefix}_load"]
            )
        out = out.merge(sub, on=[col, "date"], how="left")

    out["home_matches_3in8"] = out["home_matches_3in8"].fillna(0).astype(int)
    out["away_matches_3in8"] = out["away_matches_3in8"].fillna(0).astype(int)
    out["home_load"] = out["home_load"].fillna(0).astype(int)
    out["away_load"] = out["away_load"].fillna(0).astype(int)
    # The current match counts as one of the N: 'third match in eight days'
    # means 2 prior + this one. Load similarly includes the fixture itself.
    out["home_matches_3in8"] = ((out["home_matches_3in8"] + 1) >= 3).astype(int)
    out["away_matches_3in8"] = ((out["away_matches_3in8"] + 1) >= 3).astype(int)
    out["home_load"] = out["home_load"] + 1
    out["away_load"] = out["away_load"] + 1
    out["load_diff"] = out["home_load"] - out["away_load"]
    out["congestion_diff"] = out["home_matches_3in8"] - out["away_matches_3in8"]
    return out


def league_position(df: pd.DataFrame, reset_days: int = 100) -> pd.DataFrame:
    """Pre-match league standings position for both teams.

    Context/motivation features: a dead rubber, a relegation six-pointer, a
    team already qualified for Europe playing a meaningless final day — these
    are priced intuitively by humans but invisible to a naive statistical
    model. Position is computed strictly from results BEFORE each match, with
    standings reset at season boundaries (detected by a date gap > reset_days).

    Adds columns: home_pos, away_pos, pos_diff (home - away; negative means
    the home side is higher), home_points, away_points, points_diff.
    """
    df = df.sort_values("date").reset_index(drop=True)

    standings: dict[str, list] = {}  # team -> [points, goals_for, goals_against]
    n = len(df)
    home_pos = np.full(n, np.nan)
    away_pos = np.full(n, np.nan)
    home_pts = np.full(n, np.nan)
    away_pts = np.full(n, np.nan)
    prev_date = None

    for i, row in df.iterrows():
        # New season: reset standings if a big date gap has passed
        if prev_date is not None and (row["date"] - prev_date).days > reset_days:
            standings = {}
        prev_date = row["date"]

        if standings:
            home_pos[i] = _position(standings, row["home_team"])
            away_pos[i] = _position(standings, row["away_team"])
            home_pts[i] = standings.get(row["home_team"], (np.nan,))[0]
            away_pts[i] = standings.get(row["away_team"], (np.nan,))[0]

        # Update standings with THIS row only if it is a real result
        if not (pd.isna(row["home_goals"]) or pd.isna(row["away_goals"])):
            _update_standings(standings, row)

    out = df.copy()
    out["home_pos"] = home_pos
    out["away_pos"] = away_pos
    out["pos_diff"] = home_pos - away_pos
    out["home_points"] = home_pts
    out["away_points"] = away_pts
    out["points_diff"] = home_pts - away_pts
    return out


def _position(standings: dict, team: str):
    if team not in standings:
        return np.nan
    pts, gf, ga = standings[team]
    gd = gf - ga
    rank = 1
    for t, (p, g, ga_other) in standings.items():
        if t == team:
            continue
        if p > pts or (p == pts and (g - ga_other) > gd):
            rank += 1
    return float(rank)


def _update_standings(standings: dict, row) -> None:
    h, a = row["home_team"], row["away_team"]
    hg, ag = int(row["home_goals"]), int(row["away_goals"])
    for team, gf, ga in ((h, hg, ag), (a, ag, hg)):
        cur = standings.setdefault(team, [0, 0, 0])
        cur[1] += gf
        cur[2] += ga
        if gf > ga:
            cur[0] += 3
        elif gf == ga:
            cur[0] += 1


def pagerank_strength(
    df: pd.DataFrame, damping: float = 0.85, max_iter: int = 60, tol: float = 1e-8
) -> pd.DataFrame:
    """Transitive strength via PageRank over the directed result graph.

    Not just "did Team A beat Team B" but "did Team A beat a team that keeps
    beating good teams" — strength-of-schedule information that Elo captures
    only weakly and Dixon-Coles not at all. For each match the graph is built
    strictly from prior results (loser -> winner edges, draws treated as
    mutual) and PageRank is solved by power iteration.

    Adds columns: home_pagerank, away_pagerank, pagerank_diff.
    Teams with no prior matches get the neutral uniform score.
    """
    df = df.sort_values("date").reset_index(drop=True)
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    adj = np.zeros((n, n))  # adj[loser, winner] += 1
    home_pr = np.full(len(df), np.nan)
    away_pr = np.full(len(df), np.nan)

    for i, row in df.iterrows():
        if adj.sum() > 0:
            pr = _pagerank(adj, n, damping, max_iter, tol)
            home_pr[i] = pr[idx[row["home_team"]]]
            away_pr[i] = pr[idx[row["away_team"]]]
        else:
            home_pr[i] = 1.0 / n
            away_pr[i] = 1.0 / n

        if not (pd.isna(row["home_goals"]) or pd.isna(row["away_goals"])):
            hg, ag = int(row["home_goals"]), int(row["away_goals"])
            hi, ai = idx[row["home_team"]], idx[row["away_team"]]
            if hg > ag:
                adj[ai, hi] += 1.0
            elif ag > hg:
                adj[hi, ai] += 1.0
            else:
                adj[hi, ai] += 0.5
                adj[ai, hi] += 0.5

    out = df.copy()
    out["home_pagerank"] = home_pr
    out["away_pagerank"] = away_pr
    out["pagerank_diff"] = home_pr - away_pr
    return out


def _pagerank(
    adj: np.ndarray, n: int, damping: float, max_iter: int, tol: float
) -> np.ndarray:
    deg = adj.sum(axis=1)
    stochastic = np.divide(
        adj, deg[:, None], out=np.zeros_like(adj), where=(deg[:, None] > 0)
    )
    rank = np.ones(n) / n
    for _ in range(max_iter):
        new = (1.0 - damping) / n + damping * (rank @ stochastic)
        if np.linalg.norm(new - rank) < tol:
            rank = new
            break
        rank = new
    s = rank.sum()
    return rank / s if s > 0 else rank


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
