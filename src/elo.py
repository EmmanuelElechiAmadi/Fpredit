"""
Elo rating engine adapted for football (soccer).

Standard chess Elo assumes two outcomes. Football has draws, so we use the
football-specific formulation: actual score S = 1 for win, 0.5 for draw,
0 for loss, and expected score from the logistic curve, same as chess.

A home-advantage constant is added to the home team's rating before computing
expected score (industry standard is ~60-100 Elo points; we default to 75 and
let it be recalibrated from data).
"""

from dataclasses import dataclass, field


@dataclass
class EloEngine:
    k: float = 20.0  # update speed. Higher = more reactive to recent results
    home_advantage: float = 75.0  # elo points added to home team for expectation only
    initial_rating: float = 1500.0
    goal_diff_multiplier: bool = (
        True  # scale K by margin of victory (Elo modification common in football)
    )
    ratings: dict = field(default_factory=dict)
    history: list = field(
        default_factory=list
    )  # (date, home, away, home_elo_pre, away_elo_pre)

    def get(self, team: str) -> float:
        return self.ratings.get(team, self.initial_rating)

    def expected_score(self, home: str, away: str) -> float:
        """Probability-like expected score for the HOME team, accounting for home advantage."""
        rh = self.get(home) + self.home_advantage
        ra = self.get(away)
        return 1.0 / (1.0 + 10 ** ((ra - rh) / 400.0))

    def _margin_multiplier(self, goal_diff: int) -> float:
        """Larger wins move ratings more (World Football Elo style dampened log scale)."""
        if not self.goal_diff_multiplier or goal_diff == 0:
            return 1.0
        return min(1.0 + 0.35 * (abs(goal_diff) ** 0.5), 2.2)

    def update(self, date, home: str, away: str, home_goals: int, away_goals: int):
        rh, ra = self.get(home), self.get(away)
        self.history.append((date, home, away, rh, ra))

        exp_home = self.expected_score(home, away)
        if home_goals > away_goals:
            s_home = 1.0
        elif home_goals == away_goals:
            s_home = 0.5
        else:
            s_home = 0.0

        mult = self._margin_multiplier(home_goals - away_goals)
        delta = self.k * mult * (s_home - exp_home)

        self.ratings[home] = rh + delta
        self.ratings[away] = ra - delta
        return delta

    def fit(self, matches):
        """matches: iterable of dicts/rows with date, home_team, away_team, home_goals, away_goals,
        already sorted chronologically."""
        for m in matches:
            self.update(
                m["date"],
                m["home_team"],
                m["away_team"],
                m["home_goals"],
                m["away_goals"],
            )

    def win_draw_loss_prob(self, home: str, away: str, draw_width: float = 0.44):
        """
        Elo alone only gives an expected SCORE, not three-way probabilities.
        We convert using an empirically-reasonable logistic split: the draw probability
        is highest when teams are evenly matched and shrinks as the rating gap grows.
        This is a heuristic approximation -- Dixon-Coles below is the principled version.
        """
        rh = self.get(home) + self.home_advantage
        ra = self.get(away)
        diff = rh - ra
        p_home_or_draw_beats_away = 1.0 / (
            1.0 + 10 ** (-diff / 400.0)
        )  # win-or-draw tendency
        # draw prob peaks at diff=0 and decays with |diff|
        p_draw = draw_width * (1.0 - min(abs(diff) / 800.0, 0.9))
        p_draw = max(0.12, min(p_draw, 0.34))
        remaining = 1.0 - p_draw
        p_home = remaining * p_home_or_draw_beats_away
        p_away = remaining * (1.0 - p_home_or_draw_beats_away)
        total = p_home + p_draw + p_away
        return p_home / total, p_draw / total, p_away / total
