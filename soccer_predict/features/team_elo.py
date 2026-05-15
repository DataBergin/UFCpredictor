"""Team Elo rating system with home advantage for soccer."""

from __future__ import annotations
from typing import Any
from sports_predict.base.rating import GenericEloSystem


class SoccerEloSystem(GenericEloSystem):
    """Elo system tuned for international soccer.

    - Higher K-factor (40) for fewer games and more volatility
    - Home advantage of +100 Elo points
    - Supports draws (score_a = score_b = 0.5)
    - Confederation-aware weighting (optional)
    """

    def __init__(self, k_factor: float = 40, initial_rating: float = 1500,
                 home_advantage: float = 100):
        super().__init__(
            k_factor=k_factor,
            initial_rating=initial_rating,
            home_advantage=home_advantage,
            feature_prefix="elo",
        )

    def get_features(self, entity_a: str, entity_b: str,
                     event_date: str, **context) -> dict[str, float]:
        """Get Elo features with home advantage context."""
        features = super().get_features(entity_a, entity_b, event_date, **context)

        # Add soccer-specific: expected draw probability
        r_a = self.get_rating(entity_a)
        r_b = self.get_rating(entity_b)
        exp_a = self.expected_score(r_a.rating + self.home_advantage, r_b.rating)

        # Rough draw probability estimate based on rating closeness
        rating_diff = abs(r_a.rating - r_b.rating)
        draw_prob = max(0.05, 0.30 - rating_diff / 2000)
        features["elo_draw_prob"] = draw_prob

        return features
