"""Generic Elo rating system that works for individuals and teams."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from .feature_source import StatefulFeatureSource

logger = logging.getLogger(__name__)


@dataclass
class EloRating:
    """Rating state for a single entity."""
    rating: float = 1500.0
    fights: int = 0
    wins: int = 0
    losses: int = 0


class GenericEloSystem(StatefulFeatureSource):
    """Elo rating system parameterized for any head-to-head sport.

    Works for both individual matchups (UFC) and team matchups (soccer, MLB).
    Supports optional home-field advantage for team sports.

    Args:
        k_factor: Base K-factor for rating updates (higher = more volatile)
        initial_rating: Starting rating for new entities
        home_advantage: Elo points added to home entity (0 for neutral venues)
        adaptive_k: If True, scale K-factor by entity experience
        feature_prefix: Prefix for feature names (e.g., "elo", "team_elo")
    """

    def __init__(self, k_factor: float = 32, initial_rating: float = 1500,
                 home_advantage: float = 0, adaptive_k: bool = True,
                 feature_prefix: str = "elo"):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.home_advantage = home_advantage
        self.adaptive_k = adaptive_k
        self.prefix = feature_prefix
        self.ratings: dict[str, EloRating] = {}

    def get_rating(self, entity: str) -> EloRating:
        """Get or create rating for an entity."""
        if entity not in self.ratings:
            self.ratings[entity] = EloRating(rating=self.initial_rating)
        return self.ratings[entity]

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Expected score for A given ratings."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))

    def get_features(self, entity_a: str, entity_b: str,
                     event_date: str, **context) -> dict[str, float]:
        """Get pre-event Elo features without updating."""
        r_a = self.get_rating(entity_a)
        r_b = self.get_rating(entity_b)

        # Apply home advantage if applicable
        row = context.get("row")
        is_home = False
        if row is not None and self.home_advantage > 0:
            home_entity = row.get("home_team", row.get("entity_a", ""))
            is_home = (entity_a == home_entity)

        effective_a = r_a.rating + (self.home_advantage if is_home else 0)
        effective_b = r_b.rating + (self.home_advantage if not is_home and self.home_advantage > 0 else 0)

        p = self.prefix
        return {
            f"{p}_a": r_a.rating,
            f"{p}_b": r_b.rating,
            f"{p}_diff": r_a.rating - r_b.rating,
            f"{p}_expected_a": self.expected_score(effective_a, effective_b),
            f"{p}_games_a": r_a.fights,
            f"{p}_games_b": r_b.fights,
            f"{p}_winrate_a": r_a.wins / max(r_a.fights, 1),
            f"{p}_winrate_b": r_b.wins / max(r_b.fights, 1),
        }

    def empty_features(self) -> dict[str, float]:
        p = self.prefix
        return {
            f"{p}_a": self.initial_rating,
            f"{p}_b": self.initial_rating,
            f"{p}_diff": 0.0,
            f"{p}_expected_a": 0.5,
            f"{p}_games_a": 0,
            f"{p}_games_b": 0,
            f"{p}_winrate_a": 0.0,
            f"{p}_winrate_b": 0.0,
        }

    def update(self, result: dict[str, Any]) -> None:
        """Update ratings after an event.

        Args:
            result: Must contain 'winner' and 'loser' keys, OR
                    'entity_a', 'entity_b', 'score_a', 'score_b' for draws.
        """
        winner = result.get("winner")
        loser = result.get("loser")

        if winner and loser:
            self._update_match(winner, loser, score_a=1.0, score_b=0.0)
        elif "entity_a" in result and "entity_b" in result:
            entity_a = result["entity_a"]
            entity_b = result["entity_b"]
            score_a = result.get("score_a", 0.5)  # 0.5 = draw
            score_b = result.get("score_b", 0.5)
            self._update_match(entity_a, entity_b, score_a, score_b)

    def _update_match(self, entity_a: str, entity_b: str,
                      score_a: float, score_b: float) -> None:
        """Update ratings for a single match."""
        r_a = self.get_rating(entity_a)
        r_b = self.get_rating(entity_b)

        expected_a = self.expected_score(r_a.rating, r_b.rating)
        expected_b = 1.0 - expected_a

        k_a = self._adaptive_k(r_a) if self.adaptive_k else self.k_factor
        k_b = self._adaptive_k(r_b) if self.adaptive_k else self.k_factor

        r_a.rating += k_a * (score_a - expected_a)
        r_b.rating += k_b * (score_b - expected_b)

        r_a.fights += 1
        r_b.fights += 1
        if score_a > score_b:
            r_a.wins += 1
            r_b.losses += 1
        elif score_b > score_a:
            r_b.wins += 1
            r_a.losses += 1

    def _adaptive_k(self, rating: EloRating) -> float:
        """Scale K-factor: higher for new entities, lower for established ones."""
        if rating.fights < 5:
            return self.k_factor * 1.5
        elif rating.fights < 15:
            return self.k_factor
        else:
            return self.k_factor * 0.75

    def reset(self) -> None:
        self.ratings.clear()
