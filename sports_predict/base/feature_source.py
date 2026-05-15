"""Abstract interfaces for feature sources across all sports."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class FeatureSource(ABC):
    """Base interface for all feature generators.

    A feature source takes two competing entities (fighters, teams) and a date,
    and returns a dictionary of numerical features for that matchup.
    """

    @abstractmethod
    def get_features(self, entity_a: str, entity_b: str,
                     event_date: str, **context) -> dict[str, float]:
        """Return features for a matchup.

        Args:
            entity_a: First entity (fighter name, team name, etc.)
            entity_b: Second entity
            event_date: Date string for temporal filtering
            **context: Additional context (row data, sport-specific info)

        Returns:
            Dict mapping feature names to float values
        """
        ...

    def empty_features(self) -> dict[str, float]:
        """Return zero-filled feature dict when data is unavailable.

        Override this in subclasses to define the expected feature schema.
        """
        return {}


class StatefulFeatureSource(FeatureSource):
    """Feature source that maintains internal state (e.g., Elo ratings).

    State is updated after each event outcome is known. Features are always
    computed BEFORE the update to prevent information leakage.
    """

    @abstractmethod
    def update(self, result: dict[str, Any]) -> None:
        """Update internal state after an event outcome is known.

        Args:
            result: Dict containing outcome information. Keys depend on the sport:
                - UFC: winner, loser, method, round, etc.
                - Soccer: home_team, away_team, home_goals, away_goals
                - MLB: home_team, away_team, winner, score
        """
        ...

    def reset(self) -> None:
        """Reset all internal state. Override if the source is stateful."""
        pass
