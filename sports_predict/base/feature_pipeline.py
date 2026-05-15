"""Abstract base feature pipeline with chronological processing."""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from .feature_source import FeatureSource, StatefulFeatureSource

logger = logging.getLogger(__name__)


class BaseFeaturePipeline(ABC):
    """Sport-generic feature pipeline that processes events chronologically.

    Subclasses define how to extract entities (fighters/teams) and targets
    from each row, while the base class handles the chronological loop,
    feature source orchestration, and A/B randomization.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.sources: list[FeatureSource] = []
        self.stateful_sources: list[StatefulFeatureSource] = []

    def register_source(self, source: FeatureSource) -> None:
        """Register a feature source. Stateful sources are tracked separately."""
        self.sources.append(source)
        if isinstance(source, StatefulFeatureSource):
            self.stateful_sources.append(source)

    def process_historical(self, events_df: pd.DataFrame,
                           randomize_sides: bool = True,
                           random_seed: int = 42) -> pd.DataFrame:
        """Process all historical events chronologically to build feature matrix.

        CRITICAL: Features are computed using only data available BEFORE each event.
        Stateful sources update AFTER feature extraction to prevent leakage.

        Args:
            events_df: DataFrame with event data (must have date_column)
            randomize_sides: Randomize A/B assignment to prevent ordering leakage
            random_seed: Seed for reproducible randomization

        Returns:
            Feature matrix DataFrame with targets
        """
        rng = random.Random(random_seed)
        events_sorted = events_df.sort_values(self.date_column).reset_index(drop=True)
        all_features = []

        for idx, row in events_sorted.iterrows():
            entity_a, entity_b = self.get_entities(row)

            # Randomize A/B to prevent winner-first ordering leakage
            if randomize_sides and rng.random() < 0.5:
                entity_a, entity_b = entity_b, entity_a

            event_date = str(row.get(self.date_column, ""))

            # Extract features from all sources
            features: dict[str, float] = {}
            for source in self.sources:
                try:
                    source_feats = source.get_features(
                        entity_a, entity_b, event_date, row=row
                    )
                    features.update(source_feats)
                except Exception as e:
                    logger.warning(f"Feature source {type(source).__name__} failed: {e}")
                    features.update(source.empty_features())

            # Add targets and metadata
            targets = self.get_targets(row, entity_a, entity_b)
            features.update(targets)
            features[self.date_column] = event_date
            features["entity_a"] = entity_a
            features["entity_b"] = entity_b

            all_features.append(features)

            # Update stateful sources AFTER feature extraction
            self.update_sources(row)

        df = pd.DataFrame(all_features)
        logger.info(f"Generated feature matrix: {df.shape[0]} events, {df.shape[1]} features")
        return df

    @property
    @abstractmethod
    def date_column(self) -> str:
        """Column name containing the event date."""
        ...

    @abstractmethod
    def get_entities(self, row: pd.Series) -> tuple[str, str]:
        """Extract the two competing entities from a row.

        Returns:
            (entity_a, entity_b) — e.g., (fighter_a, fighter_b) or (home_team, away_team)
        """
        ...

    @abstractmethod
    def get_targets(self, row: pd.Series,
                    entity_a: str, entity_b: str) -> dict[str, Any]:
        """Extract target variables from a row.

        Returns:
            Dict of target columns — e.g., {"target_winner_is_a": 1, "target_method": 0}
        """
        ...

    @abstractmethod
    def update_sources(self, row: pd.Series) -> None:
        """Update all stateful feature sources after an event outcome."""
        ...
