"""International football context features."""

from __future__ import annotations
from typing import Any
from sports_predict.base.feature_source import FeatureSource


class InternationalFeatures(FeatureSource):
    """International match context: FIFA ranking, travel, confederation, h2h."""

    def get_features(self, entity_a: str, entity_b: str,
                     event_date: str, **context) -> dict[str, float]:
        row = context.get("row")
        if row is None:
            return self.empty_features()

        fifa_a = float(row.get("fifa_rank_a", 50))
        fifa_b = float(row.get("fifa_rank_b", 50))

        return {
            "fifa_rank_a": fifa_a,
            "fifa_rank_b": fifa_b,
            "fifa_rank_diff": fifa_a - fifa_b,
            "fifa_rank_ratio": fifa_a / max(fifa_b, 1),
            "travel_distance_km": float(row.get("travel_distance_km", 0)),
            "altitude_diff_m": float(row.get("altitude_diff_m", 0)),
            "is_neutral_venue": int(row.get("is_neutral_venue", 0)),
            "same_confederation": int(row.get("same_confederation", 0)),
            "h2h_wins_a": float(row.get("h2h_wins_a", 0)),
            "h2h_wins_b": float(row.get("h2h_wins_b", 0)),
            "h2h_draws": float(row.get("h2h_draws", 0)),
        }

    def empty_features(self) -> dict[str, float]:
        return {
            "fifa_rank_a": 50, "fifa_rank_b": 50,
            "fifa_rank_diff": 0, "fifa_rank_ratio": 1.0,
            "travel_distance_km": 0, "altitude_diff_m": 0,
            "is_neutral_venue": 0, "same_confederation": 0,
            "h2h_wins_a": 0, "h2h_wins_b": 0, "h2h_draws": 0,
        }
