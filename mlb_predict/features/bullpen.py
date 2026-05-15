"""Bullpen availability and workload features for MLB."""

from __future__ import annotations
from typing import Any
from sports_predict.base.feature_source import FeatureSource


class BullpenFeatures(FeatureSource):
    """Bullpen state: reliever availability based on recent usage.

    If the closer pitched 2 of the last 3 days, they're likely unavailable.
    Aggregate bullpen fatigue affects late-game win probability.
    """

    def get_features(self, entity_a: str, entity_b: str,
                     event_date: str, **context) -> dict[str, float]:
        row = context.get("row")
        if row is None:
            return self.empty_features()

        return {
            "bp_era_home": float(row.get("bp_era_home", 4.00)),
            "bp_era_away": float(row.get("bp_era_away", 4.00)),
            "bp_era_diff": float(row.get("bp_era_home", 4.0)) - float(row.get("bp_era_away", 4.0)),
            "bp_innings_last3d_home": float(row.get("bp_innings_last3d_home", 6)),
            "bp_innings_last3d_away": float(row.get("bp_innings_last3d_away", 6)),
            "closer_available_home": int(row.get("closer_available_home", 1)),
            "closer_available_away": int(row.get("closer_available_away", 1)),
            "bp_fatigue_home": float(row.get("bp_fatigue_home", 0)),
            "bp_fatigue_away": float(row.get("bp_fatigue_away", 0)),
        }

    def empty_features(self) -> dict[str, float]:
        return {
            "bp_era_home": 4.0, "bp_era_away": 4.0, "bp_era_diff": 0,
            "bp_innings_last3d_home": 6, "bp_innings_last3d_away": 6,
            "closer_available_home": 1, "closer_available_away": 1,
            "bp_fatigue_home": 0, "bp_fatigue_away": 0,
        }
