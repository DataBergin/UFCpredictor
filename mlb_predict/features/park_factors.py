"""Park factor features for MLB prediction."""

from __future__ import annotations
from typing import Any
from sports_predict.base.feature_source import FeatureSource


# Park factors relative to league average (1.0 = neutral)
# Source: FanGraphs park factors. Updated periodically.
DEFAULT_PARK_FACTORS = {
    "COL": {"runs": 1.15, "hr": 1.12},  # Coors Field
    "BOS": {"runs": 1.06, "hr": 1.08},  # Fenway Park
    "CIN": {"runs": 1.05, "hr": 1.10},  # Great American
    "TEX": {"runs": 1.04, "hr": 1.06},  # Globe Life
    "NYY": {"runs": 1.02, "hr": 1.15},  # Yankee Stadium
    "SFG": {"runs": 0.93, "hr": 0.85},  # Oracle Park
    "OAK": {"runs": 0.94, "hr": 0.88},  # Oakland Coliseum
    "MIA": {"runs": 0.95, "hr": 0.90},  # loanDepot Park
    "STL": {"runs": 0.97, "hr": 0.95},  # Busch Stadium
}


class ParkFactors(FeatureSource):
    """Park factor adjustments: run scoring environment of the home ballpark.

    Coors Field inflates offense by ~15%, Oracle Park suppresses it by ~7%.
    """

    def __init__(self, park_factors: dict | None = None):
        self.park_factors = park_factors or DEFAULT_PARK_FACTORS

    def get_features(self, entity_a: str, entity_b: str,
                     event_date: str, **context) -> dict[str, float]:
        row = context.get("row")
        park = row.get("park_id", entity_a) if row is not None else entity_a

        factors = self.park_factors.get(park, {"runs": 1.0, "hr": 1.0})

        return {
            "park_run_factor": factors.get("runs", 1.0),
            "park_hr_factor": factors.get("hr", 1.0),
            "park_is_hitter_friendly": 1 if factors.get("runs", 1.0) > 1.03 else 0,
            "park_is_pitcher_friendly": 1 if factors.get("runs", 1.0) < 0.97 else 0,
        }

    def empty_features(self) -> dict[str, float]:
        return {
            "park_run_factor": 1.0, "park_hr_factor": 1.0,
            "park_is_hitter_friendly": 0, "park_is_pitcher_friendly": 0,
        }
