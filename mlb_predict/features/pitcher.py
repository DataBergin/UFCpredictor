"""Starting pitcher features for MLB prediction."""

from __future__ import annotations
from typing import Any
from sports_predict.base.feature_source import FeatureSource


class PitcherFeatures(FeatureSource):
    """Starting pitcher features: ERA, FIP, K/9, pitch mix, rest, etc.

    The starting pitcher is the single strongest predictor in MLB.
    Requires pitcher data in the row from Baseball Reference or FanGraphs.
    """

    def get_features(self, entity_a: str, entity_b: str,
                     event_date: str, **context) -> dict[str, float]:
        row = context.get("row")
        if row is None:
            return self.empty_features()

        return {
            # Home starter
            "sp_era_home": float(row.get("sp_era_home", 4.50)),
            "sp_fip_home": float(row.get("sp_fip_home", 4.50)),
            "sp_whip_home": float(row.get("sp_whip_home", 1.30)),
            "sp_k9_home": float(row.get("sp_k9_home", 8.0)),
            "sp_bb9_home": float(row.get("sp_bb9_home", 3.0)),
            "sp_days_rest_home": float(row.get("sp_days_rest_home", 5)),
            "sp_innings_last30_home": float(row.get("sp_innings_last30_home", 25)),
            # Away starter
            "sp_era_away": float(row.get("sp_era_away", 4.50)),
            "sp_fip_away": float(row.get("sp_fip_away", 4.50)),
            "sp_whip_away": float(row.get("sp_whip_away", 1.30)),
            "sp_k9_away": float(row.get("sp_k9_away", 8.0)),
            "sp_bb9_away": float(row.get("sp_bb9_away", 3.0)),
            "sp_days_rest_away": float(row.get("sp_days_rest_away", 5)),
            "sp_innings_last30_away": float(row.get("sp_innings_last30_away", 25)),
            # Differentials
            "sp_era_diff": float(row.get("sp_era_home", 4.50)) - float(row.get("sp_era_away", 4.50)),
            "sp_fip_diff": float(row.get("sp_fip_home", 4.50)) - float(row.get("sp_fip_away", 4.50)),
            "sp_k9_diff": float(row.get("sp_k9_home", 8.0)) - float(row.get("sp_k9_away", 8.0)),
        }

    def empty_features(self) -> dict[str, float]:
        return {
            "sp_era_home": 4.50, "sp_fip_home": 4.50, "sp_whip_home": 1.30,
            "sp_k9_home": 8.0, "sp_bb9_home": 3.0, "sp_days_rest_home": 5,
            "sp_innings_last30_home": 25,
            "sp_era_away": 4.50, "sp_fip_away": 4.50, "sp_whip_away": 1.30,
            "sp_k9_away": 8.0, "sp_bb9_away": 3.0, "sp_days_rest_away": 5,
            "sp_innings_last30_away": 25,
            "sp_era_diff": 0, "sp_fip_diff": 0, "sp_k9_diff": 0,
        }
