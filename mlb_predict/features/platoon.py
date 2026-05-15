"""Platoon split features (L/R matchup advantages) for MLB."""

from __future__ import annotations
from typing import Any
from sports_predict.base.feature_source import FeatureSource


class PlatoonFeatures(FeatureSource):
    """Platoon splits: left-handed vs right-handed matchup advantages.

    Right-handed batters hit better against left-handed pitchers and vice versa.
    A lineup stacked with same-hand batters against the starter is disadvantaged.
    """

    def get_features(self, entity_a: str, entity_b: str,
                     event_date: str, **context) -> dict[str, float]:
        row = context.get("row")
        if row is None:
            return self.empty_features()

        # Starter handedness (L=0, R=1)
        sp_hand_home = 1 if str(row.get("sp_hand_home", "R")).upper() == "R" else 0
        sp_hand_away = 1 if str(row.get("sp_hand_away", "R")).upper() == "R" else 0

        # Lineup composition vs starter
        # lineup_rhb_pct = fraction of lineup batting right-handed
        lineup_rhb_home = float(row.get("lineup_rhb_pct_home", 0.55))
        lineup_rhb_away = float(row.get("lineup_rhb_pct_away", 0.55))

        # Platoon advantage: same-hand starter vs opposite-hand heavy lineup
        # Positive = advantage for home team
        home_platoon = (1 - sp_hand_away) * lineup_rhb_home + sp_hand_away * (1 - lineup_rhb_home)
        away_platoon = (1 - sp_hand_home) * lineup_rhb_away + sp_hand_home * (1 - lineup_rhb_away)

        return {
            "sp_hand_home": sp_hand_home,
            "sp_hand_away": sp_hand_away,
            "lineup_rhb_pct_home": lineup_rhb_home,
            "lineup_rhb_pct_away": lineup_rhb_away,
            "platoon_adv_home": home_platoon,
            "platoon_adv_away": away_platoon,
            "platoon_diff": home_platoon - away_platoon,
        }

    def empty_features(self) -> dict[str, float]:
        return {
            "sp_hand_home": 1, "sp_hand_away": 1,
            "lineup_rhb_pct_home": 0.55, "lineup_rhb_pct_away": 0.55,
            "platoon_adv_home": 0.5, "platoon_adv_away": 0.5, "platoon_diff": 0,
        }
