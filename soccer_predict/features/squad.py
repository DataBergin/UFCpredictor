"""Squad composition features for soccer prediction."""

from __future__ import annotations
from typing import Any
from sports_predict.base.feature_source import FeatureSource


class SquadFeatures(FeatureSource):
    """Squad-level features: market value, age, depth, injuries.

    Requires squad data to be present in the row (from Transfermarkt or similar).
    Returns zeros when data is unavailable.
    """

    def get_features(self, entity_a: str, entity_b: str,
                     event_date: str, **context) -> dict[str, float]:
        row = context.get("row")
        if row is None:
            return self.empty_features()

        return {
            "squad_value_a": float(row.get("squad_value_a", 0)),
            "squad_value_b": float(row.get("squad_value_b", 0)),
            "squad_value_ratio": self._safe_ratio(
                row.get("squad_value_a", 0), row.get("squad_value_b", 0)
            ),
            "squad_avg_age_a": float(row.get("squad_avg_age_a", 27)),
            "squad_avg_age_b": float(row.get("squad_avg_age_b", 27)),
            "squad_injuries_a": int(row.get("squad_injuries_a", 0)),
            "squad_injuries_b": int(row.get("squad_injuries_b", 0)),
            "squad_depth_a": float(row.get("squad_depth_a", 0)),
            "squad_depth_b": float(row.get("squad_depth_b", 0)),
            "squad_cl_players_a": int(row.get("squad_cl_players_a", 0)),
            "squad_cl_players_b": int(row.get("squad_cl_players_b", 0)),
        }

    def empty_features(self) -> dict[str, float]:
        return {
            "squad_value_a": 0, "squad_value_b": 0, "squad_value_ratio": 1.0,
            "squad_avg_age_a": 27, "squad_avg_age_b": 27,
            "squad_injuries_a": 0, "squad_injuries_b": 0,
            "squad_depth_a": 0, "squad_depth_b": 0,
            "squad_cl_players_a": 0, "squad_cl_players_b": 0,
        }

    @staticmethod
    def _safe_ratio(a, b) -> float:
        a, b = float(a or 0), float(b or 0)
        if b == 0:
            return 1.0
        return a / b
