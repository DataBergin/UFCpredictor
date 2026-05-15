"""Tournament-specific features for World Cup / Euros prediction."""

from __future__ import annotations
from typing import Any
from sports_predict.base.feature_source import FeatureSource


class TournamentFeatures(FeatureSource):
    """Tournament context: stage, group dynamics, rest days, pressure."""

    STAGE_MAP = {
        "group": 0, "round_of_16": 1, "r16": 1, "quarter_final": 2,
        "qf": 2, "semi_final": 3, "sf": 3, "third_place": 4,
        "final": 5,
    }

    def get_features(self, entity_a: str, entity_b: str,
                     event_date: str, **context) -> dict[str, float]:
        row = context.get("row")
        if row is None:
            return self.empty_features()

        stage = str(row.get("stage", "group")).lower().replace(" ", "_")
        stage_code = self.STAGE_MAP.get(stage, 0)
        is_knockout = 1 if stage_code >= 1 else 0

        return {
            "tournament_stage": stage_code,
            "is_knockout": is_knockout,
            "is_final": 1 if stage_code == 5 else 0,
            "rest_days_a": float(row.get("rest_days_a", 3)),
            "rest_days_b": float(row.get("rest_days_b", 3)),
            "rest_diff": float(row.get("rest_days_a", 3)) - float(row.get("rest_days_b", 3)),
            "group_points_a": float(row.get("group_points_a", 0)),
            "group_points_b": float(row.get("group_points_b", 0)),
            "must_win_a": int(row.get("must_win_a", 0)),
            "must_win_b": int(row.get("must_win_b", 0)),
        }

    def empty_features(self) -> dict[str, float]:
        return {
            "tournament_stage": 0, "is_knockout": 0, "is_final": 0,
            "rest_days_a": 3, "rest_days_b": 3, "rest_diff": 0,
            "group_points_a": 0, "group_points_b": 0,
            "must_win_a": 0, "must_win_b": 0,
        }
