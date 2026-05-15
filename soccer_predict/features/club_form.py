"""Club form features for international soccer prediction.

Between international windows, the best signal for a national team's
current quality is how its players are performing at their clubs.
A France squad with Mbappe scoring 30 goals at PSG and Griezmann
creating 15 assists at Atletico is sharper than one where key players
are benched or injured at club level.

Features are computed per-side (home/away) with differentials.
"""

from __future__ import annotations

import logging
from typing import Any

from sports_predict.base.feature_source import FeatureSource

logger = logging.getLogger(__name__)


class ClubFormFeatures(FeatureSource):
    """Aggregate club-level performance of national team players.

    Requires a ClubFormAggregator to be initialized with player data.
    When aggregator is not available, returns neutral defaults.
    """

    def __init__(self, aggregator=None):
        """
        Args:
            aggregator: Optional ClubFormAggregator instance with loaded data.
                        If None, returns default features (model still works,
                        just without club form signal).
        """
        self.aggregator = aggregator

    def get_features(self, entity_a: str, entity_b: str,
                     event_date: str, **context) -> dict[str, float]:
        row = context.get("row")

        # If no aggregator loaded, check if club form data is in the row
        if self.aggregator is None:
            return self._from_row(row) if row is not None else self.empty_features()

        # Determine season from event_date (e.g. 2022-11-20 → "2022-2023")
        season = self._date_to_season(event_date)

        # Get club form for both teams
        home_form = self.aggregator.aggregate_for_nation(entity_a, season)
        away_form = self.aggregator.aggregate_for_nation(entity_b, season)

        features = {}

        # Per-side features
        for key, val in home_form.items():
            features[f"{key}_home"] = val
        for key, val in away_form.items():
            features[f"{key}_away"] = val

        # Key differentials
        features["club_goals_diff"] = home_form["club_goals_total"] - away_form["club_goals_total"]
        features["club_xg_diff"] = home_form["club_xg_total"] - away_form["club_xg_total"]
        features["club_league_pos_diff"] = away_form["club_avg_league_pos"] - home_form["club_avg_league_pos"]  # reversed: lower pos = better
        features["club_weighted_goals_diff"] = home_form["club_weighted_goals"] - away_form["club_weighted_goals"]
        features["club_top5_pct_diff"] = home_form["club_top5_league_pct"] - away_form["club_top5_league_pct"]

        return features

    def _from_row(self, row) -> dict[str, float]:
        """Extract pre-computed club form features from a data row."""
        features = {}
        for suffix in ("home", "away"):
            features[f"club_goals_total_{suffix}"] = float(row.get(f"club_goals_total_{suffix}", 0))
            features[f"club_assists_total_{suffix}"] = float(row.get(f"club_assists_total_{suffix}", 0))
            features[f"club_xg_total_{suffix}"] = float(row.get(f"club_xg_total_{suffix}", 0))
            features[f"club_xa_total_{suffix}"] = float(row.get(f"club_xa_total_{suffix}", 0))
            features[f"club_minutes_total_{suffix}"] = float(row.get(f"club_minutes_total_{suffix}", 0))
            features[f"club_goals_per90_{suffix}"] = float(row.get(f"club_goals_per90_{suffix}", 0))
            features[f"club_xg_per90_{suffix}"] = float(row.get(f"club_xg_per90_{suffix}", 0))
            features[f"club_avg_league_pos_{suffix}"] = float(row.get(f"club_avg_league_pos_{suffix}", 10))
            features[f"club_top5_league_pct_{suffix}"] = float(row.get(f"club_top5_league_pct_{suffix}", 0))
            features[f"club_avg_age_{suffix}"] = float(row.get(f"club_avg_age_{suffix}", 27))
            features[f"club_n_players_{suffix}"] = float(row.get(f"club_n_players_{suffix}", 0))
            features[f"club_weighted_goals_{suffix}"] = float(row.get(f"club_weighted_goals_{suffix}", 0))

        features["club_goals_diff"] = features["club_goals_total_home"] - features["club_goals_total_away"]
        features["club_xg_diff"] = features["club_xg_total_home"] - features["club_xg_total_away"]
        features["club_league_pos_diff"] = features["club_avg_league_pos_away"] - features["club_avg_league_pos_home"]
        features["club_weighted_goals_diff"] = features["club_weighted_goals_home"] - features["club_weighted_goals_away"]
        features["club_top5_pct_diff"] = features["club_top5_league_pct_home"] - features["club_top5_league_pct_away"]

        return features

    def empty_features(self) -> dict[str, float]:
        features = {}
        for suffix in ("home", "away"):
            features[f"club_goals_total_{suffix}"] = 0.0
            features[f"club_assists_total_{suffix}"] = 0.0
            features[f"club_xg_total_{suffix}"] = 0.0
            features[f"club_xa_total_{suffix}"] = 0.0
            features[f"club_minutes_total_{suffix}"] = 0.0
            features[f"club_goals_per90_{suffix}"] = 0.0
            features[f"club_xg_per90_{suffix}"] = 0.0
            features[f"club_avg_league_pos_{suffix}"] = 10.0
            features[f"club_top5_league_pct_{suffix}"] = 0.0
            features[f"club_avg_age_{suffix}"] = 27.0
            features[f"club_n_players_{suffix}"] = 0.0
            features[f"club_weighted_goals_{suffix}"] = 0.0

        features["club_goals_diff"] = 0.0
        features["club_xg_diff"] = 0.0
        features["club_league_pos_diff"] = 0.0
        features["club_weighted_goals_diff"] = 0.0
        features["club_top5_pct_diff"] = 0.0

        return features

    @staticmethod
    def _date_to_season(date_str: str) -> str:
        """Convert a date to a season string.

        European club seasons span Aug-May. A date in Nov 2022 → "2022-2023".
        A date in Mar 2023 → "2022-2023".
        """
        try:
            from datetime import datetime
            if not date_str:
                return "2023-2024"
            dt = datetime.fromisoformat(str(date_str)[:10])
            if dt.month >= 7:  # Aug-Dec → season starts this year
                return f"{dt.year}-{dt.year + 1}"
            else:  # Jan-Jun → season started last year
                return f"{dt.year - 1}-{dt.year}"
        except (ValueError, TypeError):
            return "2023-2024"
