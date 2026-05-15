"""MLB feature pipeline — game-level prediction features."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from sports_predict.base import BaseFeaturePipeline, GenericEloSystem
from .pitcher import PitcherFeatures
from .park_factors import ParkFactors
from .platoon import PlatoonFeatures
from .bullpen import BullpenFeatures

logger = logging.getLogger(__name__)


class MLBFeaturePipeline(BaseFeaturePipeline):
    """Feature pipeline for MLB game prediction.

    Key differences from UFC/soccer:
    - High game volume (162/season) → lower Elo K-factor
    - Pitcher-centric: starting pitcher is the strongest single feature
    - Park factors significantly affect run scoring
    - Platoon splits: L/R matchup advantages
    - Bullpen availability depends on recent usage
    """

    date_column = "game_date"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        cfg = config or {}
        elo_cfg = cfg.get("elo", {})

        # Team Elo with low K-factor for high-volume sport
        self.elo = GenericEloSystem(
            k_factor=elo_cfg.get("k_factor", 4),
            initial_rating=elo_cfg.get("initial_rating", 1500),
            home_advantage=elo_cfg.get("home_advantage", 24),
            feature_prefix="elo",
        )
        self.register_source(self.elo)
        self.register_source(PitcherFeatures())
        self.register_source(ParkFactors())
        self.register_source(PlatoonFeatures())
        self.register_source(BullpenFeatures())

        # RAG (opt-in)
        rag_cfg = cfg.get("rag", {})
        if rag_cfg.get("enabled", False):
            from ufc_predict.features.rag_features import RAGFeatures
            rag_cfg["sport"] = "mlb"
            self.register_source(RAGFeatures(rag_cfg))

    def get_entities(self, row: pd.Series) -> tuple[str, str]:
        return row["home_team"], row["away_team"]

    def get_targets(self, row: pd.Series,
                    entity_a: str, entity_b: str) -> dict[str, Any]:
        """Binary target: did entity_a (home) win?"""
        home_runs = int(row.get("home_runs", 0))
        away_runs = int(row.get("away_runs", 0))

        return {
            "target_home_win": int(home_runs > away_runs),
            "target_home_runs": home_runs,
            "target_away_runs": away_runs,
            "target_total_runs": home_runs + away_runs,
            "target_run_diff": home_runs - away_runs,
        }

    def update_sources(self, row: pd.Series) -> None:
        """Update stateful sources after a game."""
        home = row["home_team"]
        away = row["away_team"]
        home_runs = int(row.get("home_runs", 0))
        away_runs = int(row.get("away_runs", 0))

        if home_runs > away_runs:
            result = {"winner": home, "loser": away}
        else:
            result = {"winner": away, "loser": home}

        result.update({
            "home_team": home, "away_team": away,
            "home_runs": home_runs, "away_runs": away_runs,
        })

        for source in self.stateful_sources:
            source.update(result)
