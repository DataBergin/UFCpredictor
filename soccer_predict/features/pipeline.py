"""Soccer feature pipeline — World Cup and international football."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from sports_predict.base import BaseFeaturePipeline, GenericEloSystem
from .team_elo import SoccerEloSystem
from .squad import SquadFeatures
from .tournament import TournamentFeatures
from .international import InternationalFeatures

logger = logging.getLogger(__name__)


class SoccerFeaturePipeline(BaseFeaturePipeline):
    """Feature pipeline for international soccer (World Cup, Euros, etc.).

    Key differences from UFC:
    - Team-based entities instead of individuals
    - Three-way outcome (win/draw/loss)
    - Home advantage matters significantly
    - Squad composition changes between tournaments
    """

    date_column = "match_date"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        cfg = config or {}
        elo_cfg = cfg.get("elo", {})

        # Register feature sources
        self.elo = SoccerEloSystem(
            k_factor=elo_cfg.get("k_factor", 40),
            initial_rating=elo_cfg.get("initial_rating", 1500),
            home_advantage=elo_cfg.get("home_advantage", 100),
        )
        self.register_source(self.elo)
        self.register_source(SquadFeatures())
        self.register_source(TournamentFeatures())
        self.register_source(InternationalFeatures())

        # RAG (opt-in)
        rag_cfg = cfg.get("rag", {})
        if rag_cfg.get("enabled", False):
            from ufc_predict.features.rag_features import RAGFeatures
            rag_cfg["sport"] = "soccer"
            self.register_source(RAGFeatures(rag_cfg))

    def get_entities(self, row: pd.Series) -> tuple[str, str]:
        return row["home_team"], row["away_team"]

    def get_targets(self, row: pd.Series,
                    entity_a: str, entity_b: str) -> dict[str, Any]:
        """Three-way target: 0=home win, 1=draw, 2=away win."""
        home_goals = int(row.get("home_goals", 0))
        away_goals = int(row.get("away_goals", 0))

        if home_goals > away_goals:
            result = 0  # home win
        elif home_goals == away_goals:
            result = 1  # draw
        else:
            result = 2  # away win

        return {
            "target_result": result,
            "target_home_goals": home_goals,
            "target_away_goals": away_goals,
            "target_total_goals": home_goals + away_goals,
        }

    def update_sources(self, row: pd.Series) -> None:
        """Update all stateful sources after a match."""
        home = row["home_team"]
        away = row["away_team"]
        home_goals = int(row.get("home_goals", 0))
        away_goals = int(row.get("away_goals", 0))

        # Elo update: 1.0 for win, 0.5 for draw, 0.0 for loss
        if home_goals > away_goals:
            score_a, score_b = 1.0, 0.0
        elif home_goals == away_goals:
            score_a, score_b = 0.5, 0.5
        else:
            score_a, score_b = 0.0, 1.0

        result = {
            "entity_a": home, "entity_b": away,
            "score_a": score_a, "score_b": score_b,
            "home_goals": home_goals, "away_goals": away_goals,
        }
        for source in self.stateful_sources:
            source.update(result)
