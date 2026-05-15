"""Abstract orchestration pipeline for training and prediction."""

from __future__ import annotations

import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseSportsPipeline(ABC):
    """Abstract pipeline that orchestrates data loading, feature engineering,
    training, and prediction for any sport.

    Subclasses implement sport-specific data loading, feature preparation,
    and prediction formatting.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.feature_names: list[str] = []
        self.is_trained = False

    @abstractmethod
    def load_data(self, path: str | Path) -> pd.DataFrame:
        """Load raw event data from disk."""
        ...

    @abstractmethod
    def prepare_features(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Run feature engineering on raw event data."""
        ...

    @abstractmethod
    def predict_matchup(self, entity_a: str, entity_b: str,
                        context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Predict a single matchup outcome."""
        ...

    def split_data(self, feature_df: pd.DataFrame,
                   date_column: str = "fight_date",
                   train_end: str = "2022-01-01",
                   val_end: str = "2024-01-01") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Time-based train/val/test split. Reusable across sports."""
        split_cfg = self.config.get("models", {}).get("split", {})
        train_end_ts = pd.Timestamp(split_cfg.get("train_end", train_end))
        val_end_ts = pd.Timestamp(split_cfg.get("val_end", val_end))

        dates = pd.to_datetime(feature_df[date_column], errors="coerce")

        valid_dates = dates.notna().sum()
        if valid_dates < len(feature_df) * 0.5:
            logger.warning(f"Only {valid_dates}/{len(feature_df)} valid dates. "
                           f"Using positional 60/20/20 split.")
            n = len(feature_df)
            train = feature_df.iloc[:int(n * 0.6)].copy()
            val = feature_df.iloc[int(n * 0.6):int(n * 0.8)].copy()
            test = feature_df.iloc[int(n * 0.8):].copy()
        else:
            train = feature_df[dates < train_end_ts].copy()
            val = feature_df[(dates >= train_end_ts) & (dates < val_end_ts)].copy()
            test = feature_df[dates >= val_end_ts].copy()

        logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
        return train, val, test

    def get_feature_cols(self, df: pd.DataFrame,
                         exclude_prefixes: set[str] | None = None) -> list[str]:
        """Get numeric feature columns, excluding targets and metadata."""
        default_exclude = {"target_", "entity_a", "entity_b", "fight_date",
                           "match_date", "game_date", "fighter_a", "fighter_b",
                           "home_team", "away_team", "fight_idx"}
        exclude = default_exclude | (exclude_prefixes or set())

        cols = [c for c in df.columns
                if not any(c.startswith(ex) if ex.endswith("_") else c == ex
                           for ex in exclude)]
        numeric_cols = df[cols].select_dtypes(include=[np.number]).columns.tolist()
        return numeric_cols

    def save(self, path: str | Path = "models/pipeline.pkl") -> None:
        """Save trained pipeline to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Pipeline saved to {path}")

    @classmethod
    def load(cls, path: str | Path = "models/pipeline.pkl") -> "BaseSportsPipeline":
        """Load trained pipeline from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)
