"""MLB prediction pipeline — orchestrates training and prediction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sports_predict.base import BaseSportsPipeline, GenericEnsemble
from .features.pipeline import MLBFeaturePipeline

logger = logging.getLogger(__name__)


class MLBPipeline(BaseSportsPipeline):
    """End-to-end MLB prediction pipeline.

    Handles data loading, feature engineering, model training,
    and game prediction for Major League Baseball.
    """

    def __init__(self, config_path: str | None = None):
        config = self._load_config(config_path)
        super().__init__(config)

        feat_cfg = config.get("features", {})
        self.feature_pipeline = MLBFeaturePipeline(feat_cfg)

        model_cfg = config.get("models", {})
        self.ensemble = GenericEnsemble(
            task="binary",
            config=model_cfg,
        )

    def _load_config(self, config_path: str | None) -> dict:
        """Load YAML config or return defaults."""
        if config_path and Path(config_path).exists():
            import yaml
            with open(config_path) as f:
                return yaml.safe_load(f) or {}
        return {
            "features": {
                "elo": {"k_factor": 4, "initial_rating": 1500, "home_advantage": 24},
            },
            "models": {
                "split": {"train_end": "2023-01-01", "val_end": "2024-06-01"},
            },
        }

    def load_data(self, path: str | Path) -> pd.DataFrame:
        """Load game data from CSV or Parquet."""
        path = Path(path)
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

        if "game_date" not in df.columns:
            for col in ["date", "Date", "game_date"]:
                if col in df.columns:
                    df["game_date"] = pd.to_datetime(df[col], errors="coerce")
                    break

        if "game_date" in df.columns:
            df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
            df = df.sort_values("game_date").reset_index(drop=True)

        logger.info(f"Loaded {len(df)} games from {path}")
        return df

    def prepare_features(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Build feature matrix from game data."""
        return self.feature_pipeline.process_historical(games_df)

    def train(self, feature_df: pd.DataFrame) -> dict[str, Any]:
        """Train the ensemble on the feature matrix."""
        train, val, test = self.split_data(
            feature_df, date_column="game_date",
            train_end="2023-01-01", val_end="2024-06-01",
        )

        self.feature_names = self.get_feature_cols(feature_df)
        target_col = "target_home_win"

        X_train = train[self.feature_names].fillna(0)
        y_train = train[target_col]
        X_val = val[self.feature_names].fillna(0)
        y_val = val[target_col]
        X_test = test[self.feature_names].fillna(0)
        y_test = test[target_col]

        self.ensemble.train(X_train, y_train, X_val, y_val)
        self.is_trained = True

        # Evaluate on test
        probs = self.ensemble.predict_proba(X_test)
        if probs.ndim == 2:
            pred_probs = probs[:, 1]
        else:
            pred_probs = probs
        preds = (pred_probs >= 0.5).astype(int)
        accuracy = (preds == y_test.values).mean()

        results = {
            "test_accuracy": accuracy,
            "test_size": len(test),
            "n_features": len(self.feature_names),
        }
        logger.info(f"Test accuracy: {accuracy:.1%} on {len(test)} games")
        return results

    def predict_matchup(self, home_team: str, away_team: str,
                        context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Predict a single game outcome."""
        ctx = context or {}
        game_date = ctx.get("date", "")

        features = self.feature_pipeline.get_features_for_matchup(
            home_team, away_team, game_date, **ctx
        )

        feat_df = pd.DataFrame([features]).reindex(
            columns=self.feature_names, fill_value=0
        ).fillna(0)

        probs = self.ensemble.predict_proba(feat_df)
        if probs.ndim == 2:
            home_win_prob = float(probs[0, 1])
        else:
            home_win_prob = float(probs[0])

        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_win_prob": home_win_prob,
            "away_win_prob": 1.0 - home_win_prob,
            "predicted_winner": home_team if home_win_prob >= 0.5 else away_team,
        }
