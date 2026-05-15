"""Soccer prediction pipeline — orchestrates training and prediction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sports_predict.base import BaseSportsPipeline, GenericEnsemble
from .features.pipeline import SoccerFeaturePipeline

logger = logging.getLogger(__name__)


class SoccerPipeline(BaseSportsPipeline):
    """End-to-end soccer prediction pipeline.

    Handles data loading, feature engineering, model training,
    and match prediction for international football (World Cup, Euros).
    """

    def __init__(self, config_path: str | None = None):
        config = self._load_config(config_path)
        super().__init__(config)

        feat_cfg = config.get("features", {})
        self.feature_pipeline = SoccerFeaturePipeline(feat_cfg)

        model_cfg = config.get("models", {})
        self.ensemble = GenericEnsemble(
            task="multiclass",
            n_classes=3,
            class_labels=["Home Win", "Draw", "Away Win"],
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
                "elo": {"k_factor": 40, "initial_rating": 1500, "home_advantage": 100},
            },
            "models": {
                "split": {"train_end": "2018-06-01", "val_end": "2022-06-01"},
            },
        }

    def load_data(self, path: str | Path) -> pd.DataFrame:
        """Load match data from CSV or Parquet."""
        path = Path(path)
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

        # Ensure date column
        if "match_date" not in df.columns:
            for col in ["date", "Date", "match_date"]:
                if col in df.columns:
                    df["match_date"] = pd.to_datetime(df[col], errors="coerce")
                    break

        if "match_date" in df.columns:
            df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
            df = df.sort_values("match_date").reset_index(drop=True)

        logger.info(f"Loaded {len(df)} matches from {path}")
        return df

    def prepare_features(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """Build feature matrix from match data."""
        return self.feature_pipeline.process_historical(matches_df)

    def train(self, feature_df: pd.DataFrame) -> dict[str, Any]:
        """Train the ensemble on the feature matrix."""
        train, val, test = self.split_data(
            feature_df, date_column="match_date",
            train_end="2018-06-01", val_end="2022-06-01",
        )

        self.feature_names = self.get_feature_cols(feature_df)
        target_col = "target_result"

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
        preds = np.argmax(probs, axis=1)
        accuracy = (preds == y_test.values).mean()

        results = {
            "test_accuracy": accuracy,
            "test_size": len(test),
            "n_features": len(self.feature_names),
        }
        logger.info(f"Test accuracy: {accuracy:.1%} on {len(test)} matches")
        return results

    def predict_matchup(self, home_team: str, away_team: str,
                        context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Predict a single match outcome."""
        ctx = context or {}
        match_date = ctx.get("date", "")

        features = self.feature_pipeline.get_features_for_matchup(
            home_team, away_team, match_date, **ctx
        )

        feat_df = pd.DataFrame([features]).reindex(
            columns=self.feature_names, fill_value=0
        ).fillna(0)

        probs = self.ensemble.predict_proba(feat_df)[0]

        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_win_prob": float(probs[0]),
            "draw_prob": float(probs[1]),
            "away_win_prob": float(probs[2]),
            "predicted_outcome": ["Home Win", "Draw", "Away Win"][np.argmax(probs)],
        }
