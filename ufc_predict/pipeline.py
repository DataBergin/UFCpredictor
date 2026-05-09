"""Main training and prediction pipeline orchestration."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap

from .config import get_config
from .features.pipeline import FeaturePipeline
from .models.ensemble import StackedEnsemble
from .models.gradient_boost import LGBMFightModel
from .evaluation.report import EvaluationReport

logger = logging.getLogger(__name__)


class UFCPipeline:
    """Full training and prediction pipeline."""

    def __init__(self, config_path: str | Path | None = None):
        self.config = get_config(config_path)
        self.feature_pipeline = FeaturePipeline(self.config.get("features", {}))

        model_cfg = self.config.get("models", {})
        self.winner_model = StackedEnsemble(
            task="winner", config=model_cfg,
            method=model_cfg.get("meta", {}).get("method", "logistic")
        )
        self.method_model = LGBMFightModel(task="method", params=model_cfg.get("lgbm", {}))
        self.round_model = LGBMFightModel(task="round", params=model_cfg.get("lgbm", {}))
        self.duration_model = LGBMFightModel(task="duration", params=model_cfg.get("lgbm", {}))

        self.feature_names: list[str] = []
        self.is_trained = False

    def load_data(self, fights_path: str | Path) -> pd.DataFrame:
        """Load fight data from CSV/parquet."""
        from .utils import fight_duration_seconds, parse_date

        path = Path(fights_path)
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

        # Compute duration_seconds from round + time if missing
        if "duration_seconds" not in df.columns and "round" in df.columns and "time" in df.columns:
            df["duration_seconds"] = df.apply(
                lambda r: fight_duration_seconds(
                    int(r["round"]) if pd.notna(r["round"]) and str(r["round"]).isdigit() else None,
                    str(r["time"]) if pd.notna(r["time"]) else ""
                ), axis=1
            )

        # Ensure date column exists
        if "date" not in df.columns and "event_date" in df.columns:
            df["date"] = df["event_date"].apply(parse_date)

        return df

    def prepare_features(self, fights_df: pd.DataFrame,
                         fighter_info: dict[str, dict] | None = None) -> pd.DataFrame:
        """Run feature engineering pipeline on fight data."""
        if fighter_info:
            for fighter, info in fighter_info.items():
                self.feature_pipeline.set_fighter_info(fighter, info)

        feature_df = self.feature_pipeline.process_historical_fights(fights_df)

        # Leakage check
        self.feature_pipeline.verify_no_leakage(feature_df)

        return feature_df

    def split_data(self, feature_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Time-based train/val/test split."""
        split_cfg = self.config.get("models", {}).get("split", {})
        train_end = pd.Timestamp(split_cfg.get("train_end", "2022-01-01"))
        val_end = pd.Timestamp(split_cfg.get("val_end", "2024-01-01"))

        dates = pd.to_datetime(feature_df["fight_date"], errors="coerce")

        # If dates couldn't be parsed, fall back to positional split
        valid_dates = dates.notna().sum()
        if valid_dates < len(feature_df) * 0.5:
            logger.warning(f"Only {valid_dates}/{len(feature_df)} fights have valid dates. "
                           f"Using positional 60/20/20 split instead.")
            n = len(feature_df)
            train = feature_df.iloc[:int(n * 0.6)].copy()
            val = feature_df.iloc[int(n * 0.6):int(n * 0.8)].copy()
            test = feature_df.iloc[int(n * 0.8):].copy()
        else:
            train = feature_df[dates < train_end].copy()
            val = feature_df[(dates >= train_end) & (dates < val_end)].copy()
            test = feature_df[dates >= val_end].copy()

        logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
        return train, val, test

    def get_feature_cols(self, df: pd.DataFrame) -> list[str]:
        """Get feature columns (exclude targets and metadata)."""
        exclude = {"target_winner_is_a", "target_method", "target_round",
                   "target_duration_seconds", "fight_date", "fighter_a",
                   "fighter_b", "fight_idx"}
        cols = [c for c in df.columns if c not in exclude]
        # Drop any non-numeric columns
        numeric_cols = df[cols].select_dtypes(include=[np.number]).columns.tolist()
        return numeric_cols

    def train(self, feature_df: pd.DataFrame) -> dict[str, Any]:
        """Train all models on the prepared feature matrix."""
        train_df, val_df, test_df = self.split_data(feature_df)
        self.feature_names = self.get_feature_cols(train_df)

        X_train = train_df[self.feature_names].fillna(0)
        X_val = val_df[self.feature_names].fillna(0)
        X_test = test_df[self.feature_names].fillna(0)

        # Winner model (stacked ensemble)
        logger.info("Training winner prediction model...")
        y_train_w = train_df["target_winner_is_a"]
        y_val_w = val_df["target_winner_is_a"]

        self.winner_model.fit(
            X_train, y_train_w, X_val, y_val_w,
            train_df["fighter_a"], train_df["fighter_b"],
            val_df["fighter_a"], val_df["fighter_b"],
        )

        # Method model
        logger.info("Training method of victory model...")
        y_train_m = train_df["target_method"].dropna().astype(int)
        valid_m_train = y_train_m.index
        self.method_model.fit(
            X_train.loc[valid_m_train], y_train_m,
            X_val, val_df["target_method"].fillna(2).astype(int),
        )

        # Round model (encode as 0-5: R1, R2, R3, R4, R5, Decision)
        logger.info("Training round prediction model...")
        y_train_r = self._encode_round(train_df["target_round"])
        y_val_r = self._encode_round(val_df["target_round"])
        valid_r = y_train_r.notna()
        if valid_r.sum() > 100:
            self.round_model.fit(
                X_train[valid_r], y_train_r[valid_r].astype(int),
                X_val, y_val_r.fillna(5).astype(int),
            )

        # Duration model
        logger.info("Training duration regression model...")
        y_train_d = train_df["target_duration_seconds"].dropna()
        valid_d = y_train_d.index
        if len(valid_d) > 100:
            self.duration_model.fit(
                X_train.loc[valid_d], y_train_d,
                X_val, val_df["target_duration_seconds"].fillna(val_df["target_duration_seconds"].median()),
            )

        self.is_trained = True

        # Evaluate on test set
        results = self.evaluate(X_test, test_df)
        return results

    def evaluate(self, X_test: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Any]:
        """Evaluate all models on test set."""
        evaluator = EvaluationReport(output_dir="reports")

        # Winner predictions
        winner_preds = self.winner_model.predict(
            X_test, test_df["fighter_a"], test_df["fighter_b"]
        )

        # Method predictions
        method_preds = self.method_model.predict(X_test)

        # Duration predictions
        if self.duration_model.model is not None:
            duration_preds = self.duration_model.predict(X_test)
        else:
            duration_preds = None
            logger.warning("Duration model was not trained — skipping duration evaluation")

        # Closing line for comparison
        closing_probs = test_df.get("closing_prob_a")
        if closing_probs is not None:
            closing_probs = closing_probs.values

        report = evaluator.generate(
            y_true=test_df["target_winner_is_a"].values,
            predictions={"winner": winner_preds},
            closing_probs=closing_probs,
            method_true=test_df["target_method"].fillna(2).astype(int).values,
            method_pred=method_preds if method_preds.ndim == 2 else None,
            duration_true=test_df["target_duration_seconds"].values if duration_preds is not None else None,
            duration_pred=duration_preds,
            feature_importance=self.winner_model.feature_importance_combined(),
        )

        print(report.get("summary", ""))
        return report

    def predict_fight(self, fighter_a: str, fighter_b: str,
                      fight_info: dict[str, Any] | None = None,
                      closing_line: float | None = None) -> dict[str, Any]:
        """Predict a single fight. Returns full prediction breakdown."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")

        fight_info = fight_info or {}
        features = self.feature_pipeline.predict_features(fighter_a, fighter_b, fight_info)

        X = pd.DataFrame([features]).reindex(columns=self.feature_names, fill_value=0).fillna(0)
        fighters_a = pd.Series([fighter_a])
        fighters_b = pd.Series([fighter_b])

        # Winner prediction with uncertainty
        winner_result = self.winner_model.predict_with_uncertainty(X, fighters_a, fighters_b)
        win_prob_a = float(winner_result["prediction"][0])

        # Method prediction
        method_probs = self.method_model.predict(X)
        if method_probs.ndim == 2:
            method_dist = {
                "KO/TKO": float(method_probs[0, 0]),
                "Submission": float(method_probs[0, 1]),
                "Decision": float(method_probs[0, 2]),
            }
        else:
            method_dist = {"KO/TKO": 0.33, "Submission": 0.33, "Decision": 0.34}

        # Round prediction
        round_probs = self.round_model.predict(X)
        if round_probs.ndim == 2 and round_probs.shape[1] >= 4:
            round_labels = ["R1", "R2", "R3", "R4", "R5", "Decision"]
            round_dist = {
                round_labels[i]: float(round_probs[0, i])
                for i in range(min(len(round_labels), round_probs.shape[1]))
            }
        else:
            round_dist = {"R1": 0.15, "R2": 0.15, "R3": 0.20, "Decision": 0.50}

        # Duration prediction
        if self.duration_model.model is not None:
            expected_duration = float(self.duration_model.predict(X)[0])
        else:
            expected_duration = 900.0

        # SHAP analysis for explainability
        shap_drivers = self._get_shap_drivers(X)

        # Edge vs closing line
        edge = None
        if closing_line:
            edge = win_prob_a - closing_line

        result = {
            "fighter_a": fighter_a,
            "fighter_b": fighter_b,
            "win_prob_a": win_prob_a,
            "win_prob_b": 1 - win_prob_a,
            "ci_90_lower": float(winner_result["ci_lower"][0]),
            "ci_90_upper": float(winner_result["ci_upper"][0]),
            "method_distribution": method_dist,
            "round_distribution": round_dist,
            "expected_duration_seconds": expected_duration,
            "expected_duration_display": self._format_duration(expected_duration),
            "model_agreement": {
                "lgbm": float(winner_result["lgbm"][0]),
                "xgb": float(winner_result["xgb"][0]),
                "neural": float(winner_result["neural"][0]),
            },
            "top_shap_drivers": shap_drivers,
            "edge_vs_line": edge,
            "predicted_winner": fighter_a if win_prob_a > 0.5 else fighter_b,
        }

        return result

    def _get_shap_drivers(self, X: pd.DataFrame, top_n: int = 5) -> list[dict[str, Any]]:
        """Get top SHAP drivers for the prediction."""
        try:
            lgbm_model = self.winner_model.lgbm.model
            if lgbm_model is None:
                return []
            explainer = shap.TreeExplainer(lgbm_model)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            importance = np.abs(shap_values[0])
            top_idx = np.argsort(importance)[-top_n:][::-1]

            drivers = []
            for idx in top_idx:
                feat_name = self.feature_names[idx]
                drivers.append({
                    "feature": feat_name,
                    "shap_value": float(shap_values[0][idx]),
                    "feature_value": float(X.iloc[0, idx]),
                    "direction": "+" if shap_values[0][idx] > 0 else "-",
                })
            return drivers
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            return []

    def _encode_round(self, round_series: pd.Series) -> pd.Series:
        """Encode round to integer: R1=0, R2=1, R3=2, R4=3, R5=4, Decision=5."""
        def encode(val):
            if pd.isna(val):
                return np.nan
            val_str = str(val).strip()
            if val_str.isdigit():
                r = int(val_str)
                return min(r - 1, 5)
            return 5
        return round_series.apply(encode)

    def _format_duration(self, seconds: float) -> str:
        """Format duration seconds to human readable."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"

    def save(self, path: str | Path = "models/pipeline.pkl") -> None:
        """Save trained pipeline to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Pipeline saved to {path}")

    @staticmethod
    def load(path: str | Path = "models/pipeline.pkl") -> "UFCPipeline":
        """Load trained pipeline from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)
