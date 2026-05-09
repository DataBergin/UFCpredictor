"""Stacked ensemble: combines GBM and neural net predictions via meta-learner."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

from .gradient_boost import LGBMFightModel, XGBFightModel
from .neural import FighterNeuralNet

logger = logging.getLogger(__name__)


class StackedEnsemble:
    """
    Level-2 meta-learner that stacks:
    - LightGBM predictions
    - XGBoost predictions
    - Neural net predictions (with fighter embeddings)

    Uses logistic regression or isotonic calibration as the combiner.
    """

    def __init__(self, task: str = "winner", method: str = "logistic",
                 config: dict[str, Any] | None = None):
        self.task = task
        self.method = method
        self.config = config or {}

        self.lgbm = LGBMFightModel(task=task, params=self.config.get("lgbm", {}))
        self.xgb = XGBFightModel(task=task, params=self.config.get("xgb", {}))
        self.neural = FighterNeuralNet(task=task, params=self.config.get("neural", {}))

        self.meta_model = None
        self.feature_names: list[str] = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame, y_val: pd.Series,
            fighters_a_train: pd.Series, fighters_b_train: pd.Series,
            fighters_a_val: pd.Series, fighters_b_val: pd.Series) -> None:
        """Train all base models and the meta-learner."""
        self.feature_names = list(X_train.columns)

        # Train base models
        logger.info("Training LightGBM base model...")
        self.lgbm.fit(X_train, y_train, X_val, y_val)

        logger.info("Training XGBoost base model...")
        self.xgb.fit(X_train, y_train, X_val, y_val)

        logger.info("Training Neural Net base model...")
        self.neural.fit(
            X_train, y_train, fighters_a_train, fighters_b_train,
            X_val, y_val, fighters_a_val, fighters_b_val
        )

        # Calibrate base models on validation set
        self.lgbm.calibrate(X_val, y_val)
        self.xgb.calibrate(X_val, y_val)
        self.neural.calibrate(X_val, y_val, fighters_a_val, fighters_b_val)

        # Generate meta-features from validation predictions
        meta_X = self._get_meta_features(X_val, fighters_a_val, fighters_b_val)

        # Train meta-learner
        logger.info(f"Training meta-learner ({self.method})...")
        if self.task == "winner":
            if self.method == "logistic":
                self.meta_model = LogisticRegression(C=1.0, max_iter=1000)
                self.meta_model.fit(meta_X, y_val)
            else:
                self.meta_model = IsotonicRegression(out_of_bounds="clip")
                avg_pred = meta_X.mean(axis=1)
                self.meta_model.fit(avg_pred, y_val)
        elif self.task == "duration":
            from sklearn.linear_model import Ridge
            self.meta_model = Ridge(alpha=1.0)
            self.meta_model.fit(meta_X, y_val)
        else:
            self.meta_model = LogisticRegression(C=1.0, max_iter=1000, multi_class="multinomial")
            self.meta_model.fit(meta_X, y_val)

        logger.info("Stacked ensemble training complete")

    def _get_meta_features(self, X: pd.DataFrame, fighters_a: pd.Series,
                           fighters_b: pd.Series) -> np.ndarray:
        """Generate meta-features from base model predictions."""
        lgbm_pred = self.lgbm.predict(X)
        xgb_pred = self.xgb.predict(X)
        neural_pred = self.neural.predict(X, fighters_a, fighters_b)

        if self.task == "winner" or self.task == "duration":
            return np.column_stack([lgbm_pred, xgb_pred, neural_pred])
        else:
            return np.column_stack([lgbm_pred, xgb_pred, neural_pred])

    def predict(self, X: pd.DataFrame, fighters_a: pd.Series,
                fighters_b: pd.Series) -> np.ndarray:
        """Get final ensemble predictions."""
        meta_X = self._get_meta_features(X, fighters_a, fighters_b)

        if self.task == "winner":
            if self.method == "logistic":
                return self.meta_model.predict_proba(meta_X)[:, 1]
            else:
                avg_pred = meta_X.mean(axis=1)
                return self.meta_model.transform(avg_pred)
        elif self.task == "duration":
            return self.meta_model.predict(meta_X)
        else:
            return self.meta_model.predict_proba(meta_X)

    def predict_with_uncertainty(self, X: pd.DataFrame, fighters_a: pd.Series,
                                 fighters_b: pd.Series,
                                 n_bootstrap: int = 100) -> dict[str, np.ndarray]:
        """Predict with bootstrap-based confidence intervals."""
        lgbm_pred = self.lgbm.predict(X)
        xgb_pred = self.xgb.predict(X)
        neural_pred = self.neural.predict(X, fighters_a, fighters_b)

        predictions = np.column_stack([lgbm_pred, xgb_pred, neural_pred])

        # Bootstrap confidence intervals from model disagreement
        mean_pred = predictions.mean(axis=1)
        std_pred = predictions.std(axis=1)

        # 90% CI assuming approximate normality
        ci_lower = np.clip(mean_pred - 1.645 * std_pred, 0, 1)
        ci_upper = np.clip(mean_pred + 1.645 * std_pred, 0, 1)

        # Final ensemble prediction
        final_pred = self.predict(X, fighters_a, fighters_b)

        return {
            "prediction": final_pred,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "std": std_pred,
            "lgbm": lgbm_pred,
            "xgb": xgb_pred,
            "neural": neural_pred,
        }

    def feature_importance_combined(self) -> pd.DataFrame:
        """Get combined feature importance from GBM models."""
        lgbm_imp = self.lgbm.feature_importance()
        xgb_imp = self.xgb.feature_importance()

        if lgbm_imp.empty:
            return xgb_imp

        merged = lgbm_imp.merge(xgb_imp, on="feature", suffixes=("_lgbm", "_xgb"))
        merged["importance_lgbm"] = merged["importance_lgbm"] / merged["importance_lgbm"].max()
        merged["importance_xgb"] = merged["importance_xgb"] / merged["importance_xgb"].max()
        merged["importance_combined"] = (merged["importance_lgbm"] + merged["importance_xgb"]) / 2
        return merged.sort_values("importance_combined", ascending=False)
