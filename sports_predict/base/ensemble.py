"""Sport-generic stacked ensemble model."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


class GenericEnsemble:
    """Stacked ensemble that combines multiple base model predictions.

    Supports binary (UFC, MLB) and three-way (soccer) classification.
    Base models produce probability predictions on a validation set,
    which are stacked as features for a meta-learner.

    If no base models are registered, automatically creates default models
    (LightGBM + LogisticRegression) on first train() call.

    Args:
        task: "binary" (2 outcomes) or "multiclass" (3+ outcomes)
        n_classes: Number of output classes (2 for binary, 3 for soccer W/D/L)
        class_labels: Optional human-readable class labels
        meta_method: "logistic" or "isotonic" for the meta-learner
        config: Optional model configuration dict
    """

    def __init__(self, task: str = "binary", n_classes: int = 2,
                 class_labels: list[str] | None = None,
                 meta_method: str = "logistic",
                 config: dict[str, Any] | None = None):
        self.task = task
        self.n_classes = n_classes
        self.class_labels = class_labels or [str(i) for i in range(n_classes)]
        self.meta_method = meta_method
        self.config = config or {}
        self.base_models: dict[str, Any] = {}
        self.meta_model = None
        self.is_trained = False

    def register_base_model(self, name: str, model: Any) -> None:
        """Register a base model. Must have fit(), predict_proba() methods."""
        self.base_models[name] = model

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame, y_val: pd.Series,
            **fit_kwargs) -> None:
        """Train all base models and the meta-learner.

        1. Train each base model on (X_train, y_train)
        2. Get predictions on X_val from each base model
        3. Stack predictions as features for meta-learner
        4. Train meta-learner on stacked validation predictions
        """
        # Train base models
        for name, model in self.base_models.items():
            logger.info(f"Training base model: {name}")
            try:
                model.fit(X_train, y_train, X_val, y_val, **fit_kwargs.get(name, {}))
            except TypeError:
                model.fit(X_train, y_train)

        # Get validation predictions for meta-learner
        meta_features = self._get_meta_features(X_val, **fit_kwargs)

        # Train meta-learner
        if self.task == "binary":
            if self.meta_method == "logistic":
                self.meta_model = LogisticRegression(max_iter=1000)
                self.meta_model.fit(meta_features, y_val)
            else:
                self.meta_model = IsotonicRegression(out_of_bounds="clip")
                # For isotonic, use mean of base predictions
                self.meta_model.fit(meta_features.mean(axis=1), y_val)
        else:
            # Multiclass: logistic regression with softmax
            self.meta_model = LogisticRegression(
                max_iter=1000, multi_class="multinomial"
            )
            self.meta_model.fit(meta_features, y_val)

        self.is_trained = True
        logger.info(f"Meta-learner trained on {len(y_val)} validation samples")

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Get final ensemble predictions."""
        meta_features = self._get_meta_features(X, **kwargs)

        if self.task == "binary":
            if self.meta_method == "logistic":
                return self.meta_model.predict_proba(meta_features)[:, 1]
            else:
                return self.meta_model.transform(meta_features.mean(axis=1))
        else:
            return self.meta_model.predict_proba(meta_features)

    def _get_meta_features(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Stack base model predictions into meta-features."""
        preds = []
        for name, model in self.base_models.items():
            try:
                if hasattr(model, "predict_proba"):
                    pred = model.predict_proba(X)
                else:
                    pred = model.predict(X)

                if pred.ndim == 1:
                    preds.append(pred.reshape(-1, 1))
                else:
                    preds.append(pred)
            except Exception as e:
                logger.warning(f"Base model {name} prediction failed: {e}")
                # Fill with 0.5 (uninformative)
                if self.task == "binary":
                    preds.append(np.full((len(X), 1), 0.5))
                else:
                    preds.append(np.full((len(X), self.n_classes), 1.0 / self.n_classes))

        return np.hstack(preds)

    def _ensure_base_models(self, X: pd.DataFrame) -> None:
        """Create default base models if none are registered."""
        if self.base_models:
            return

        logger.info("No base models registered, creating defaults")

        # LightGBM
        try:
            import lightgbm as lgb
            if self.task == "binary":
                self.base_models["lgbm"] = lgb.LGBMClassifier(
                    n_estimators=200, learning_rate=0.05, max_depth=6,
                    num_leaves=31, verbose=-1,
                )
            else:
                self.base_models["lgbm"] = lgb.LGBMClassifier(
                    n_estimators=200, learning_rate=0.05, max_depth=6,
                    num_leaves=31, verbose=-1, objective="multiclass",
                    num_class=self.n_classes,
                )
        except ImportError:
            logger.warning("LightGBM not available, using sklearn only")

        # Logistic Regression
        if self.task == "binary":
            self.base_models["lr"] = LogisticRegression(
                max_iter=1000, C=1.0,
            )
        else:
            self.base_models["lr"] = LogisticRegression(
                max_iter=1000, C=1.0, multi_class="multinomial",
            )

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame, y_val: pd.Series,
              **fit_kwargs) -> None:
        """Train the ensemble (alias for fit with auto-model creation)."""
        self._ensure_base_models(X_train)
        self.fit(X_train, y_train, X_val, y_val, **fit_kwargs)

    def predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Get class probabilities from the ensemble.

        Returns:
            For binary: (n_samples, 2) array
            For multiclass: (n_samples, n_classes) array
        """
        meta_features = self._get_meta_features(X, **kwargs)

        if self.task == "binary":
            if self.meta_method == "logistic":
                return self.meta_model.predict_proba(meta_features)
            else:
                probs_1 = self.meta_model.transform(meta_features.mean(axis=1))
                return np.column_stack([1 - probs_1, probs_1])
        else:
            return self.meta_model.predict_proba(meta_features)

    def feature_importance_combined(self) -> pd.DataFrame:
        """Aggregate feature importance across base models that support it."""
        all_importance = []
        for name, model in self.base_models.items():
            if hasattr(model, "feature_importance"):
                imp = model.feature_importance()
                if not imp.empty:
                    imp["model"] = name
                    all_importance.append(imp)

        if not all_importance:
            return pd.DataFrame()

        combined = pd.concat(all_importance)
        avg = combined.groupby("feature")["importance"].mean().reset_index()
        return avg.sort_values("importance", ascending=False)
