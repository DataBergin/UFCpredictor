"""LightGBM and XGBoost models for fight outcome prediction."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


class LGBMFightModel:
    """LightGBM model for fight prediction with monotonic constraints."""

    def __init__(self, task: str = "winner", params: dict[str, Any] | None = None):
        """
        Args:
            task: 'winner' (binary), 'method' (multiclass), 'round' (multiclass), 'duration' (regression)
            params: LightGBM parameters override
        """
        self.task = task
        self.model = None
        self.calibrator = None
        self.feature_names: list[str] = []

        default_params = {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_depth": 6,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "verbose": -1,
        }
        if params:
            default_params.update(params)
        self.params = default_params

    def _get_objective(self) -> dict[str, Any]:
        if self.task == "winner":
            return {"objective": "binary", "metric": "binary_logloss"}
        elif self.task in ("method", "round"):
            n_classes = 3 if self.task == "method" else 6
            return {"objective": "multiclass", "num_class": n_classes, "metric": "multi_logloss"}
        elif self.task == "duration":
            return {"objective": "regression", "metric": "rmse"}
        return {}

    def _get_monotone_constraints(self, feature_names: list[str]) -> list[int]:
        """Set monotonic constraints: higher Elo should increase win probability."""
        constraints = []
        positive_features = ["elo_diff", "elo_a", "glicko_diff", "closing_prob_a", "elo_div_diff"]
        negative_features = ["elo_b", "glicko_b", "closing_prob_b"]

        for feat in feature_names:
            if any(pf in feat for pf in positive_features) and self.task == "winner":
                constraints.append(1)
            elif any(nf in feat for nf in negative_features) and self.task == "winner":
                constraints.append(-1)
            else:
                constraints.append(0)
        return constraints

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame | None = None, y_val: pd.Series | None = None,
            early_stopping_rounds: int = 50) -> None:
        """Train the LightGBM model."""
        self.feature_names = list(X_train.columns)

        obj_params = self._get_objective()
        all_params = {**self.params, **obj_params}

        if self.task == "winner":
            all_params["monotone_constraints"] = self._get_monotone_constraints(self.feature_names)

        callbacks = [lgb.log_evaluation(50)]
        if X_val is not None and early_stopping_rounds:
            callbacks.append(lgb.early_stopping(early_stopping_rounds))

        if self.task == "duration":
            self.model = lgb.LGBMRegressor(**all_params)
        elif self.task in ("method", "round"):
            self.model = lgb.LGBMClassifier(**all_params)
        else:
            self.model = lgb.LGBMClassifier(**all_params)

        fit_params: dict[str, Any] = {"callbacks": callbacks}
        if X_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]

        self.model.fit(X_train, y_train, **fit_params)
        logger.info(f"LGBM {self.task} model trained. Best iteration: {self.model.best_iteration_}")

    def calibrate(self, X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """Calibrate probabilities using isotonic regression on validation set."""
        if self.task == "duration":
            return

        raw_probs = self.predict_proba(X_val)
        if self.task == "winner":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(raw_probs, y_val)
        else:
            self.calibrator = []
            for c in range(raw_probs.shape[1]):
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(raw_probs[:, c], (y_val == c).astype(int))
                self.calibrator.append(iso)

        logger.info(f"LGBM {self.task} model calibrated on {len(y_val)} samples")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get raw predicted probabilities."""
        if self.task == "duration":
            return self.model.predict(X)

        probs = self.model.predict_proba(X)
        if self.task == "winner":
            return probs[:, 1] if probs.ndim == 2 else probs
        return probs

    def predict_calibrated(self, X: pd.DataFrame) -> np.ndarray:
        """Get calibrated predicted probabilities."""
        raw = self.predict_proba(X)
        if self.calibrator is None:
            return raw

        if self.task == "winner":
            return self.calibrator.transform(raw)
        else:
            calibrated = np.zeros_like(raw)
            for c, iso in enumerate(self.calibrator):
                calibrated[:, c] = iso.transform(raw[:, c])
            row_sums = calibrated.sum(axis=1, keepdims=True)
            calibrated /= np.maximum(row_sums, 1e-8)
            return calibrated

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Get final predictions."""
        if self.task == "duration":
            return self.model.predict(X)
        return self.predict_calibrated(X)

    def feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if self.model is None:
            return pd.DataFrame()
        importance = self.model.feature_importances_
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False)


class XGBFightModel:
    """XGBoost model — same interface as LGBM for ensemble compatibility."""

    def __init__(self, task: str = "winner", params: dict[str, Any] | None = None):
        self.task = task
        self.model = None
        self.calibrator = None
        self.feature_names: list[str] = []

        default_params = {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "verbosity": 0,
            "tree_method": "hist",
        }
        if params:
            default_params.update(params)
        self.params = default_params

    def _get_objective(self) -> dict[str, str]:
        if self.task == "winner":
            return {"objective": "binary:logistic", "eval_metric": "logloss"}
        elif self.task in ("method", "round"):
            n_classes = 3 if self.task == "method" else 6
            return {"objective": "multi:softprob", "eval_metric": "mlogloss",
                    "num_class": str(n_classes)}
        elif self.task == "duration":
            return {"objective": "reg:squarederror", "eval_metric": "rmse"}
        return {}

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame | None = None, y_val: pd.Series | None = None,
            early_stopping_rounds: int = 50) -> None:
        """Train XGBoost model."""
        self.feature_names = list(X_train.columns)

        obj_params = self._get_objective()
        all_params = {**self.params, **obj_params}

        if self.task == "duration":
            self.model = xgb.XGBRegressor(**all_params)
        else:
            self.model = xgb.XGBClassifier(**all_params)

        fit_params: dict[str, Any] = {}
        if X_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = False

        self.model.fit(X_train, y_train, **fit_params)
        logger.info(f"XGB {self.task} model trained")

    def calibrate(self, X_val: pd.DataFrame, y_val: pd.Series) -> None:
        if self.task == "duration":
            return
        raw_probs = self.predict_proba(X_val)
        if self.task == "winner":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(raw_probs, y_val)
        else:
            self.calibrator = []
            for c in range(raw_probs.shape[1]):
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(raw_probs[:, c], (y_val == c).astype(int))
                self.calibrator.append(iso)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task == "duration":
            return self.model.predict(X)
        probs = self.model.predict_proba(X)
        if self.task == "winner":
            return probs[:, 1] if probs.ndim == 2 else probs
        return probs

    def predict_calibrated(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.predict_proba(X)
        if self.calibrator is None:
            return raw
        if self.task == "winner":
            return self.calibrator.transform(raw)
        else:
            calibrated = np.zeros_like(raw)
            for c, iso in enumerate(self.calibrator):
                calibrated[:, c] = iso.transform(raw[:, c])
            row_sums = calibrated.sum(axis=1, keepdims=True)
            calibrated /= np.maximum(row_sums, 1e-8)
            return calibrated

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.task == "duration":
            return self.model.predict(X)
        return self.predict_calibrated(X)

    def feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            return pd.DataFrame()
        importance = self.model.feature_importances_
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False)
