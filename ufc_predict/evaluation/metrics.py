"""Core evaluation metrics for fight prediction models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    log_loss, brier_score_loss, accuracy_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error,
)


class FightMetrics:
    """Compute all evaluation metrics for fight predictions."""

    @staticmethod
    def winner_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                       closing_line_prob: np.ndarray | None = None) -> dict[str, float]:
        """Compute metrics for winner prediction task."""
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = {
            "log_loss": log_loss(y_true, y_prob),
            "brier_score": brier_score_loss(y_true, y_prob),
            "accuracy": accuracy_score(y_true, y_pred),
            "n_samples": len(y_true),
        }

        # Calibration metrics
        metrics["ece"] = FightMetrics._expected_calibration_error(y_true, y_prob)

        # Confidence-stratified accuracy
        high_conf = y_prob >= 0.7
        if high_conf.sum() > 0:
            metrics["accuracy_high_conf"] = accuracy_score(y_true[high_conf], y_pred[high_conf])
            metrics["n_high_conf"] = int(high_conf.sum())

        low_conf = (y_prob >= 0.4) & (y_prob <= 0.6)
        if low_conf.sum() > 0:
            metrics["accuracy_toss_up"] = accuracy_score(y_true[low_conf], y_pred[low_conf])

        # Beat the closing line?
        if closing_line_prob is not None:
            valid = ~np.isnan(closing_line_prob)
            if valid.sum() > 0:
                cl_log_loss = log_loss(y_true[valid], closing_line_prob[valid])
                model_log_loss = log_loss(y_true[valid], y_prob[valid])
                metrics["closing_line_log_loss"] = cl_log_loss
                metrics["model_vs_closing_line"] = cl_log_loss - model_log_loss
                metrics["beats_closing_line"] = model_log_loss < cl_log_loss

        return metrics

    @staticmethod
    def method_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
        """Metrics for method of victory prediction."""
        y_pred = y_prob.argmax(axis=1)
        method_names = ["KO/TKO", "Submission", "Decision"]

        metrics = {
            "log_loss": log_loss(y_true, y_prob),
            "accuracy": accuracy_score(y_true, y_pred),
        }

        for i, name in enumerate(method_names):
            mask = y_true == i
            if mask.sum() > 0:
                metrics[f"accuracy_{name}"] = accuracy_score(
                    (y_true == i).astype(int),
                    (y_pred == i).astype(int)
                )
                metrics[f"n_{name}"] = int(mask.sum())

        return metrics

    @staticmethod
    def round_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
        """Metrics for round of finish prediction."""
        y_pred = y_prob.argmax(axis=1)
        return {
            "log_loss": log_loss(y_true, y_prob),
            "accuracy": accuracy_score(y_true, y_pred),
            "within_1_round": float(np.abs(y_true - y_pred).mean() <= 1),
        }

    @staticmethod
    def duration_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """Metrics for fight duration regression."""
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "median_ae": float(np.median(np.abs(y_true - y_pred))),
            "within_60s": float(np.mean(np.abs(y_true - y_pred) <= 60)),
            "within_120s": float(np.mean(np.abs(y_true - y_pred) <= 120)),
        }

    @staticmethod
    def _expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                                    n_bins: int = 10) -> float:
        """Expected Calibration Error."""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            ece += mask.sum() * abs(bin_acc - bin_conf)
        return ece / len(y_true)

    @staticmethod
    def baseline_metrics(y_true: np.ndarray, closing_probs: np.ndarray | None = None) -> dict[str, dict]:
        """Compute baseline benchmarks to compare against."""
        baselines = {}

        # Always-pick-favorite baseline
        baselines["always_favorite"] = {
            "accuracy": max(y_true.mean(), 1 - y_true.mean()),
            "description": "Always predict the favorite wins",
        }

        # Coin flip
        baselines["coin_flip"] = {
            "log_loss": log_loss(y_true, np.full_like(y_true, 0.5, dtype=float)),
            "accuracy": 0.5,
        }

        # Closing line (the real benchmark)
        if closing_probs is not None:
            valid = ~np.isnan(closing_probs)
            if valid.sum() > 0:
                cl_pred = (closing_probs[valid] >= 0.5).astype(int)
                baselines["closing_line"] = {
                    "log_loss": log_loss(y_true[valid], closing_probs[valid]),
                    "brier_score": brier_score_loss(y_true[valid], closing_probs[valid]),
                    "accuracy": accuracy_score(y_true[valid], cl_pred),
                    "ece": FightMetrics._expected_calibration_error(y_true[valid], closing_probs[valid]),
                }

        return baselines
