"""Calibration analysis and plotting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class CalibrationAnalysis:
    """Analyze and plot model calibration."""

    @staticmethod
    def calibration_curve(y_true: np.ndarray, y_prob: np.ndarray,
                          n_bins: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute calibration curve data."""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_accuracies = []
        bin_counts = []

        for i in range(n_bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_centers.append(y_prob[mask].mean())
                bin_accuracies.append(y_true[mask].mean())
                bin_counts.append(mask.sum())

        return np.array(bin_centers), np.array(bin_accuracies), np.array(bin_counts)

    @staticmethod
    def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray,
                         model_name: str = "Model",
                         closing_line_prob: np.ndarray | None = None,
                         save_path: str | Path | None = None) -> plt.Figure:
        """Plot calibration curve with optional closing line comparison."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Calibration curve
        centers, accs, counts = CalibrationAnalysis.calibration_curve(y_true, y_prob)
        ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax1.plot(centers, accs, "b-o", label=model_name, markersize=8)

        if closing_line_prob is not None:
            valid = ~np.isnan(closing_line_prob)
            if valid.sum() > 0:
                cl_centers, cl_accs, _ = CalibrationAnalysis.calibration_curve(
                    y_true[valid], closing_line_prob[valid]
                )
                ax1.plot(cl_centers, cl_accs, "r-s", label="Closing Line", markersize=6)

        ax1.set_xlabel("Predicted Probability")
        ax1.set_ylabel("Actual Win Rate")
        ax1.set_title("Calibration Curve")
        ax1.legend()
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=0.3)

        # Prediction distribution
        ax2.hist(y_prob, bins=30, alpha=0.7, color="steelblue", edgecolor="black")
        ax2.axvline(0.5, color="red", linestyle="--", alpha=0.7)
        ax2.set_xlabel("Predicted Probability (Fighter A wins)")
        ax2.set_ylabel("Count")
        ax2.set_title("Prediction Distribution")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                              labels: list[str] | None = None,
                              save_path: str | Path | None = None) -> plt.Figure:
        """Plot confusion matrix."""
        from sklearn.metrics import confusion_matrix as cm

        matrix = cm(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=labels or "auto",
                    yticklabels=labels or "auto")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    @staticmethod
    def reliability_by_confidence(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
        """Break down accuracy by prediction confidence bands."""
        bands = [(0.5, 0.55), (0.55, 0.6), (0.6, 0.65), (0.65, 0.7),
                 (0.7, 0.75), (0.75, 0.8), (0.8, 0.85), (0.85, 0.9), (0.9, 1.0)]

        rows = []
        prob_favored = np.maximum(y_prob, 1 - y_prob)
        correct = ((y_prob >= 0.5) == y_true).astype(int)

        for low, high in bands:
            mask = (prob_favored >= low) & (prob_favored < high)
            if mask.sum() > 0:
                rows.append({
                    "confidence_band": f"{low:.0%}-{high:.0%}",
                    "n_fights": int(mask.sum()),
                    "accuracy": correct[mask].mean(),
                    "expected_accuracy": prob_favored[mask].mean(),
                    "calibration_gap": correct[mask].mean() - prob_favored[mask].mean(),
                })

        return pd.DataFrame(rows)
