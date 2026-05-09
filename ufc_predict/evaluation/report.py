"""Full evaluation report generation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .metrics import FightMetrics
from .calibration import CalibrationAnalysis
from .roi import ROISimulator

logger = logging.getLogger(__name__)


class EvaluationReport:
    """Generate comprehensive evaluation report for the prediction system."""

    def __init__(self, output_dir: str | Path = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, y_true: np.ndarray, predictions: dict[str, np.ndarray],
                 closing_probs: np.ndarray | None = None,
                 method_true: np.ndarray | None = None,
                 method_pred: np.ndarray | None = None,
                 duration_true: np.ndarray | None = None,
                 duration_pred: np.ndarray | None = None,
                 feature_importance: pd.DataFrame | None = None) -> dict[str, Any]:
        """Generate full evaluation report with all metrics and plots."""
        report: dict[str, Any] = {}

        # Winner prediction metrics
        winner_prob = predictions.get("winner", predictions.get("prediction"))
        if winner_prob is not None:
            report["winner_metrics"] = FightMetrics.winner_metrics(
                y_true, winner_prob, closing_probs
            )
            report["baselines"] = FightMetrics.baseline_metrics(y_true, closing_probs)

            # Calibration
            report["calibration"] = CalibrationAnalysis.reliability_by_confidence(
                y_true, winner_prob
            ).to_dict("records")

            CalibrationAnalysis.plot_calibration(
                y_true, winner_prob, "Ensemble",
                closing_probs,
                save_path=self.output_dir / "calibration.png"
            )
            plt.close()

            # ROI simulation
            if closing_probs is not None:
                valid = ~np.isnan(closing_probs)
                if valid.sum() > 50:
                    roi_sim = ROISimulator(bet_size=100, edge_threshold=0.05)
                    roi_results = roi_sim.simulate(
                        y_true[valid], winner_prob[valid], closing_probs[valid]
                    )
                    report["roi"] = {k: v for k, v in roi_results.items() if k != "results_df"}
                    roi_sim.plot_equity_curve(
                        roi_results, save_path=self.output_dir / "equity_curve.png"
                    )
                    plt.close()

        # Method prediction metrics
        if method_true is not None and method_pred is not None:
            report["method_metrics"] = FightMetrics.method_metrics(method_true, method_pred)
            y_pred_class = method_pred.argmax(axis=1)
            CalibrationAnalysis.plot_confusion_matrix(
                method_true, y_pred_class,
                labels=["KO/TKO", "Submission", "Decision"],
                save_path=self.output_dir / "method_confusion.png"
            )
            plt.close()

        # Duration regression metrics
        if duration_true is not None and duration_pred is not None:
            report["duration_metrics"] = FightMetrics.duration_metrics(
                duration_true, duration_pred
            )

        # Feature importance with SHAP grouping
        if feature_importance is not None:
            report["top_features"] = feature_importance.head(30).to_dict("records")
            report["feature_tier_importance"] = self._group_by_tier(feature_importance)

        # Summary
        report["summary"] = self._generate_summary(report)

        # Save report
        self._save_report(report)

        return report

    def _group_by_tier(self, importance_df: pd.DataFrame) -> dict[str, float]:
        """Group feature importance by feature tier."""
        tier_map = {
            "elo": "T1: Ratings",
            "glicko": "T1: Ratings",
            "closing_prob": "T1: Market",
            "opening_prob": "T1: Market",
            "line_move": "T1: Market",
            "age": "T1: Age",
            "ewm_": "T2: Stats",
            "streak": "T2: Form",
            "style": "T2: Matchup",
            "reach": "T2: Physical",
            "height": "T2: Physical",
            "stance": "T2: Physical",
            "td_advantage": "T2: Matchup",
            "layoff": "T3: Context",
            "weight": "T3: Context",
            "camp": "T3: Context",
            "altitude": "T4: Venue",
            "ref_": "T4: Referee",
        }

        tier_importance: dict[str, float] = {}
        feat_col = "feature"
        imp_col = "importance_combined" if "importance_combined" in importance_df.columns else "importance"

        for _, row in importance_df.iterrows():
            feat = row[feat_col]
            imp = row[imp_col]
            assigned = False
            for pattern, tier in tier_map.items():
                if pattern in feat:
                    tier_importance[tier] = tier_importance.get(tier, 0) + imp
                    assigned = True
                    break
            if not assigned:
                tier_importance["Other"] = tier_importance.get("Other", 0) + imp

        return dict(sorted(tier_importance.items(), key=lambda x: x[1], reverse=True))

    def _generate_summary(self, report: dict) -> str:
        """Generate a text summary of the evaluation."""
        lines = ["=" * 60, "UFC FIGHT PREDICTION - EVALUATION REPORT", "=" * 60, ""]

        if "winner_metrics" in report:
            m = report["winner_metrics"]
            lines.append(f"WINNER PREDICTION:")
            lines.append(f"  Log Loss:     {m['log_loss']:.4f}")
            lines.append(f"  Brier Score:  {m['brier_score']:.4f}")
            lines.append(f"  Accuracy:     {m['accuracy']:.1%}")
            lines.append(f"  ECE:          {m['ece']:.4f}")
            lines.append(f"  Samples:      {m['n_samples']}")

            if "model_vs_closing_line" in m:
                edge = m["model_vs_closing_line"]
                lines.append(f"  vs Closing:   {'BEATS' if edge > 0 else 'LOSES TO'} "
                             f"closing line by {abs(edge):.4f} log loss")
            lines.append("")

        if "baselines" in report:
            lines.append("BASELINES:")
            for name, metrics in report["baselines"].items():
                desc = metrics.get("description", name)
                ll = metrics.get("log_loss", "N/A")
                acc = metrics.get("accuracy", "N/A")
                lines.append(f"  {name}: LL={ll}, Acc={acc}")
            lines.append("")

        if "roi" in report:
            r = report["roi"]
            lines.append("ROI SIMULATION (flat $100 bet, >5% edge):")
            lines.append(f"  Total Bets:   {r['total_bets']}")
            lines.append(f"  ROI:          {r['roi_pct']:.1f}%")
            lines.append(f"  Win Rate:     {r['win_rate']:.1%}")
            lines.append(f"  Total P&L:    ${r['total_pnl']:.0f}")
            lines.append(f"  Max Drawdown: ${r['max_drawdown']:.0f}")
            lines.append("")

        if "method_metrics" in report:
            m = report["method_metrics"]
            lines.append(f"METHOD OF VICTORY:")
            lines.append(f"  Accuracy:     {m['accuracy']:.1%}")
            lines.append(f"  Log Loss:     {m['log_loss']:.4f}")
            lines.append("")

        if "duration_metrics" in report:
            m = report["duration_metrics"]
            lines.append(f"FIGHT DURATION:")
            lines.append(f"  RMSE:         {m['rmse']:.0f}s")
            lines.append(f"  MAE:          {m['mae']:.0f}s")
            lines.append(f"  Within 60s:   {m['within_60s']:.1%}")
            lines.append(f"  Within 120s:  {m['within_120s']:.1%}")
            lines.append("")

        return "\n".join(lines)

    def _save_report(self, report: dict) -> None:
        """Save report summary to file."""
        summary = report.get("summary", "")
        with open(self.output_dir / "evaluation_report.txt", "w") as f:
            f.write(summary)
        logger.info(f"Report saved to {self.output_dir}")
