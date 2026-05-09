"""Fighter statistics features with recency weighting."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FighterStatsFeatures:
    """Compute rolling, recency-weighted fighter statistics features."""

    STAT_COLS = [
        "sig_str_landed_pm", "sig_str_accuracy", "sig_str_absorbed_pm",
        "sig_str_defense", "td_avg", "td_accuracy", "td_defense",
        "sub_avg", "ctrl_time_sec", "knockdowns_landed", "knockdowns_absorbed",
        "head_str_pct", "body_str_pct", "leg_str_pct", "distance_str_pct",
        "clinch_str_pct", "ground_str_pct",
    ]

    def __init__(self, decay_halflife: int = 5,
                 windows: list[int] | None = None,
                 time_windows_months: list[int] | None = None):
        self.decay_halflife = decay_halflife
        self.windows = windows or [3, 5, 10]
        self.time_windows_months = time_windows_months or [6, 12, 24]
        self.fighter_history: dict[str, list[dict]] = {}

    def add_fight(self, fighter: str, stats: dict[str, Any], fight_date: str = "") -> None:
        """Record a fight's stats for a fighter."""
        if fighter not in self.fighter_history:
            self.fighter_history[fighter] = []
        record = {**stats, "date": fight_date}
        self.fighter_history[fighter].append(record)

    def get_features(self, fighter: str, as_of_date: str = "") -> dict[str, float]:
        """Compute all stat features for a fighter using only fights before as_of_date."""
        history = self.fighter_history.get(fighter, [])
        if as_of_date:
            history = [h for h in history if str(h.get("date", "")) < as_of_date]

        if not history:
            return self._empty_features(fighter)

        features: dict[str, float] = {}
        features["n_fights"] = len(history)

        # Exponentially weighted averages
        exp_features = self._exponential_weighted_stats(history)
        features.update(exp_features)

        # Window-based rolling averages
        for window in self.windows:
            window_features = self._window_stats(history, window)
            features.update({f"{k}_last{window}": v for k, v in window_features.items()})

        # Win/loss streak
        features.update(self._streak_features(history))

        # Performance trend
        features.update(self._trend_features(history))

        # Finish rate features
        features.update(self._finish_features(history))

        # Durability features
        features.update(self._durability_features(history))

        return features

    def _exponential_weighted_stats(self, history: list[dict]) -> dict[str, float]:
        """Compute exponentially decay-weighted statistics."""
        features = {}
        n = len(history)
        weights = np.array([2 ** (-(n - 1 - i) / self.decay_halflife) for i in range(n)])
        weights /= weights.sum()

        for col in self.STAT_COLS:
            values = np.array([h.get(col, 0.0) for h in history], dtype=float)
            valid = ~np.isnan(values)
            if valid.any():
                w = weights[valid]
                w /= w.sum()
                features[f"ewm_{col}"] = float(np.average(values[valid], weights=w))
            else:
                features[f"ewm_{col}"] = 0.0

        return features

    def _window_stats(self, history: list[dict], window: int) -> dict[str, float]:
        """Compute average stats over last N fights."""
        recent = history[-window:]
        features = {}
        for col in self.STAT_COLS:
            values = [h.get(col, 0.0) for h in recent if h.get(col) is not None]
            features[col] = np.mean(values) if values else 0.0
        return features

    def _streak_features(self, history: list[dict]) -> dict[str, float]:
        """Win/loss streak and quality-adjusted form."""
        streak = 0
        streak_type = None

        for h in reversed(history):
            result = h.get("result", "").lower()
            if "win" in result or result == "w":
                if streak_type == "win" or streak_type is None:
                    streak += 1
                    streak_type = "win"
                else:
                    break
            elif "loss" in result or result == "l":
                if streak_type == "loss" or streak_type is None:
                    streak -= 1
                    streak_type = "loss"
                else:
                    break
            else:
                break

        return {
            "win_streak": max(streak, 0),
            "loss_streak": abs(min(streak, 0)),
            "streak_value": streak,
        }

    def _trend_features(self, history: list[dict]) -> dict[str, float]:
        """Is fighter improving or declining?"""
        if len(history) < 4:
            return {"perf_trend": 0.0, "output_trend": 0.0}

        recent_half = history[len(history) // 2:]
        earlier_half = history[:len(history) // 2]

        def avg_stat(fights, col):
            vals = [f.get(col, 0) for f in fights if f.get(col) is not None]
            return np.mean(vals) if vals else 0.0

        recent_output = avg_stat(recent_half, "sig_str_landed_pm")
        earlier_output = avg_stat(earlier_half, "sig_str_landed_pm")
        output_trend = recent_output - earlier_output

        recent_wins = sum(1 for f in recent_half if "win" in str(f.get("result", "")).lower())
        earlier_wins = sum(1 for f in earlier_half if "win" in str(f.get("result", "")).lower())
        perf_trend = (recent_wins / max(len(recent_half), 1)) - (earlier_wins / max(len(earlier_half), 1))

        return {"perf_trend": perf_trend, "output_trend": output_trend}

    def _finish_features(self, history: list[dict]) -> dict[str, float]:
        """KO rate, submission rate, decision rate."""
        n = len(history)
        if n == 0:
            return {"ko_rate": 0.0, "sub_rate": 0.0, "dec_rate": 0.0, "finish_rate": 0.0}

        wins = [h for h in history if "win" in str(h.get("result", "")).lower()]
        n_wins = len(wins)

        ko_wins = sum(1 for h in wins if "ko" in str(h.get("method", "")).lower())
        sub_wins = sum(1 for h in wins if "sub" in str(h.get("method", "")).lower())
        dec_wins = sum(1 for h in wins if "dec" in str(h.get("method", "")).lower())

        return {
            "ko_rate": ko_wins / max(n_wins, 1),
            "sub_rate": sub_wins / max(n_wins, 1),
            "dec_rate": dec_wins / max(n_wins, 1),
            "finish_rate": (ko_wins + sub_wins) / max(n_wins, 1),
            "ko_wins": ko_wins,
            "sub_wins": sub_wins,
        }

    def _durability_features(self, history: list[dict]) -> dict[str, float]:
        """Chin durability, cut proneness, KO absorption."""
        losses = [h for h in history if "loss" in str(h.get("result", "")).lower()]
        n_losses = len(losses)

        ko_losses = sum(1 for h in losses if "ko" in str(h.get("method", "")).lower())
        sub_losses = sum(1 for h in losses if "sub" in str(h.get("method", "")).lower())

        total_kd_absorbed = sum(h.get("knockdowns_absorbed", 0) for h in history)

        last_ko_loss_idx = None
        for i, h in enumerate(reversed(history)):
            if "loss" in str(h.get("result", "")).lower() and "ko" in str(h.get("method", "")).lower():
                last_ko_loss_idx = i
                break

        return {
            "ko_loss_rate": ko_losses / max(n_losses, 1),
            "sub_loss_rate": sub_losses / max(n_losses, 1),
            "total_ko_losses": ko_losses,
            "total_kd_absorbed": total_kd_absorbed,
            "fights_since_ko_loss": last_ko_loss_idx if last_ko_loss_idx is not None else 99,
            "been_kod": 1 if ko_losses > 0 else 0,
        }

    def _empty_features(self, fighter: str) -> dict[str, float]:
        """Return zero-filled features for unknown fighters."""
        features = {"n_fights": 0}
        for col in self.STAT_COLS:
            features[f"ewm_{col}"] = 0.0
            for w in self.windows:
                features[f"{col}_last{w}"] = 0.0
        features.update({
            "win_streak": 0, "loss_streak": 0, "streak_value": 0,
            "perf_trend": 0.0, "output_trend": 0.0,
            "ko_rate": 0.0, "sub_rate": 0.0, "dec_rate": 0.0, "finish_rate": 0.0,
            "ko_wins": 0, "sub_wins": 0,
            "ko_loss_rate": 0.0, "sub_loss_rate": 0.0,
            "total_ko_losses": 0, "total_kd_absorbed": 0,
            "fights_since_ko_loss": 99, "been_kod": 0,
        })
        return features

    def get_differential_features(self, fighter_a: str, fighter_b: str,
                                  as_of_date: str = "") -> dict[str, float]:
        """Compute difference features between two fighters."""
        feats_a = self.get_features(fighter_a, as_of_date)
        feats_b = self.get_features(fighter_b, as_of_date)

        diff_features = {}
        for key in feats_a:
            diff_features[f"a_{key}"] = feats_a[key]
            diff_features[f"b_{key}"] = feats_b[key]
            if isinstance(feats_a[key], (int, float)) and isinstance(feats_b[key], (int, float)):
                diff_features[f"diff_{key}"] = feats_a[key] - feats_b[key]

        return diff_features
