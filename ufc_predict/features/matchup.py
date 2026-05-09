"""Style matchup and physical comparison features."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STYLE_CLUSTERS = {
    "striker": {"sig_str_landed_pm": 1.5, "td_avg": -1.0, "sub_avg": -1.0},
    "wrestler": {"td_avg": 1.5, "ctrl_time_sec": 1.0, "sig_str_landed_pm": -0.5},
    "grappler": {"sub_avg": 1.5, "td_avg": 1.0, "sig_str_landed_pm": -1.0},
    "balanced": {"sig_str_landed_pm": 0.5, "td_avg": 0.5, "sub_avg": 0.5},
}

STANCE_EDGE = {
    ("southpaw", "orthodox"): 0.03,
    ("orthodox", "southpaw"): -0.03,
    ("switch", "orthodox"): 0.01,
    ("switch", "southpaw"): 0.01,
    ("orthodox", "orthodox"): 0.0,
    ("southpaw", "southpaw"): 0.0,
}


class MatchupFeatures:
    """Compute style matchup and physical advantage features."""

    def __init__(self):
        self.fighter_profiles: dict[str, dict[str, Any]] = {}

    def set_profile(self, fighter: str, profile: dict[str, Any]) -> None:
        self.fighter_profiles[fighter] = profile

    def classify_style(self, stats: dict[str, float]) -> tuple[str, dict[str, float]]:
        """Classify fighter style based on their stats."""
        scores = {}
        for style, weights in STYLE_CLUSTERS.items():
            score = 0.0
            for stat, weight in weights.items():
                val = stats.get(stat, 0.0)
                score += val * weight
            scores[style] = score

        best_style = max(scores, key=scores.get)
        return best_style, scores

    def get_features(self, fighter_a: str, fighter_b: str,
                     stats_a: dict[str, float], stats_b: dict[str, float]) -> dict[str, float]:
        """Compute matchup features between two fighters."""
        features: dict[str, float] = {}

        # Style classification
        style_a, scores_a = self.classify_style(stats_a)
        style_b, scores_b = self.classify_style(stats_b)

        features["style_a_striker"] = scores_a.get("striker", 0)
        features["style_a_wrestler"] = scores_a.get("wrestler", 0)
        features["style_a_grappler"] = scores_a.get("grappler", 0)
        features["style_b_striker"] = scores_b.get("striker", 0)
        features["style_b_wrestler"] = scores_b.get("wrestler", 0)
        features["style_b_grappler"] = scores_b.get("grappler", 0)

        # Style mismatch indicators
        features["striker_vs_grappler"] = float(
            style_a == "striker" and style_b in ("grappler", "wrestler")
        )
        features["grappler_vs_striker"] = float(
            style_a in ("grappler", "wrestler") and style_b == "striker"
        )

        # Physical features
        prof_a = self.fighter_profiles.get(fighter_a, {})
        prof_b = self.fighter_profiles.get(fighter_b, {})

        height_a = self._parse_height(prof_a.get("height", ""))
        height_b = self._parse_height(prof_b.get("height", ""))
        reach_a = self._parse_reach(prof_a.get("reach", ""))
        reach_b = self._parse_reach(prof_b.get("reach", ""))

        features["height_diff"] = (height_a - height_b) if height_a and height_b else 0.0
        features["reach_diff"] = (reach_a - reach_b) if reach_a and reach_b else 0.0

        # Reach advantage interaction: more valuable in striking matchups
        if style_a == "striker" and style_b == "striker":
            features["reach_diff_striking"] = features["reach_diff"]
        else:
            features["reach_diff_striking"] = features["reach_diff"] * 0.3

        # Reach-to-height ratio (ape index)
        if height_a and reach_a:
            features["ape_index_a"] = reach_a / height_a
        else:
            features["ape_index_a"] = 1.0
        if height_b and reach_b:
            features["ape_index_b"] = reach_b / height_b
        else:
            features["ape_index_b"] = 1.0
        features["ape_index_diff"] = features["ape_index_a"] - features["ape_index_b"]

        # Stance matchup
        stance_a = prof_a.get("stance", "orthodox").lower()
        stance_b = prof_b.get("stance", "orthodox").lower()
        features["stance_edge"] = STANCE_EDGE.get((stance_a, stance_b), 0.0)
        features["southpaw_vs_orthodox"] = float(stance_a == "southpaw" and stance_b == "orthodox")

        # Grappling advantage
        td_a = stats_a.get("td_avg", 0)
        td_b = stats_b.get("td_avg", 0)
        td_def_a = stats_a.get("td_defense", 0.5)
        td_def_b = stats_b.get("td_defense", 0.5)

        features["td_advantage_a"] = td_a * (1 - td_def_b)
        features["td_advantage_b"] = td_b * (1 - td_def_a)
        features["td_advantage_diff"] = features["td_advantage_a"] - features["td_advantage_b"]

        # Striking differential
        str_a = stats_a.get("sig_str_landed_pm", 0)
        str_b = stats_b.get("sig_str_landed_pm", 0)
        str_def_a = stats_a.get("sig_str_defense", 0.5)
        str_def_b = stats_b.get("sig_str_defense", 0.5)

        features["striking_advantage_a"] = str_a * (1 - str_def_b)
        features["striking_advantage_b"] = str_b * (1 - str_def_a)
        features["striking_advantage_diff"] = features["striking_advantage_a"] - features["striking_advantage_b"]

        # Submission threat
        sub_a = stats_a.get("sub_avg", 0)
        sub_b = stats_b.get("sub_avg", 0)
        features["sub_threat_diff"] = sub_a - sub_b

        return features

    def _parse_height(self, height_str: str) -> float | None:
        """Parse height string to inches."""
        if not height_str or height_str == "--":
            return None
        try:
            if "'" in height_str:
                parts = height_str.replace('"', '').split("'")
                feet = int(parts[0].strip())
                inches = int(parts[1].strip()) if len(parts) > 1 and parts[1].strip() else 0
                return feet * 12.0 + inches
            elif "cm" in height_str.lower():
                cm = float(height_str.lower().replace("cm", "").strip())
                return cm / 2.54
        except (ValueError, IndexError):
            pass
        return None

    def _parse_reach(self, reach_str: str) -> float | None:
        """Parse reach string to inches."""
        if not reach_str or reach_str == "--":
            return None
        try:
            cleaned = reach_str.replace('"', '').replace("'", "").strip()
            if "cm" in cleaned.lower():
                cm = float(cleaned.lower().replace("cm", "").strip())
                return cm / 2.54
            return float(cleaned)
        except ValueError:
            return None
