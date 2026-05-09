"""Contextual features: age, layoff, weight cuts, venue, referee, etc."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

HIGH_ALTITUDE_VENUES = {
    "denver": 5280,
    "mexico city": 7350,
    "salt lake city": 4226,
    "albuquerque": 5312,
    "bogota": 8660,
    "johannesburg": 5751,
}

APEX_OCTAGON_SMALL = True

MAJOR_CAMPS = [
    "american top team", "jackson-wink", "city kickboxing",
    "aka", "team alpha male", "sanford mma", "xtreme couture",
    "tristar", "nova uniao", "kings mma", "tiger muay thai",
    "elevation fight team", "kill cliff fc", "fortis mma",
]


class ContextualFeatures:
    """Compute contextual/situational features for a fight."""

    def __init__(self, peak_age: float = 29.0, decline_rate: float = 0.02):
        self.peak_age = peak_age
        self.decline_rate = decline_rate

    def get_features(self, fighter_a_info: dict[str, Any],
                     fighter_b_info: dict[str, Any],
                     fight_info: dict[str, Any]) -> dict[str, float]:
        """Compute all contextual features for a fight."""
        features: dict[str, float] = {}

        fight_date = fight_info.get("date")
        if isinstance(fight_date, str):
            try:
                fight_date = datetime.strptime(fight_date, "%Y-%m-%d").date()
            except ValueError:
                fight_date = None

        # Age features
        features.update(self._age_features(fighter_a_info, fighter_b_info, fight_date))

        # Layoff features
        features.update(self._layoff_features(fighter_a_info, fighter_b_info, fight_date))

        # Weight cut indicators
        features.update(self._weight_features(fighter_a_info, fighter_b_info))

        # Card position and round count
        features.update(self._card_features(fight_info))

        # Venue/altitude
        features.update(self._venue_features(fight_info))

        # Camp features
        features.update(self._camp_features(fighter_a_info, fighter_b_info))

        # Referee features
        features.update(self._referee_features(fight_info))

        # Short notice
        features.update(self._notice_features(fighter_a_info, fighter_b_info, fight_info))

        return features

    def _age_features(self, info_a: dict, info_b: dict,
                      fight_date: date | None) -> dict[str, float]:
        """Age, age delta, miles-on-clock features."""
        age_a = self._compute_age(info_a.get("dob"), fight_date)
        age_b = self._compute_age(info_b.get("dob"), fight_date)

        features = {
            "age_a": age_a or 30.0,
            "age_b": age_b or 30.0,
            "age_diff": (age_a or 30.0) - (age_b or 30.0),
            "age_abs_diff": abs((age_a or 30.0) - (age_b or 30.0)),
        }

        # Age relative to peak
        if age_a:
            features["age_past_peak_a"] = max(0, age_a - self.peak_age)
            features["age_decline_factor_a"] = max(0, (age_a - self.peak_age) * self.decline_rate)
        else:
            features["age_past_peak_a"] = 0.0
            features["age_decline_factor_a"] = 0.0

        if age_b:
            features["age_past_peak_b"] = max(0, age_b - self.peak_age)
            features["age_decline_factor_b"] = max(0, (age_b - self.peak_age) * self.decline_rate)
        else:
            features["age_past_peak_b"] = 0.0
            features["age_decline_factor_b"] = 0.0

        # Miles on clock
        features["career_minutes_a"] = info_a.get("total_fight_minutes", 0)
        features["career_minutes_b"] = info_b.get("total_fight_minutes", 0)
        features["career_strikes_absorbed_a"] = info_a.get("total_strikes_absorbed", 0)
        features["career_strikes_absorbed_b"] = info_b.get("total_strikes_absorbed", 0)
        features["career_kd_absorbed_a"] = info_a.get("total_kd_absorbed", 0)
        features["career_kd_absorbed_b"] = info_b.get("total_kd_absorbed", 0)

        return features

    def _layoff_features(self, info_a: dict, info_b: dict,
                         fight_date: date | None) -> dict[str, float]:
        """Ring rust and layoff indicators."""
        layoff_a = self._compute_layoff(info_a.get("last_fight_date"), fight_date)
        layoff_b = self._compute_layoff(info_b.get("last_fight_date"), fight_date)

        features = {
            "layoff_days_a": layoff_a or 180,
            "layoff_days_b": layoff_b or 180,
            "layoff_diff": (layoff_a or 180) - (layoff_b or 180),
        }

        # Layoff buckets
        for label, fighter_layoff, suffix in [("a", layoff_a, "_a"), ("b", layoff_b, "_b")]:
            days = fighter_layoff or 180
            features[f"layoff_short{suffix}"] = float(days < 90)
            features[f"layoff_medium{suffix}"] = float(90 <= days < 365)
            features[f"layoff_long{suffix}"] = float(365 <= days < 730)
            features[f"layoff_very_long{suffix}"] = float(days >= 730)

        # Layoff x age interaction
        age_a = info_a.get("age") or 30
        age_b = info_b.get("age") or 30
        features["layoff_x_age_a"] = (layoff_a or 180) * max(0, age_a - 30) / 365
        features["layoff_x_age_b"] = (layoff_b or 180) * max(0, age_b - 30) / 365

        # Coming off KO loss
        features["last_ko_loss_a"] = float(info_a.get("last_result_ko_loss", False))
        features["last_ko_loss_b"] = float(info_b.get("last_result_ko_loss", False))

        return features

    def _weight_features(self, info_a: dict, info_b: dict) -> dict[str, float]:
        """Weight cut and missed weight indicators."""
        return {
            "weight_misses_a": info_a.get("weight_misses_count", 0),
            "weight_misses_b": info_b.get("weight_misses_count", 0),
            "recent_weight_miss_a": float(info_a.get("recent_weight_miss", False)),
            "recent_weight_miss_b": float(info_b.get("recent_weight_miss", False)),
            "weight_class_changes_a": info_a.get("weight_class_changes", 0),
            "weight_class_changes_b": info_b.get("weight_class_changes", 0),
            "moving_up_a": float(info_a.get("moving_up", False)),
            "moving_up_b": float(info_b.get("moving_up", False)),
            "moving_down_a": float(info_a.get("moving_down", False)),
            "moving_down_b": float(info_b.get("moving_down", False)),
        }

    def _card_features(self, fight_info: dict) -> dict[str, float]:
        """Card position, main event, title fight indicators."""
        return {
            "is_main_event": float(fight_info.get("is_main_event", False)),
            "is_co_main": float(fight_info.get("is_co_main", False)),
            "is_title_fight": float(fight_info.get("is_title_fight", False)),
            "scheduled_rounds": fight_info.get("scheduled_rounds", 3),
            "is_five_rounds": float(fight_info.get("scheduled_rounds", 3) == 5),
            "card_position": fight_info.get("card_position", 5),
        }

    def _venue_features(self, fight_info: dict) -> dict[str, float]:
        """Venue altitude and octagon size."""
        location = str(fight_info.get("location", "")).lower()
        altitude = 0
        for city, alt in HIGH_ALTITUDE_VENUES.items():
            if city in location:
                altitude = alt
                break

        is_apex = "apex" in location or "las vegas" in location
        return {
            "altitude_feet": altitude,
            "is_high_altitude": float(altitude > 4000),
            "is_apex": float(is_apex),
            "small_octagon": float(is_apex),
        }

    def _camp_features(self, info_a: dict, info_b: dict) -> dict[str, float]:
        """Training camp features."""
        camp_a = str(info_a.get("camp", "")).lower()
        camp_b = str(info_b.get("camp", "")).lower()

        features = {
            "major_camp_a": float(any(c in camp_a for c in MAJOR_CAMPS)),
            "major_camp_b": float(any(c in camp_b for c in MAJOR_CAMPS)),
            "camp_change_a": float(info_a.get("camp_changed", False)),
            "camp_change_b": float(info_b.get("camp_changed", False)),
        }
        return features

    def _referee_features(self, fight_info: dict) -> dict[str, float]:
        """Referee tendencies."""
        ref = str(fight_info.get("referee", "")).lower()
        # Known early-stoppage refs vs late-stoppage
        early_refs = ["herb dean", "marc goddard", "jason herzog"]
        late_refs = ["mario yamasaki", "steve mazzagatti", "kim winslow"]

        return {
            "ref_early_stoppage": float(any(r in ref for r in early_refs)),
            "ref_late_stoppage": float(any(r in ref for r in late_refs)),
        }

    def _notice_features(self, info_a: dict, info_b: dict,
                         fight_info: dict) -> dict[str, float]:
        """Short notice replacement indicators."""
        return {
            "short_notice_a": float(info_a.get("short_notice", False)),
            "short_notice_b": float(info_b.get("short_notice", False)),
            "any_short_notice": float(
                info_a.get("short_notice", False) or info_b.get("short_notice", False)
            ),
            "days_notice_a": info_a.get("days_notice", 42),
            "days_notice_b": info_b.get("days_notice", 42),
        }

    def _compute_age(self, dob: str | None, fight_date: date | None) -> float | None:
        if not dob or not fight_date:
            return None
        try:
            if isinstance(dob, str):
                for fmt in ("%b %d, %Y", "%Y-%m-%d", "%B %d, %Y"):
                    try:
                        dob_date = datetime.strptime(dob, fmt).date()
                        break
                    except ValueError:
                        continue
                else:
                    return None
            else:
                dob_date = dob

            age = (fight_date - dob_date).days / 365.25
            return round(age, 1)
        except Exception:
            return None

    def _compute_layoff(self, last_fight_date: str | None,
                        fight_date: date | None) -> int | None:
        if not last_fight_date or not fight_date:
            return None
        try:
            if isinstance(last_fight_date, str):
                last = datetime.strptime(last_fight_date, "%Y-%m-%d").date()
            else:
                last = last_fight_date
            return (fight_date - last).days
        except Exception:
            return None
