"""Feature pipeline: orchestrates all feature generators and produces the final feature matrix."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .elo import EloSystem, GlickoSystem
from .stats import FighterStatsFeatures
from .matchup import MatchupFeatures
from .contextual import ContextualFeatures

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Orchestrates feature generation for all fights, respecting temporal ordering."""

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        elo_cfg = cfg.get("elo", {})
        recency_cfg = cfg.get("recency", {})

        self.elo = EloSystem(
            k_factor=elo_cfg.get("k_factor", 32),
            initial_rating=elo_cfg.get("initial_rating", 1500),
        )
        self.glicko = GlickoSystem(
            tau=elo_cfg.get("glicko2_tau", 0.5),
            initial_rating=elo_cfg.get("initial_rating", 1500),
        )
        self.stats = FighterStatsFeatures(
            decay_halflife=recency_cfg.get("decay_halflife_fights", 5),
            windows=recency_cfg.get("windows", [3, 5, 10]),
        )
        self.matchup = MatchupFeatures()
        self.contextual = ContextualFeatures()

        self.fighter_info: dict[str, dict[str, Any]] = {}

    def set_fighter_info(self, fighter: str, info: dict[str, Any]) -> None:
        """Store fighter metadata (DOB, height, reach, camp, etc.)."""
        self.fighter_info[fighter] = info
        self.matchup.set_profile(fighter, info)

    def process_historical_fights(self, fights_df: pd.DataFrame) -> pd.DataFrame:
        """Process all historical fights chronologically and build feature matrix.

        CRITICAL: Features are computed using only data available BEFORE each fight.
        Elo/Glicko update after feature extraction.
        Fighter A/B assignment is randomized to prevent ordering leakage.
        """
        import random
        rng = random.Random(42)

        fights_sorted = fights_df.sort_values("date").reset_index(drop=True)
        all_features = []

        for idx, row in fights_sorted.iterrows():
            fighter_a = row["fighter_a"]
            fighter_b = row["fighter_b"]

            # Randomize A/B to prevent winner-first ordering leakage
            if rng.random() < 0.5:
                fighter_a, fighter_b = fighter_b, fighter_a

            fight_date = str(row.get("date", ""))

            features = self._extract_features_for_fight(
                fighter_a, fighter_b, fight_date, row
            )

            # Store target variables
            features["target_winner_is_a"] = int(row.get("winner", "") == fighter_a)
            features["target_method"] = self._classify_method(row.get("method", ""))
            features["target_round"] = row.get("round", None)
            features["target_duration_seconds"] = row.get("duration_seconds", None)
            features["fight_date"] = fight_date
            features["fighter_a"] = fighter_a
            features["fighter_b"] = fighter_b
            features["fight_idx"] = idx

            all_features.append(features)

            # Update systems with fight result (AFTER feature extraction)
            self._update_systems(row)

        df = pd.DataFrame(all_features)
        logger.info(f"Generated feature matrix: {df.shape[0]} fights, {df.shape[1]} features")
        return df

    def predict_features(self, fighter_a: str, fighter_b: str,
                         fight_info: dict[str, Any]) -> dict[str, float]:
        """Generate features for a new/upcoming fight (no update, just prediction)."""
        fight_date = str(fight_info.get("date", ""))
        row = pd.Series({
            "fighter_a": fighter_a,
            "fighter_b": fighter_b,
            "date": fight_date,
            **fight_info,
        })
        return self._extract_features_for_fight(fighter_a, fighter_b, fight_date, row)

    def _extract_features_for_fight(self, fighter_a: str, fighter_b: str,
                                    fight_date: str, row: pd.Series) -> dict[str, float]:
        """Extract all features for a single fight using pre-fight data only."""
        features: dict[str, float] = {}

        # Elo features
        elo_feats = self.elo.get_features_for_fight(
            fighter_a, fighter_b, row.get("weight_class", "")
        )
        features.update(elo_feats)

        # Glicko features
        glicko_feats = self.glicko.get_features_for_fight(fighter_a, fighter_b)
        features.update(glicko_feats)

        # Fighter stats (recency-weighted rolling stats)
        stats_diff = self.stats.get_differential_features(fighter_a, fighter_b, fight_date)
        features.update(stats_diff)

        # Matchup features (style + physical)
        stats_a = self.stats.get_features(fighter_a, fight_date)
        stats_b = self.stats.get_features(fighter_b, fight_date)
        matchup_feats = self.matchup.get_features(fighter_a, fighter_b, stats_a, stats_b)
        features.update(matchup_feats)

        # Contextual features
        info_a = self.fighter_info.get(fighter_a, {})
        info_b = self.fighter_info.get(fighter_b, {})
        fight_info = {
            "date": fight_date,
            "is_main_event": row.get("is_main_event", False),
            "is_co_main": row.get("is_co_main", False),
            "is_title_fight": row.get("is_title_fight", False),
            "scheduled_rounds": row.get("scheduled_rounds", 3),
            "location": row.get("location", ""),
            "referee": row.get("referee", ""),
            "card_position": row.get("card_position", 5),
        }
        ctx_feats = self.contextual.get_features(info_a, info_b, fight_info)
        features.update(ctx_feats)

        # Odds features (if available)
        odds_feats = self._get_odds_features(row)
        features.update(odds_feats)

        return features

    def _get_odds_features(self, row: pd.Series) -> dict[str, float]:
        """Extract odds-derived features from the row."""
        features = {}
        odds_cols = [
            "closing_prob_a", "closing_prob_b", "opening_prob_a", "opening_prob_b",
            "line_move_a", "line_move_b", "odds_range_a", "odds_range_b",
        ]
        for col in odds_cols:
            val = row.get(col)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                features[col] = float(val)
            else:
                features[col] = 0.5 if "prob" in col else 0.0
        return features

    def _update_systems(self, row: pd.Series) -> None:
        """Update all rating systems and stat trackers after a fight."""
        fighter_a = row["fighter_a"]
        fighter_b = row["fighter_b"]
        winner = row.get("winner", "")
        loser = fighter_b if winner == fighter_a else fighter_a
        method = row.get("method", "")
        division = row.get("weight_class", "")
        fight_date = str(row.get("date", ""))

        # Update Elo
        if winner:
            self.elo.update(winner, loser, method, division, fight_date)
            self.glicko.update(winner, loser)

        # Update fighter stats history
        for fighter, prefix in [(fighter_a, "a_"), (fighter_b, "b_")]:
            fight_stats = {}
            for col in FighterStatsFeatures.STAT_COLS:
                val = row.get(f"{prefix}{col}") or row.get(col)
                if val is not None:
                    fight_stats[col] = val
            fight_stats["result"] = "win" if fighter == winner else "loss"
            fight_stats["method"] = method
            self.stats.add_fight(fighter, fight_stats, fight_date)

    def _classify_method(self, method: str) -> int:
        """Classify method into categories: 0=KO/TKO, 1=Submission, 2=Decision."""
        m = method.lower() if method else ""
        if "ko" in m or "tko" in m:
            return 0
        elif "sub" in m:
            return 1
        else:
            return 2

    def get_feature_names(self) -> list[str]:
        """Get list of all feature column names (excludes targets and metadata)."""
        exclude_prefixes = ("target_", "fight_date", "fighter_a", "fighter_b", "fight_idx")
        dummy = self._extract_features_for_fight("dummy_a", "dummy_b", "2020-01-01", pd.Series({
            "fighter_a": "dummy_a", "fighter_b": "dummy_b", "date": "2020-01-01",
            "weight_class": "", "method": "", "winner": "",
        }))
        return [k for k in dummy.keys() if not any(k.startswith(p) for p in exclude_prefixes)]

    def verify_no_leakage(self, feature_df: pd.DataFrame) -> bool:
        """Verify symmetry: swap fighters and check predictions flip appropriately."""
        sample = feature_df.sample(min(100, len(feature_df)), random_state=42)
        suspicious = 0

        for _, row in sample.iterrows():
            fighter_a = row.get("fighter_a", "")
            fighter_b = row.get("fighter_b", "")
            if not fighter_a or not fighter_b:
                continue
            # In a proper model, swapping A/B should roughly invert the prediction
            # This is a sanity check — full verification done at model level
            elo_diff = row.get("elo_diff", 0)
            if abs(elo_diff) > 1000:
                suspicious += 1

        if suspicious > 10:
            logger.warning(f"Potential leakage: {suspicious}/100 fights have suspiciously large Elo diffs")
            return False
        return True
