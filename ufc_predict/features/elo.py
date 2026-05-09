"""Elo and Glicko-2 rating systems for fighters."""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EloRating:
    rating: float = 1500.0
    fights: int = 0
    wins: int = 0
    losses: int = 0


@dataclass
class GlickoRating:
    rating: float = 1500.0
    rd: float = 350.0  # rating deviation
    vol: float = 0.06  # volatility
    fights: int = 0


class EloSystem:
    """Standard Elo rating system with method-specific variants."""

    def __init__(self, k_factor: float = 32.0, initial_rating: float = 1500.0):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings: dict[str, EloRating] = {}
        self.ko_ratings: dict[str, EloRating] = {}
        self.sub_ratings: dict[str, EloRating] = {}
        self.dec_ratings: dict[str, EloRating] = {}
        self.division_ratings: dict[str, dict[str, EloRating]] = {}
        self.history: list[dict[str, Any]] = []

    def get_rating(self, fighter: str) -> EloRating:
        if fighter not in self.ratings:
            self.ratings[fighter] = EloRating(self.initial_rating)
        return self.ratings[fighter]

    def get_method_rating(self, fighter: str, method: str) -> EloRating:
        store = self._method_store(method)
        if fighter not in store:
            store[fighter] = EloRating(self.initial_rating)
        return store[fighter]

    def get_division_rating(self, fighter: str, division: str) -> EloRating:
        if division not in self.division_ratings:
            self.division_ratings[division] = {}
        if fighter not in self.division_ratings[division]:
            self.division_ratings[division][fighter] = EloRating(self.initial_rating)
        return self.division_ratings[division][fighter]

    def _method_store(self, method: str) -> dict[str, EloRating]:
        method_lower = method.lower() if method else ""
        if "ko" in method_lower or "tko" in method_lower:
            return self.ko_ratings
        elif "sub" in method_lower:
            return self.sub_ratings
        else:
            return self.dec_ratings

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def update(self, winner: str, loser: str, method: str = "",
               division: str = "", fight_date: str = "") -> dict[str, Any]:
        """Update ratings after a fight. Returns pre-fight features."""
        rating_w = self.get_rating(winner)
        rating_l = self.get_rating(loser)

        pre_fight = {
            "elo_winner": rating_w.rating,
            "elo_loser": rating_l.rating,
            "elo_diff": rating_w.rating - rating_l.rating,
            "elo_expected": self.expected_score(rating_w.rating, rating_l.rating),
        }

        # Method-specific ratings (pre-fight snapshot)
        if method:
            method_w = self.get_method_rating(winner, method)
            method_l = self.get_method_rating(loser, method)
            method_key = self._method_key(method)
            pre_fight[f"elo_{method_key}_winner"] = method_w.rating
            pre_fight[f"elo_{method_key}_loser"] = method_l.rating
            pre_fight[f"elo_{method_key}_diff"] = method_w.rating - method_l.rating

        # Division-specific
        if division:
            div_w = self.get_division_rating(winner, division)
            div_l = self.get_division_rating(loser, division)
            pre_fight["elo_div_winner"] = div_w.rating
            pre_fight["elo_div_loser"] = div_l.rating
            pre_fight["elo_div_diff"] = div_w.rating - div_l.rating

        # Perform updates
        k = self._adaptive_k(rating_w, rating_l)
        expected_w = self.expected_score(rating_w.rating, rating_l.rating)

        rating_w.rating += k * (1.0 - expected_w)
        rating_l.rating += k * (0.0 - (1.0 - expected_w))
        rating_w.fights += 1
        rating_w.wins += 1
        rating_l.fights += 1
        rating_l.losses += 1

        # Update method-specific
        if method:
            method_w = self.get_method_rating(winner, method)
            method_l = self.get_method_rating(loser, method)
            exp_m = self.expected_score(method_w.rating, method_l.rating)
            method_w.rating += self.k_factor * (1.0 - exp_m)
            method_l.rating += self.k_factor * (0.0 - (1.0 - exp_m))
            method_w.fights += 1
            method_l.fights += 1

        # Update division-specific
        if division:
            div_w = self.get_division_rating(winner, division)
            div_l = self.get_division_rating(loser, division)
            exp_d = self.expected_score(div_w.rating, div_l.rating)
            div_w.rating += self.k_factor * (1.0 - exp_d)
            div_l.rating += self.k_factor * (0.0 - (1.0 - exp_d))
            div_w.fights += 1
            div_l.fights += 1

        self.history.append({
            "winner": winner,
            "loser": loser,
            "method": method,
            "division": division,
            "date": fight_date,
            **pre_fight,
        })

        return pre_fight

    def _adaptive_k(self, rating_a: EloRating, rating_b: EloRating) -> float:
        """Adaptive K-factor: higher for new fighters, lower for established."""
        min_fights = min(rating_a.fights, rating_b.fights)
        if min_fights < 5:
            return self.k_factor * 1.5
        elif min_fights < 15:
            return self.k_factor
        else:
            return self.k_factor * 0.75

    def _method_key(self, method: str) -> str:
        method_lower = method.lower() if method else ""
        if "ko" in method_lower or "tko" in method_lower:
            return "ko"
        elif "sub" in method_lower:
            return "sub"
        return "dec"

    def get_features_for_fight(self, fighter_a: str, fighter_b: str,
                               division: str = "") -> dict[str, float]:
        """Get pre-fight Elo features without updating."""
        r_a = self.get_rating(fighter_a)
        r_b = self.get_rating(fighter_b)

        features = {
            "elo_a": r_a.rating,
            "elo_b": r_b.rating,
            "elo_diff": r_a.rating - r_b.rating,
            "elo_expected_a": self.expected_score(r_a.rating, r_b.rating),
            "elo_fights_a": r_a.fights,
            "elo_fights_b": r_b.fights,
            "elo_winrate_a": r_a.wins / max(r_a.fights, 1),
            "elo_winrate_b": r_b.wins / max(r_b.fights, 1),
        }

        # Method-specific
        for method_key, store in [("ko", self.ko_ratings), ("sub", self.sub_ratings), ("dec", self.dec_ratings)]:
            ra = store.get(fighter_a, EloRating(self.initial_rating))
            rb = store.get(fighter_b, EloRating(self.initial_rating))
            features[f"elo_{method_key}_a"] = ra.rating
            features[f"elo_{method_key}_b"] = rb.rating
            features[f"elo_{method_key}_diff"] = ra.rating - rb.rating

        # Division-specific
        if division and division in self.division_ratings:
            div_store = self.division_ratings[division]
            da = div_store.get(fighter_a, EloRating(self.initial_rating))
            db = div_store.get(fighter_b, EloRating(self.initial_rating))
            features["elo_div_a"] = da.rating
            features["elo_div_b"] = db.rating
            features["elo_div_diff"] = da.rating - db.rating

        return features

    def process_fights_chronologically(self, fights_df: pd.DataFrame) -> pd.DataFrame:
        """Process all fights in chronological order, returning features per fight."""
        fights_sorted = fights_df.sort_values("date").reset_index(drop=True)
        feature_rows = []

        for _, row in fights_sorted.iterrows():
            winner = row.get("winner", "")
            loser = row.get("loser", "")
            if not winner or not loser:
                fighter_a = row.get("fighter_a", "")
                fighter_b = row.get("fighter_b", "")
                winner = row.get("winner", fighter_a)
                loser = fighter_b if winner == fighter_a else fighter_a

            features = self.get_features_for_fight(
                row.get("fighter_a", winner),
                row.get("fighter_b", loser),
                row.get("weight_class", ""),
            )
            features["fight_idx"] = _
            feature_rows.append(features)

            self.update(
                winner=winner,
                loser=loser,
                method=row.get("method", ""),
                division=row.get("weight_class", ""),
                fight_date=str(row.get("date", "")),
            )

        return pd.DataFrame(feature_rows)


class GlickoSystem:
    """Glicko-2 rating system — better for fighters with few fights (high uncertainty)."""

    def __init__(self, tau: float = 0.5, initial_rating: float = 1500.0,
                 initial_rd: float = 350.0, initial_vol: float = 0.06):
        self.tau = tau
        self.initial_rating = initial_rating
        self.initial_rd = initial_rd
        self.initial_vol = initial_vol
        self.ratings: dict[str, GlickoRating] = {}

    def get_rating(self, fighter: str) -> GlickoRating:
        if fighter not in self.ratings:
            self.ratings[fighter] = GlickoRating(
                self.initial_rating, self.initial_rd, self.initial_vol
            )
        return self.ratings[fighter]

    def _g(self, rd: float) -> float:
        return 1.0 / math.sqrt(1.0 + 3.0 * rd**2 / (math.pi**2))

    def _e(self, mu: float, mu_j: float, rd_j: float) -> float:
        return 1.0 / (1.0 + math.exp(-self._g(rd_j) * (mu - mu_j)))

    def _to_glicko2(self, rating: float, rd: float) -> tuple[float, float]:
        mu = (rating - 1500.0) / 173.7178
        phi = rd / 173.7178
        return mu, phi

    def _from_glicko2(self, mu: float, phi: float) -> tuple[float, float]:
        rating = mu * 173.7178 + 1500.0
        rd = phi * 173.7178
        return rating, rd

    def update(self, winner: str, loser: str) -> dict[str, float]:
        """Update Glicko-2 ratings after a fight."""
        r_w = self.get_rating(winner)
        r_l = self.get_rating(loser)

        pre_features = {
            "glicko_winner": r_w.rating,
            "glicko_loser": r_l.rating,
            "glicko_rd_winner": r_w.rd,
            "glicko_rd_loser": r_l.rd,
            "glicko_diff": r_w.rating - r_l.rating,
            "glicko_vol_winner": r_w.vol,
            "glicko_vol_loser": r_l.vol,
        }

        mu_w, phi_w = self._to_glicko2(r_w.rating, r_w.rd)
        mu_l, phi_l = self._to_glicko2(r_l.rating, r_l.rd)

        # Update winner
        g_l = self._g(phi_l)
        e_w = self._e(mu_w, mu_l, phi_l)
        v_w = 1.0 / (g_l**2 * e_w * (1 - e_w))
        delta_w = v_w * g_l * (1.0 - e_w)

        new_phi_w = 1.0 / math.sqrt(1.0 / phi_w**2 + 1.0 / v_w)
        new_mu_w = mu_w + new_phi_w**2 * g_l * (1.0 - e_w)

        # Update loser
        g_w = self._g(phi_w)
        e_l = self._e(mu_l, mu_w, phi_w)
        v_l = 1.0 / (g_w**2 * e_l * (1 - e_l))

        new_phi_l = 1.0 / math.sqrt(1.0 / phi_l**2 + 1.0 / v_l)
        new_mu_l = mu_l + new_phi_l**2 * g_w * (0.0 - e_l)

        r_w.rating, r_w.rd = self._from_glicko2(new_mu_w, new_phi_w)
        r_l.rating, r_l.rd = self._from_glicko2(new_mu_l, new_phi_l)
        r_w.fights += 1
        r_l.fights += 1

        return pre_features

    def get_features_for_fight(self, fighter_a: str, fighter_b: str) -> dict[str, float]:
        """Get pre-fight Glicko-2 features."""
        r_a = self.get_rating(fighter_a)
        r_b = self.get_rating(fighter_b)

        mu_a, phi_a = self._to_glicko2(r_a.rating, r_a.rd)
        mu_b, phi_b = self._to_glicko2(r_b.rating, r_b.rd)

        return {
            "glicko_a": r_a.rating,
            "glicko_b": r_b.rating,
            "glicko_diff": r_a.rating - r_b.rating,
            "glicko_rd_a": r_a.rd,
            "glicko_rd_b": r_b.rd,
            "glicko_vol_a": r_a.vol,
            "glicko_vol_b": r_b.vol,
            "glicko_expected_a": self._e(mu_a, mu_b, phi_b),
            "glicko_uncertainty": r_a.rd + r_b.rd,
        }

    def process_fights_chronologically(self, fights_df: pd.DataFrame) -> pd.DataFrame:
        """Process all fights and return Glicko features per fight."""
        fights_sorted = fights_df.sort_values("date").reset_index(drop=True)
        feature_rows = []

        for idx, row in fights_sorted.iterrows():
            fighter_a = row.get("fighter_a", "")
            fighter_b = row.get("fighter_b", "")

            features = self.get_features_for_fight(fighter_a, fighter_b)
            feature_rows.append(features)

            winner = row.get("winner", "")
            loser = fighter_b if winner == fighter_a else fighter_a
            if winner:
                self.update(winner, loser)

        return pd.DataFrame(feature_rows)
