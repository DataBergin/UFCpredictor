"""Tests for feature engineering modules."""

import numpy as np
import pandas as pd
import pytest

from ufc_predict.features.elo import EloSystem, GlickoSystem
from ufc_predict.features.stats import FighterStatsFeatures
from ufc_predict.features.matchup import MatchupFeatures
from ufc_predict.features.contextual import ContextualFeatures
from ufc_predict.utils import american_to_implied_prob, devig_odds, normalize_name


class TestEloSystem:
    def test_initial_rating(self):
        elo = EloSystem()
        r = elo.get_rating("Test Fighter")
        assert r.rating == 1500.0
        assert r.fights == 0

    def test_update_winner_gains(self):
        elo = EloSystem(k_factor=32)
        pre = elo.get_features_for_fight("A", "B")
        elo.update("A", "B")
        assert elo.get_rating("A").rating > 1500.0
        assert elo.get_rating("B").rating < 1500.0

    def test_symmetry(self):
        elo = EloSystem()
        feats_ab = elo.get_features_for_fight("X", "Y")
        assert abs(feats_ab["elo_expected_a"] + (1 - feats_ab["elo_expected_a"]) - 1.0) < 1e-10

    def test_method_specific_elo(self):
        elo = EloSystem()
        elo.update("A", "B", method="KO/TKO")
        elo.update("A", "C", method="KO/TKO")
        elo.update("A", "D", method="Submission")
        feats = elo.get_features_for_fight("A", "E")
        assert feats["elo_ko_a"] > feats["elo_sub_a"]

    def test_chronological_ordering(self):
        elo = EloSystem()
        elo.update("A", "B")
        elo.update("B", "C")
        # A beat B, B beat C => A should be rated higher than C
        assert elo.get_rating("A").rating > elo.get_rating("C").rating


class TestGlickoSystem:
    def test_initial_rd(self):
        glicko = GlickoSystem()
        r = glicko.get_rating("New Fighter")
        assert r.rd == 350.0

    def test_rd_decreases_with_fights(self):
        glicko = GlickoSystem()
        initial_rd = glicko.get_rating("A").rd
        glicko.update("A", "B")
        assert glicko.get_rating("A").rd < initial_rd

    def test_uncertainty_feature(self):
        glicko = GlickoSystem()
        feats = glicko.get_features_for_fight("Unknown1", "Unknown2")
        # Two unknown fighters should have high uncertainty
        assert feats["glicko_uncertainty"] > 600


class TestFighterStats:
    def test_empty_fighter(self):
        stats = FighterStatsFeatures()
        feats = stats.get_features("unknown_fighter")
        assert feats["n_fights"] == 0

    def test_adding_fights(self):
        stats = FighterStatsFeatures()
        stats.add_fight("A", {"sig_str_landed_pm": 5.0, "result": "win", "method": "KO"}, "2023-01-01")
        stats.add_fight("A", {"sig_str_landed_pm": 6.0, "result": "win", "method": "KO"}, "2023-06-01")
        feats = stats.get_features("A")
        assert feats["n_fights"] == 2
        assert feats["ko_rate"] == 1.0

    def test_recency_weighting(self):
        stats = FighterStatsFeatures(decay_halflife=2)
        stats.add_fight("A", {"sig_str_landed_pm": 2.0, "result": "win"}, "2020-01-01")
        stats.add_fight("A", {"sig_str_landed_pm": 8.0, "result": "win"}, "2023-01-01")
        feats = stats.get_features("A")
        # Recent fight should be weighted more heavily
        assert feats["ewm_sig_str_landed_pm"] > 5.0

    def test_as_of_date_filtering(self):
        stats = FighterStatsFeatures()
        stats.add_fight("A", {"sig_str_landed_pm": 5.0, "result": "win"}, "2022-01-01")
        stats.add_fight("A", {"sig_str_landed_pm": 10.0, "result": "win"}, "2024-01-01")
        feats = stats.get_features("A", as_of_date="2023-01-01")
        assert feats["n_fights"] == 1


class TestMatchupFeatures:
    def test_stance_edge(self):
        matchup = MatchupFeatures()
        matchup.set_profile("A", {"stance": "Southpaw"})
        matchup.set_profile("B", {"stance": "Orthodox"})
        feats = matchup.get_features("A", "B",
                                     {"sig_str_landed_pm": 5.0, "td_avg": 1.0, "sub_avg": 0.5},
                                     {"sig_str_landed_pm": 4.0, "td_avg": 2.0, "sub_avg": 0.3})
        assert feats["southpaw_vs_orthodox"] == 1.0
        assert feats["stance_edge"] > 0

    def test_reach_advantage(self):
        matchup = MatchupFeatures()
        matchup.set_profile("A", {"reach": "76", "height": "6'2\""})
        matchup.set_profile("B", {"reach": "70", "height": "5'10\""})
        feats = matchup.get_features("A", "B", {}, {})
        assert feats["reach_diff"] > 0
        assert feats["height_diff"] > 0


class TestUtils:
    def test_american_to_implied(self):
        assert abs(american_to_implied_prob(-200) - 0.6667) < 0.001
        assert abs(american_to_implied_prob(200) - 0.3333) < 0.001
        assert abs(american_to_implied_prob(-100) - 0.5) < 0.001

    def test_devig(self):
        # -200 / +170 typical vig line
        prob_a = american_to_implied_prob(-200)
        prob_b = american_to_implied_prob(170)
        devig_a, devig_b = devig_odds(prob_a, prob_b)
        assert abs(devig_a + devig_b - 1.0) < 1e-10

    def test_normalize_name(self):
        assert normalize_name("Jon Jones") == "jon jones"
        assert normalize_name("Jon 'Bones' Jones") == "jon bones jones"
        assert normalize_name("  Anderson  Silva  ") == "anderson silva"


class TestContextual:
    def test_altitude_detection(self):
        ctx = ContextualFeatures()
        feats = ctx.get_features({}, {}, {"location": "Denver, Colorado", "date": "2024-01-01"})
        assert feats["is_high_altitude"] == 1.0
        assert feats["altitude_feet"] == 5280

    def test_apex_detection(self):
        ctx = ContextualFeatures()
        feats = ctx.get_features({}, {}, {"location": "UFC APEX, Las Vegas", "date": "2024-01-01"})
        assert feats["is_apex"] == 1.0
        assert feats["small_octagon"] == 1.0
