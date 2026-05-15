"""Club-to-national-team form aggregation.

The core insight: national team players play 40-60 club matches per year
but only 5-10 international matches. Between World Cups, the best signal
for a national team's strength is how its players are performing at their
clubs.

This module:
1. Maps players to national teams via nationality from FBref rosters
2. Aggregates club-level performance (goals, assists, minutes, xG) per nation
3. Weights by league strength (Premier League > Ligue 1)
4. Computes club league position for each player's club

Example: Before a World Cup match, France's squad might have players at
PSG (1st in Ligue 1), Real Madrid (2nd in La Liga), Arsenal (1st in PL).
Their average league position, combined goals/assists at club level, and
aggregate xG tell us how sharp those players are right now.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# FBref nationality codes → standard country names for national teams
# FBref uses 3-letter codes that sometimes differ from FIFA codes
NATIONALITY_TO_COUNTRY = {
    # Major national teams
    "ENG": "England", "FRA": "France", "GER": "Germany", "ESP": "Spain",
    "ITA": "Italy", "BRA": "Brazil", "ARG": "Argentina", "POR": "Portugal",
    "NED": "Netherlands", "BEL": "Belgium", "CRO": "Croatia", "URU": "Uruguay",
    "COL": "Colombia", "MEX": "Mexico", "USA": "United States", "JPN": "Japan",
    "KOR": "South Korea", "AUS": "Australia", "SEN": "Senegal", "MAR": "Morocco",
    "GHA": "Ghana", "CMR": "Cameroon", "NGA": "Nigeria", "CIV": "Ivory Coast",
    "TUN": "Tunisia", "ALG": "Algeria", "EGY": "Egypt",
    "SUI": "Switzerland", "AUT": "Austria", "DEN": "Denmark", "SWE": "Sweden",
    "NOR": "Norway", "POL": "Poland", "CZE": "Czech Republic", "SCO": "Scotland",
    "WAL": "Wales", "IRL": "Ireland", "SRB": "Serbia", "UKR": "Ukraine",
    "ROU": "Romania", "GRE": "Greece", "TUR": "Turkey", "RUS": "Russia",
    "CHI": "Chile", "PAR": "Paraguay", "ECU": "Ecuador", "PER": "Peru",
    "VEN": "Venezuela", "BOL": "Bolivia", "CRC": "Costa Rica", "PAN": "Panama",
    "HON": "Honduras", "JAM": "Jamaica", "CAN": "Canada",
    "IRN": "Iran", "KSA": "Saudi Arabia", "QAT": "Qatar", "UAE": "UAE",
    "CHN": "China", "IND": "India",
}

# League strength weights (used to weight player contributions)
# Based on UEFA coefficients — PL and La Liga are strongest
LEAGUE_WEIGHTS = {
    "premier_league": 1.0,
    "la_liga": 0.95,
    "bundesliga": 0.85,
    "serie_a": 0.85,
    "ligue_1": 0.75,
}


class ClubFormAggregator:
    """Aggregate club-level player performance by national team.

    Takes player rosters (with nationality) and club results, and produces
    per-national-team features that capture how well a country's players
    are performing at the club level.
    """

    def __init__(self):
        self.player_data: pd.DataFrame = pd.DataFrame()
        self.standings_data: dict[tuple[str, str], pd.DataFrame] = {}
        self._nation_cache: dict[str, dict[str, float]] = {}

    def load_player_data(self, players_df: pd.DataFrame) -> None:
        """Load player-level data with nationality, club stats.

        Expected columns: player_name, nationality, club, league, season,
        appearances, starts, minutes, goals, assists, xg, xa
        """
        self.player_data = players_df.copy()

        # Map nationality codes to country names
        if "nationality" in self.player_data.columns:
            self.player_data["country"] = (
                self.player_data["nationality"]
                .map(NATIONALITY_TO_COUNTRY)
                .fillna(self.player_data["nationality"])
            )

        n_countries = self.player_data["country"].nunique() if "country" in self.player_data.columns else 0
        logger.info(f"Loaded {len(self.player_data)} player records, "
                     f"{n_countries} nationalities")

    def load_standings(self, standings: list[dict[str, Any]],
                       league: str, season: str) -> None:
        """Load league standings for club position lookups."""
        df = pd.DataFrame(standings)
        self.standings_data[(league, season)] = df

    def get_club_position(self, club: str, league: str,
                          season: str) -> int:
        """Get a club's league position (1=champion, 20=last)."""
        key = (league, season)
        if key not in self.standings_data:
            return 10  # default mid-table
        df = self.standings_data[key]
        match = df[df["club"].str.lower() == club.lower()]
        if match.empty:
            return 10
        return int(match.iloc[0].get("league_position", 10))

    def aggregate_for_nation(self, country: str,
                             season: str | None = None,
                             min_minutes: int = 200) -> dict[str, float]:
        """Aggregate club form for a national team.

        Args:
            country: Country name (e.g. "France", "Brazil")
            season: Filter to a specific season (e.g. "2023-2024")
            min_minutes: Only include players with this many minutes

        Returns:
            Dict of aggregated features:
            - club_goals_total: total goals scored at club level
            - club_assists_total: total assists
            - club_xg_total: total expected goals
            - club_minutes_total: total minutes across all players
            - club_avg_league_pos: avg league position of players' clubs
            - club_top5_league_pct: % of players in top-5 league
            - club_avg_age: average age of squad
            - club_n_players: number of eligible players found
            - club_weighted_goals: goals weighted by league strength
        """
        if self.player_data.empty:
            return self._empty_features()

        # Cache key
        cache_key = f"{country}_{season}_{min_minutes}"
        if cache_key in self._nation_cache:
            return self._nation_cache[cache_key]

        mask = self.player_data["country"].str.lower() == country.lower()
        if season:
            mask &= self.player_data["season"] == season
        if "minutes" in self.player_data.columns:
            mask &= self.player_data["minutes"] >= min_minutes

        players = self.player_data[mask]

        if players.empty:
            result = self._empty_features()
            self._nation_cache[cache_key] = result
            return result

        # Basic aggregates
        n = len(players)
        goals = players["goals"].sum() if "goals" in players.columns else 0
        assists = players["assists"].sum() if "assists" in players.columns else 0
        minutes = players["minutes"].sum() if "minutes" in players.columns else 0
        xg = players["xg"].sum() if "xg" in players.columns else 0.0
        xa = players["xa"].sum() if "xa" in players.columns else 0.0

        # Average age
        avg_age = players["age"].mean() if "age" in players.columns else 27.0

        # League position of each player's club
        positions = []
        league_weights_sum = 0.0
        weighted_goals = 0.0
        top5_count = 0

        for _, player in players.iterrows():
            league = player.get("league", "")
            club = player.get("club", "")
            p_season = player.get("season", season or "")

            if league in LEAGUE_WEIGHTS:
                top5_count += 1
                weight = LEAGUE_WEIGHTS[league]
                league_weights_sum += weight
                weighted_goals += float(player.get("goals", 0)) * weight

                pos = self.get_club_position(club, league, p_season)
                positions.append(pos)

        avg_league_pos = sum(positions) / len(positions) if positions else 10.0
        top5_pct = top5_count / n if n > 0 else 0.0

        # Per-90 rates
        per90_factor = 90.0 / max(minutes, 1)
        goals_per90 = goals * per90_factor * n  # team-level per-90
        xg_per90 = xg * per90_factor * n

        result = {
            "club_goals_total": float(goals),
            "club_assists_total": float(assists),
            "club_xg_total": float(xg),
            "club_xa_total": float(xa),
            "club_minutes_total": float(minutes),
            "club_goals_per90": float(goals_per90),
            "club_xg_per90": float(xg_per90),
            "club_avg_league_pos": float(avg_league_pos),
            "club_top5_league_pct": float(top5_pct),
            "club_avg_age": float(avg_age),
            "club_n_players": float(n),
            "club_weighted_goals": float(weighted_goals),
        }

        self._nation_cache[cache_key] = result
        return result

    def _empty_features(self) -> dict[str, float]:
        return {
            "club_goals_total": 0.0, "club_assists_total": 0.0,
            "club_xg_total": 0.0, "club_xa_total": 0.0,
            "club_minutes_total": 0.0,
            "club_goals_per90": 0.0, "club_xg_per90": 0.0,
            "club_avg_league_pos": 10.0,
            "club_top5_league_pct": 0.0,
            "club_avg_age": 27.0,
            "club_n_players": 0.0,
            "club_weighted_goals": 0.0,
        }
