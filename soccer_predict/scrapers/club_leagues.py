"""Scraper for top European club leagues from FBref.

Players on national teams play 40-60 club matches per year but only
5-10 international matches. Club form is the best proxy for current
player quality between international windows.

Covers: Premier League, La Liga, Bundesliga, Serie A, Ligue 1.
"""

from __future__ import annotations

import re
import logging
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

from sports_predict.scrapers.base import BaseScraper

logger = logging.getLogger(__name__)

FBREF_BASE = "https://fbref.com"

# FBref competition IDs for top 5 European leagues
LEAGUE_IDS = {
    "premier_league": {"id": 9, "name": "Premier League", "country": "England"},
    "la_liga":         {"id": 12, "name": "La Liga", "country": "Spain"},
    "bundesliga":      {"id": 20, "name": "Bundesliga", "country": "Germany"},
    "serie_a":         {"id": 11, "name": "Serie A", "country": "Italy"},
    "ligue_1":         {"id": 13, "name": "Ligue 1", "country": "France"},
}

# Season schedule URL pattern:
# /en/comps/{id}/{season}/schedule/{season}-{LeagueName}-Scores-and-Fixtures
# e.g. /en/comps/9/2023-2024/schedule/2023-2024-Premier-League-Scores-and-Fixtures


class ClubLeagueScraper(BaseScraper):
    """Scrape match results and player stats from top European club leagues."""

    SOURCE_NAME = "fbref_clubs"

    def __init__(self, cache_dir: str = "data/cache",
                 delay_range: tuple[float, float] = (4.0, 8.0)):
        # FBref rate limits aggressively — be polite
        super().__init__(cache_dir=cache_dir, delay_range=delay_range)

    # ------------------------------------------------------------------
    # Match results
    # ------------------------------------------------------------------

    def scrape_league_season(self, league_key: str,
                             season: str) -> list[dict[str, Any]]:
        """Scrape all match results for a league-season.

        Args:
            league_key: Key from LEAGUE_IDS (e.g. "premier_league")
            season: Season string like "2023-2024"

        Returns:
            List of match dicts with date, teams, score, venue.
        """
        info = LEAGUE_IDS.get(league_key)
        if info is None:
            logger.warning(f"Unknown league: {league_key}")
            return []

        league_name_url = info["name"].replace(" ", "-")
        url = (f"{FBREF_BASE}/en/comps/{info['id']}/{season}"
               f"/schedule/{season}-{league_name_url}-Scores-and-Fixtures")

        html = self.fetch(url)
        soup = BeautifulSoup(html, "lxml")

        matches = []
        table = soup.select_one("table.stats_table")
        if table is None:
            logger.warning(f"No schedule table for {league_key} {season}")
            return []

        rows = table.select("tbody tr:not(.thead)")
        for row in rows:
            match = self._parse_match_row(row, league_key, season, info)
            if match:
                matches.append(match)

        logger.info(f"{info['name']} {season}: {len(matches)} matches")
        return matches

    def _parse_match_row(self, row, league_key: str,
                         season: str, info: dict) -> dict[str, Any] | None:
        """Parse a single match row from the schedule table."""
        try:
            data: dict[str, Any] = {
                "league": league_key,
                "league_name": info["name"],
                "country": info["country"],
                "season": season,
            }

            for col in row.select("td, th"):
                stat = col.get("data-stat", "")

                if stat == "date":
                    link = col.select_one("a")
                    data["match_date"] = (link.get_text(strip=True)
                                          if link else col.get_text(strip=True))

                elif stat == "home_team":
                    link = col.select_one("a")
                    data["home_team"] = (link.get_text(strip=True)
                                         if link else col.get_text(strip=True))
                    if link and link.get("href"):
                        data["home_team_url"] = FBREF_BASE + link["href"]

                elif stat == "away_team":
                    link = col.select_one("a")
                    data["away_team"] = (link.get_text(strip=True)
                                         if link else col.get_text(strip=True))
                    if link and link.get("href"):
                        data["away_team_url"] = FBREF_BASE + link["href"]

                elif stat == "score":
                    score_text = col.get_text(strip=True)
                    goals = self._parse_score(score_text)
                    if goals:
                        data["home_goals"], data["away_goals"] = goals

                elif stat == "venue":
                    data["venue"] = col.get_text(strip=True)

                elif stat == "match_report":
                    link = col.select_one("a")
                    if link and link.get("href"):
                        data["match_report_url"] = FBREF_BASE + link["href"]

                elif stat == "gameweek":
                    gw = col.get_text(strip=True)
                    data["gameweek"] = int(gw) if gw.isdigit() else gw

                elif stat == "xg_a":
                    xg = col.get_text(strip=True)
                    if xg:
                        try:
                            data["home_xg"] = float(xg)
                        except ValueError:
                            pass

                elif stat == "xg_b":
                    xg = col.get_text(strip=True)
                    if xg:
                        try:
                            data["away_xg"] = float(xg)
                        except ValueError:
                            pass

            if "home_team" in data and "away_team" in data:
                return data
            return None

        except Exception as e:
            logger.debug(f"Failed to parse club match row: {e}")
            return None

    @staticmethod
    def _parse_score(score_text: str) -> tuple[int, int] | None:
        score_text = score_text.split("(")[0].strip()
        m = re.match(r"(\d+)\s*[:\-–]\s*(\d+)", score_text)
        if m:
            return int(m.group(1)), int(m.group(2))
        return None

    # ------------------------------------------------------------------
    # Squad / player rosters with nationality
    # ------------------------------------------------------------------

    def scrape_squad_roster(self, team_url: str,
                            season: str) -> list[dict[str, Any]]:
        """Scrape a club's squad roster, including player nationalities.

        This is the key link: we know which club each player is at,
        and their nationality tells us which national team they represent.
        """
        # Construct stats URL from team URL
        # e.g. /en/squads/18bb7c10/2023-2024/Arsenal-Stats
        if "/squads/" not in team_url:
            return []

        html = self.fetch(team_url)
        soup = BeautifulSoup(html, "lxml")

        players = []
        table = soup.select_one("#stats_standard_combined, #stats_standard_9")
        if table is None:
            table = soup.select_one("table.stats_table")
        if table is None:
            return []

        rows = table.select("tbody tr:not(.thead)")
        for row in rows:
            player = self._parse_player_row(row, season)
            if player:
                players.append(player)

        logger.info(f"Roster from {team_url}: {len(players)} players")
        return players

    def _parse_player_row(self, row, season: str) -> dict[str, Any] | None:
        """Parse a player row from a squad stats table."""
        try:
            data: dict[str, Any] = {"season": season}

            for col in row.select("td, th"):
                stat = col.get("data-stat", "")

                if stat == "player":
                    link = col.select_one("a")
                    data["player_name"] = (link.get_text(strip=True)
                                           if link else col.get_text(strip=True))
                    if link and link.get("href"):
                        data["player_url"] = FBREF_BASE + link["href"]

                elif stat == "nationality":
                    # Nationality is often a flag + country code
                    text = col.get_text(strip=True)
                    # Extract country code (last 3 chars are usually the code)
                    data["nationality"] = text[-3:].strip() if len(text) >= 2 else text

                elif stat == "position":
                    data["position"] = col.get_text(strip=True)

                elif stat == "age":
                    age_text = col.get_text(strip=True)
                    # FBref age format: "25-123" (years-days)
                    if "-" in age_text:
                        data["age"] = int(age_text.split("-")[0])
                    elif age_text.isdigit():
                        data["age"] = int(age_text)

                elif stat == "games":
                    val = col.get_text(strip=True)
                    data["appearances"] = int(val) if val.isdigit() else 0

                elif stat == "games_starts":
                    val = col.get_text(strip=True)
                    data["starts"] = int(val) if val.isdigit() else 0

                elif stat == "minutes":
                    val = col.get_text(strip=True).replace(",", "")
                    data["minutes"] = int(val) if val.isdigit() else 0

                elif stat == "goals":
                    val = col.get_text(strip=True)
                    data["goals"] = int(val) if val.isdigit() else 0

                elif stat == "assists":
                    val = col.get_text(strip=True)
                    data["assists"] = int(val) if val.isdigit() else 0

                elif stat == "xg":
                    val = col.get_text(strip=True)
                    try:
                        data["xg"] = float(val)
                    except (ValueError, TypeError):
                        pass

                elif stat == "xg_assist":
                    val = col.get_text(strip=True)
                    try:
                        data["xa"] = float(val)
                    except (ValueError, TypeError):
                        pass

            if "player_name" in data:
                return data
            return None

        except Exception as e:
            logger.debug(f"Failed to parse player row: {e}")
            return None

    # ------------------------------------------------------------------
    # League standings
    # ------------------------------------------------------------------

    def scrape_league_standings(self, league_key: str,
                                season: str) -> list[dict[str, Any]]:
        """Scrape league standings (points, GD, xG) for a season."""
        info = LEAGUE_IDS.get(league_key)
        if info is None:
            return []

        league_name_url = info["name"].replace(" ", "-")
        url = (f"{FBREF_BASE}/en/comps/{info['id']}/{season}"
               f"/{season}-{league_name_url}-Stats")

        html = self.fetch(url)
        soup = BeautifulSoup(html, "lxml")

        standings = []
        table = soup.select_one("#results{}_overall".format(season))
        if table is None:
            # Try generic standings table
            table = soup.select_one("table.stats_table")
        if table is None:
            return []

        rows = table.select("tbody tr:not(.thead)")
        for row in rows:
            team_data: dict[str, Any] = {
                "league": league_key,
                "season": season,
            }
            for col in row.select("td, th"):
                stat = col.get("data-stat", "")
                val = col.get_text(strip=True)

                if stat == "team":
                    link = col.select_one("a")
                    team_data["club"] = (link.get_text(strip=True)
                                         if link else val)
                elif stat == "rank":
                    team_data["league_position"] = int(val) if val.isdigit() else 0
                elif stat == "points":
                    team_data["points"] = int(val) if val.isdigit() else 0
                elif stat == "wins":
                    team_data["wins"] = int(val) if val.isdigit() else 0
                elif stat == "losses":
                    team_data["losses"] = int(val) if val.isdigit() else 0
                elif stat == "draws":
                    team_data["draws"] = int(val) if val.isdigit() else 0
                elif stat == "goals_for":
                    team_data["goals_for"] = int(val) if val.isdigit() else 0
                elif stat == "goals_against":
                    team_data["goals_against"] = int(val) if val.isdigit() else 0
                elif stat == "goal_diff":
                    team_data["goal_diff"] = int(val) if val.lstrip("-").isdigit() else 0
                elif stat == "xg_for":
                    try:
                        team_data["xg_for"] = float(val)
                    except ValueError:
                        pass
                elif stat == "xg_against":
                    try:
                        team_data["xg_against"] = float(val)
                    except ValueError:
                        pass

            if "club" in team_data:
                standings.append(team_data)

        return standings

    # ------------------------------------------------------------------
    # Bulk scrape
    # ------------------------------------------------------------------

    def scrape_all_leagues(self, seasons: list[str] | None = None,
                           leagues: list[str] | None = None) -> pd.DataFrame:
        """Scrape match results across leagues and seasons.

        Args:
            seasons: e.g. ["2022-2023", "2023-2024"]. Defaults to last 3 seasons.
            leagues: Keys from LEAGUE_IDS. Defaults to all 5.

        Returns:
            DataFrame of all club match results.
        """
        if seasons is None:
            seasons = ["2021-2022", "2022-2023", "2023-2024"]
        if leagues is None:
            leagues = list(LEAGUE_IDS.keys())

        all_matches = []
        for league in leagues:
            for season in seasons:
                matches = self.scrape_league_season(league, season)
                all_matches.extend(matches)

        df = pd.DataFrame(all_matches)
        if not df.empty and "match_date" in df.columns:
            df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
            df = df.sort_values("match_date").reset_index(drop=True)
            df["source"] = "fbref_clubs"

        logger.info(f"Total: {len(df)} club matches across {len(leagues)} leagues")
        return df

    def scrape_all(self, **kwargs) -> pd.DataFrame:
        """Default scrape_all delegates to scrape_all_leagues."""
        return self.scrape_all_leagues(**kwargs)
