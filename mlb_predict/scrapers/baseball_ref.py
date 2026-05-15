"""Scraper for Baseball-Reference.com — MLB game logs and stats.

Baseball Reference is the primary source for historical MLB data including
game results, box scores, pitcher stats, and team standings.
"""

from __future__ import annotations

import re
import logging
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

from sports_predict.scrapers.base import BaseScraper

logger = logging.getLogger(__name__)

BREF_BASE = "https://www.baseball-reference.com"


class BaseballReferenceScraper(BaseScraper):
    """Scraper for MLB data from Baseball-Reference.com."""

    SOURCE_NAME = "baseball_ref"

    def __init__(self, cache_dir: str = "data/cache",
                 delay_range: tuple[float, float] = (3.0, 6.0)):
        # Baseball Reference has strict rate limiting
        super().__init__(cache_dir=cache_dir, delay_range=delay_range)

    def scrape_season_schedule(self, year: int) -> list[dict[str, Any]]:
        """Scrape all game results for a given MLB season."""
        url = f"{BREF_BASE}/leagues/majors/{year}-schedule.shtml"
        html = self.fetch(url)
        soup = BeautifulSoup(html, "lxml")

        games = []
        # Schedule page has game entries grouped by date
        game_entries = soup.select("div.game_summary, p.game")
        if not game_entries:
            # Try the table format instead
            games = self._parse_schedule_table(soup, year)
        else:
            for entry in game_entries:
                game = self._parse_game_summary(entry, year)
                if game:
                    games.append(game)

        logger.info(f"{year} season: scraped {len(games)} games")
        return games

    def _parse_schedule_table(self, soup: BeautifulSoup, year: int) -> list[dict]:
        """Parse schedule from the table format."""
        games = []
        current_date = ""

        for el in soup.select("h3, p.game"):
            if el.name == "h3":
                current_date = el.get_text(strip=True)
            elif el.name == "p":
                game = self._parse_game_line(el, current_date, year)
                if game:
                    games.append(game)

        return games

    def _parse_game_summary(self, entry, year: int) -> dict[str, Any] | None:
        """Parse a game_summary div into a game dict."""
        try:
            teams = entry.select("td.right, td.left")
            if len(teams) < 2:
                links = entry.select("a")
                if len(links) >= 2:
                    return {
                        "season": year,
                        "away_team": links[0].get_text(strip=True),
                        "home_team": links[1].get_text(strip=True),
                    }
                return None

            away_name = teams[0].get_text(strip=True)
            home_name = teams[1].get_text(strip=True)

            scores = entry.select("td.right strong, td.left strong")
            away_runs = int(scores[0].get_text(strip=True)) if len(scores) > 0 else 0
            home_runs = int(scores[1].get_text(strip=True)) if len(scores) > 1 else 0

            return {
                "season": year,
                "away_team": away_name,
                "home_team": home_name,
                "away_runs": away_runs,
                "home_runs": home_runs,
            }
        except Exception as e:
            logger.debug(f"Failed to parse game summary: {e}")
            return None

    def _parse_game_line(self, el, date_str: str, year: int) -> dict[str, Any] | None:
        """Parse a single game line from schedule page."""
        try:
            links = el.select("a")
            text = el.get_text(strip=True)

            # Pattern: "TeamA (runs) @ TeamB (runs)"
            match = re.search(r"(.+?)\s*\((\d+)\)\s*[@vs.]+\s*(.+?)\s*\((\d+)\)", text)
            if match:
                return {
                    "season": year,
                    "game_date": date_str,
                    "away_team": match.group(1).strip(),
                    "away_runs": int(match.group(2)),
                    "home_team": match.group(3).strip(),
                    "home_runs": int(match.group(4)),
                }

            # Fallback: just grab team names from links
            if len(links) >= 2:
                return {
                    "season": year,
                    "game_date": date_str,
                    "away_team": links[0].get_text(strip=True),
                    "home_team": links[1].get_text(strip=True),
                }
            return None
        except Exception:
            return None

    def scrape_pitcher_game_log(self, pitcher_url: str,
                                year: int) -> list[dict[str, Any]]:
        """Scrape a pitcher's game log for a given season."""
        url = f"{pitcher_url}/gamelog/{year}"
        html = self.fetch(url)
        soup = BeautifulSoup(html, "lxml")

        entries = []
        table = soup.select_one("#pitching_gamelogs")
        if table is None:
            return entries

        rows = table.select("tbody tr:not(.thead)")
        for row in rows:
            if row.get("class") and "spacer" in row.get("class", []):
                continue
            entry = {}
            for td in row.select("td, th"):
                stat = td.get("data-stat", "")
                val = td.get_text(strip=True)
                if stat and val:
                    entry[stat] = val
            if entry:
                entries.append(entry)

        return entries

    def scrape_team_batting(self, team_abbr: str, year: int) -> dict[str, Any]:
        """Scrape team batting stats for a season."""
        url = f"{BREF_BASE}/teams/{team_abbr}/{year}.shtml"
        html = self.fetch(url)
        soup = BeautifulSoup(html, "lxml")

        stats: dict[str, Any] = {"team": team_abbr, "season": year}

        batting = soup.select_one("#team_batting")
        if batting:
            footer = batting.select_one("tfoot tr")
            if footer:
                for td in footer.select("td"):
                    stat = td.get("data-stat", "")
                    val = td.get_text(strip=True)
                    if stat and val:
                        stats[f"bat_{stat}"] = val

        return stats

    def scrape_all(self, years: list[int] | None = None) -> pd.DataFrame:
        """Scrape all game results for given seasons."""
        if years is None:
            years = list(range(2020, 2026))

        all_games = []
        for year in years:
            games = self.scrape_season_schedule(year)
            all_games.extend(games)

        df = pd.DataFrame(all_games)
        if not df.empty:
            if "game_date" in df.columns:
                df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
                df = df.sort_values("game_date").reset_index(drop=True)
            df["source"] = "baseball_ref"

        logger.info(f"Total: {len(df)} MLB games scraped")
        return df
