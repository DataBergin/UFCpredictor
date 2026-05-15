"""Scraper for FBref.com — international football match data.

FBref provides comprehensive match results and statistics for World Cup,
Euros, Copa America, and other international tournaments.
"""

from __future__ import annotations

import re
import logging
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

from sports_predict.scrapers.base import BaseScraper

logger = logging.getLogger(__name__)

# FBref URLs for international tournaments
FBREF_BASE = "https://fbref.com"
WORLD_CUP_URLS = {
    2022: "/en/comps/1/2022/schedule/2022-FIFA-World-Cup-Scores-and-Fixtures",
    2018: "/en/comps/1/2018/schedule/2018-FIFA-World-Cup-Scores-and-Fixtures",
    2014: "/en/comps/1/2014/schedule/2014-FIFA-World-Cup-Scores-and-Fixtures",
    2010: "/en/comps/1/2010/schedule/2010-FIFA-World-Cup-Scores-and-Fixtures",
}


class FBrefScraper(BaseScraper):
    """Scraper for international football results from FBref."""

    SOURCE_NAME = "fbref"

    def __init__(self, cache_dir: str = "data/cache",
                 delay_range: tuple[float, float] = (3.0, 6.0)):
        # FBref is strict on rate limiting — use longer delays
        super().__init__(cache_dir=cache_dir, delay_range=delay_range)

    def scrape_world_cup(self, year: int) -> list[dict[str, Any]]:
        """Scrape all matches from a specific World Cup."""
        path = WORLD_CUP_URLS.get(year)
        if path is None:
            logger.warning(f"No URL configured for {year} World Cup")
            return []

        url = f"{FBREF_BASE}{path}"
        html = self.fetch(url)
        soup = BeautifulSoup(html, "lxml")

        matches = []
        table = soup.select_one("table.stats_table")
        if table is None:
            logger.warning(f"No match table found for {year} World Cup")
            return []

        rows = table.select("tbody tr:not(.thead)")
        for row in rows:
            cols = row.select("td, th")
            if len(cols) < 5:
                continue

            match = self._parse_match_row(cols, year)
            if match:
                matches.append(match)

        logger.info(f"{year} World Cup: scraped {len(matches)} matches")
        return matches

    def _parse_match_row(self, cols: list, year: int) -> dict[str, Any] | None:
        """Parse a single match row from the FBref schedule table."""
        try:
            # Column layout varies, but generally:
            # date, time, home_team, score, away_team, attendance, venue, ...
            data: dict[str, Any] = {"tournament": f"World Cup {year}"}

            for col in cols:
                stat = col.get("data-stat", "")

                if stat == "date":
                    date_link = col.select_one("a")
                    data["match_date"] = date_link.get_text(strip=True) if date_link else col.get_text(strip=True)

                elif stat == "home_team":
                    team_link = col.select_one("a")
                    data["home_team"] = team_link.get_text(strip=True) if team_link else col.get_text(strip=True)

                elif stat == "away_team":
                    team_link = col.select_one("a")
                    data["away_team"] = team_link.get_text(strip=True) if team_link else col.get_text(strip=True)

                elif stat == "score":
                    score_text = col.get_text(strip=True)
                    goals = self._parse_score(score_text)
                    if goals:
                        data["home_goals"], data["away_goals"] = goals

                elif stat == "venue":
                    data["venue"] = col.get_text(strip=True)

                elif stat == "round":
                    data["stage"] = col.get_text(strip=True)

                elif stat == "attendance":
                    att_text = col.get_text(strip=True).replace(",", "")
                    data["attendance"] = int(att_text) if att_text.isdigit() else 0

            if "home_team" in data and "away_team" in data:
                return data
            return None

        except Exception as e:
            logger.debug(f"Failed to parse match row: {e}")
            return None

    def _parse_score(self, score_text: str) -> tuple[int, int] | None:
        """Parse a score string like '2-1' or '3-1 (1-0)' into (home, away)."""
        # Handle extra time scores: take the main score
        score_text = score_text.split("(")[0].strip()
        match = re.match(r"(\d+)\s*[:\-–]\s*(\d+)", score_text)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None

    def scrape_team_stats(self, team_url: str) -> dict[str, Any]:
        """Scrape team-level aggregate stats from FBref."""
        html = self.fetch(team_url)
        soup = BeautifulSoup(html, "lxml")

        stats: dict[str, Any] = {"team_url": team_url}

        # Standard stats table
        standard = soup.select_one("#stats_standard_combined")
        if standard:
            footer = standard.select_one("tfoot tr")
            if footer:
                for td in footer.select("td"):
                    stat = td.get("data-stat", "")
                    val = td.get_text(strip=True)
                    if stat and val:
                        stats[f"std_{stat}"] = val

        return stats

    def scrape_all(self, years: list[int] | None = None) -> pd.DataFrame:
        """Scrape all World Cup matches for given years."""
        if years is None:
            years = sorted(WORLD_CUP_URLS.keys())

        all_matches = []
        for year in years:
            matches = self.scrape_world_cup(year)
            all_matches.extend(matches)

        df = pd.DataFrame(all_matches)
        if not df.empty and "match_date" in df.columns:
            df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
            df = df.sort_values("match_date").reset_index(drop=True)
            df["source"] = "fbref"

        logger.info(f"Total: {len(df)} World Cup matches scraped")
        return df
