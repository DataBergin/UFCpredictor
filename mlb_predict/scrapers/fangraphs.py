"""Scraper for FanGraphs — advanced MLB statistics.

FanGraphs provides advanced metrics like FIP, xFIP, wOBA, WAR, Statcast
data, and park factors that go beyond traditional box score stats.
"""

from __future__ import annotations

import re
import logging
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

from sports_predict.scrapers.base import BaseScraper

logger = logging.getLogger(__name__)

FANGRAPHS_BASE = "https://www.fangraphs.com"


class FanGraphsScraper(BaseScraper):
    """Scraper for advanced MLB stats from FanGraphs."""

    SOURCE_NAME = "fangraphs"

    def __init__(self, cache_dir: str = "data/cache",
                 delay_range: tuple[float, float] = (3.0, 6.0)):
        super().__init__(cache_dir=cache_dir, delay_range=delay_range)

    def scrape_pitcher_leaderboard(self, year: int,
                                   min_ip: int = 20) -> pd.DataFrame:
        """Scrape pitcher leaderboard with advanced stats for a season.

        Returns DataFrame with: Name, Team, W, L, ERA, FIP, xFIP, SIERA,
        K/9, BB/9, HR/9, BABIP, WAR, etc.
        """
        # FanGraphs leaderboard API endpoint
        url = (f"{FANGRAPHS_BASE}/leaders.aspx?pos=all&stats=pit"
               f"&lg=all&qual={min_ip}&type=8&season={year}"
               f"&month=0&season1={year}&ind=0&page=1_500")
        html = self.fetch(url)
        soup = BeautifulSoup(html, "lxml")

        rows_data = []
        table = soup.select_one("table.rgMasterTable")
        if table is None:
            # Try alternate table selector
            table = soup.select_one("#LeaderBoard1_dg1_ctl00")

        if table is None:
            logger.warning(f"No pitcher leaderboard table found for {year}")
            return pd.DataFrame()

        headers = []
        header_row = table.select_one("thead tr")
        if header_row:
            headers = [th.get_text(strip=True) for th in header_row.select("th")]

        rows = table.select("tbody tr")
        for row in rows:
            cells = row.select("td")
            if not cells:
                continue
            values = [c.get_text(strip=True) for c in cells]
            if headers:
                rows_data.append(dict(zip(headers, values)))
            else:
                rows_data.append({"raw_" + str(i): v for i, v in enumerate(values)})

        df = pd.DataFrame(rows_data)
        if not df.empty:
            df["season"] = year
            df["source"] = "fangraphs"

        logger.info(f"FanGraphs {year}: {len(df)} pitchers scraped")
        return df

    def scrape_batter_leaderboard(self, year: int,
                                  min_pa: int = 50) -> pd.DataFrame:
        """Scrape batter leaderboard with advanced stats for a season."""
        url = (f"{FANGRAPHS_BASE}/leaders.aspx?pos=all&stats=bat"
               f"&lg=all&qual={min_pa}&type=8&season={year}"
               f"&month=0&season1={year}&ind=0&page=1_500")
        html = self.fetch(url)
        soup = BeautifulSoup(html, "lxml")

        rows_data = []
        table = soup.select_one("table.rgMasterTable")
        if table is None:
            table = soup.select_one("#LeaderBoard1_dg1_ctl00")

        if table is None:
            logger.warning(f"No batter leaderboard table found for {year}")
            return pd.DataFrame()

        headers = []
        header_row = table.select_one("thead tr")
        if header_row:
            headers = [th.get_text(strip=True) for th in header_row.select("th")]

        rows = table.select("tbody tr")
        for row in rows:
            cells = row.select("td")
            if not cells:
                continue
            values = [c.get_text(strip=True) for c in cells]
            if headers:
                rows_data.append(dict(zip(headers, values)))
            else:
                rows_data.append({"raw_" + str(i): v for i, v in enumerate(values)})

        df = pd.DataFrame(rows_data)
        if not df.empty:
            df["season"] = year
            df["source"] = "fangraphs"

        logger.info(f"FanGraphs {year}: {len(df)} batters scraped")
        return df

    def scrape_park_factors(self, year: int) -> pd.DataFrame:
        """Scrape park factors for a season."""
        url = f"{FANGRAPHS_BASE}/guts.aspx?type=pf&teamid=0&season={year}"
        html = self.fetch(url)
        soup = BeautifulSoup(html, "lxml")

        rows_data = []
        table = soup.select_one("table.rgMasterTable")
        if table is None:
            logger.warning(f"No park factors table found for {year}")
            return pd.DataFrame()

        headers = []
        header_row = table.select_one("thead tr")
        if header_row:
            headers = [th.get_text(strip=True) for th in header_row.select("th")]

        rows = table.select("tbody tr")
        for row in rows:
            cells = row.select("td")
            if not cells:
                continue
            values = [c.get_text(strip=True) for c in cells]
            if headers:
                rows_data.append(dict(zip(headers, values)))

        df = pd.DataFrame(rows_data)
        if not df.empty:
            df["season"] = year
            df["source"] = "fangraphs_pf"

        logger.info(f"FanGraphs park factors {year}: {len(df)} parks")
        return df

    def scrape_all(self, years: list[int] | None = None) -> pd.DataFrame:
        """Scrape pitcher leaderboards for all given years."""
        if years is None:
            years = list(range(2020, 2026))

        frames = []
        for year in years:
            df = self.scrape_pitcher_leaderboard(year)
            if not df.empty:
                frames.append(df)

        if frames:
            result = pd.concat(frames, ignore_index=True)
            logger.info(f"Total: {len(result)} pitcher-season records")
            return result
        return pd.DataFrame()
