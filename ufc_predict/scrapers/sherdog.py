"""Scraper for Sherdog.com — complete fight history including pre-UFC bouts."""

from __future__ import annotations

import re
import logging
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

from .base import BaseScraper
from ..utils import parse_date, normalize_name

logger = logging.getLogger(__name__)

BASE_URL = "https://www.sherdog.com"
SEARCH_URL = f"{BASE_URL}/stats/fightfinder"


class SherdogScraper(BaseScraper):
    SOURCE_NAME = "sherdog"

    def search_fighter(self, name: str) -> list[dict[str, str]]:
        """Search for a fighter by name, return list of matches."""
        url = f"{SEARCH_URL}?SearchTxt={name.replace(' ', '+')}"
        html = self.fetch(url)
        soup = BeautifulSoup(html, "lxml")

        results = []
        table = soup.select_one("table.fightfinder_result")
        if not table:
            return results

        rows = table.select("tr")[1:]  # skip header
        for row in rows:
            cols = row.select("td")
            if len(cols) < 3:
                continue
            link = cols[0].select_one("a")
            if link:
                results.append({
                    "name": link.get_text(strip=True),
                    "url": BASE_URL + link["href"],
                    "nickname": cols[1].get_text(strip=True) if len(cols) > 1 else "",
                    "association": cols[2].get_text(strip=True) if len(cols) > 2 else "",
                })
        return results

    def scrape_fighter(self, fighter_url: str) -> dict[str, Any]:
        """Scrape complete fighter profile and fight history from Sherdog."""
        html = self.fetch(fighter_url)
        soup = BeautifulSoup(html, "lxml")

        info: dict[str, Any] = {"sherdog_url": fighter_url}

        name_el = soup.select_one("span.fn")
        if name_el:
            info["name"] = name_el.get_text(strip=True)

        nickname_el = soup.select_one("span.nickname")
        if nickname_el:
            info["nickname"] = nickname_el.get_text(strip=True)

        bio_section = soup.select_one("div.bio_graph")
        if bio_section:
            items = bio_section.select("span")
            for item in items:
                title = item.select_one("strong")
                if not title:
                    continue
                label = title.get_text(strip=True).lower()
                value = item.get_text(strip=True).replace(title.get_text(strip=True), "").strip()
                if "age" in label or "birthday" in label:
                    info["dob"] = value
                elif "height" in label:
                    info["height"] = value
                elif "weight" in label:
                    info["weight"] = value
                elif "association" in label or "camp" in label:
                    info["camp"] = value
                elif "nationality" in label or "country" in label:
                    info["nationality"] = value
                elif "weight class" in label:
                    info["weight_class"] = value
                elif "locality" in label:
                    info["locality"] = value

        info["fights"] = self._parse_fight_history(soup)
        return info

    def _parse_fight_history(self, soup: BeautifulSoup) -> list[dict[str, Any]]:
        """Parse all fights from a fighter's Sherdog page."""
        fights = []

        fight_tables = soup.select("div.fight_history")
        if not fight_tables:
            fight_tables = soup.select("section.fightHistory")

        for table in fight_tables:
            rows = table.select("tr")
            for row in rows:
                cols = row.select("td")
                if len(cols) < 6:
                    continue

                result_el = cols[0]
                result_text = result_el.get_text(strip=True).lower()

                opponent_el = cols[1].select_one("a")
                opponent = opponent_el.get_text(strip=True) if opponent_el else cols[1].get_text(strip=True)
                opponent_url = ""
                if opponent_el and opponent_el.get("href"):
                    opponent_url = BASE_URL + opponent_el["href"]

                event_el = cols[2].select_one("a")
                event_name = event_el.get_text(strip=True) if event_el else cols[2].get_text(strip=True)

                method = cols[3].get_text(strip=True) if len(cols) > 3 else ""
                round_num = cols[4].get_text(strip=True) if len(cols) > 4 else ""
                time = cols[5].get_text(strip=True) if len(cols) > 5 else ""

                date_el = None
                if len(cols) > 6:
                    date_el = cols[6]
                elif event_el:
                    date_span = cols[2].select_one("span.sub_line")
                    if date_span:
                        date_el = date_span

                fight_date = date_el.get_text(strip=True) if date_el else ""

                fight = {
                    "result": result_text,
                    "opponent": opponent,
                    "opponent_url": opponent_url,
                    "event": event_name,
                    "method": method,
                    "round": round_num,
                    "time": time,
                    "date": fight_date,
                }
                fights.append(fight)

        return fights

    def scrape_fighter_full_record(self, name: str) -> dict[str, Any] | None:
        """Search and scrape full record for a fighter by name."""
        results = self.search_fighter(name)
        if not results:
            logger.warning(f"No Sherdog results for: {name}")
            return None

        best_match = None
        name_norm = normalize_name(name)
        for r in results:
            if normalize_name(r["name"]) == name_norm:
                best_match = r
                break
        if not best_match:
            best_match = results[0]

        return self.scrape_fighter(best_match["url"])

    def build_fight_history_df(self, fighter_data: dict[str, Any]) -> pd.DataFrame:
        """Convert a fighter's scraped data to a DataFrame of their fights."""
        if not fighter_data or "fights" not in fighter_data:
            return pd.DataFrame()

        df = pd.DataFrame(fighter_data["fights"])
        if not df.empty:
            df["fighter"] = fighter_data.get("name", "")
            df["fighter_norm"] = normalize_name(fighter_data.get("name", ""))
            df["source"] = "sherdog"
            if "date" in df.columns:
                df["parsed_date"] = df["date"].apply(parse_date)
        return df
