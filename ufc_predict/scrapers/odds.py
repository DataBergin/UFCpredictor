"""Scraper for historical betting odds from BestFightOdds.com."""

from __future__ import annotations

import re
import logging
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

from .base import BaseScraper
from ..utils import normalize_name, american_to_implied_prob, devig_odds

logger = logging.getLogger(__name__)

BASE_URL = "https://www.bestfightodds.com"


class OddsScraper(BaseScraper):
    SOURCE_NAME = "odds"

    def scrape_event_odds(self, event_url: str) -> list[dict[str, Any]]:
        """Scrape odds for all fights in an event."""
        html = self.fetch(event_url)
        soup = BeautifulSoup(html, "lxml")

        fights = []
        table = soup.select_one("table.odds-table, table.content-list")
        if not table:
            return fights

        rows = table.select("tr")
        current_fight: dict[str, Any] = {}

        for row in rows:
            name_el = row.select_one("th.opponentCell a, td.name a")
            if not name_el:
                continue

            fighter_name = name_el.get_text(strip=True)
            odds_cells = row.select("td.pointed, td.bestOddsLink, td.odds")
            odds_values = []
            for cell in odds_cells:
                text = cell.get_text(strip=True)
                if text and text != "-":
                    try:
                        odds_values.append(int(text.replace("+", "")))
                    except ValueError:
                        odds_values.append(None)
                else:
                    odds_values.append(None)

            if not current_fight.get("fighter_a"):
                current_fight = {
                    "fighter_a": fighter_name,
                    "fighter_a_norm": normalize_name(fighter_name),
                    "fighter_a_odds": odds_values,
                }
            else:
                current_fight["fighter_b"] = fighter_name
                current_fight["fighter_b_norm"] = normalize_name(fighter_name)
                current_fight["fighter_b_odds"] = odds_values

                current_fight.update(self._compute_odds_features(current_fight))
                fights.append(current_fight)
                current_fight = {}

        return fights

    def _compute_odds_features(self, fight: dict[str, Any]) -> dict[str, Any]:
        """Compute derived odds features: opening, closing, movement, devigged probs."""
        features: dict[str, Any] = {}

        a_odds = [o for o in fight.get("fighter_a_odds", []) if o is not None]
        b_odds = [o for o in fight.get("fighter_b_odds", []) if o is not None]

        if a_odds and b_odds:
            # Opening odds (first available)
            features["opening_odds_a"] = a_odds[0]
            features["opening_odds_b"] = b_odds[0]
            features["opening_prob_a"] = american_to_implied_prob(a_odds[0])
            features["opening_prob_b"] = american_to_implied_prob(b_odds[0])

            # Closing odds (last available)
            features["closing_odds_a"] = a_odds[-1]
            features["closing_odds_b"] = b_odds[-1]
            features["closing_prob_a_raw"] = american_to_implied_prob(a_odds[-1])
            features["closing_prob_b_raw"] = american_to_implied_prob(b_odds[-1])

            # Devigged closing probabilities
            devig_a, devig_b = devig_odds(
                features["closing_prob_a_raw"],
                features["closing_prob_b_raw"]
            )
            features["closing_prob_a"] = devig_a
            features["closing_prob_b"] = devig_b

            # Line movement
            features["line_move_a"] = features["closing_prob_a"] - features["opening_prob_a"]
            features["line_move_b"] = features["closing_prob_b"] - features["opening_prob_b"]

            # Consensus spread (max odds diff across books)
            if len(a_odds) > 1:
                features["odds_range_a"] = max(a_odds) - min(a_odds)
                features["odds_range_b"] = max(b_odds) - min(b_odds)
            else:
                features["odds_range_a"] = 0
                features["odds_range_b"] = 0

        return features

    def scrape_event_list(self) -> list[dict[str, str]]:
        """Get list of all events with odds data."""
        url = f"{BASE_URL}/events"
        html = self.fetch(url)
        soup = BeautifulSoup(html, "lxml")

        events = []
        links = soup.select("a[href*='/events/']")
        for link in links:
            href = link.get("href", "")
            name = link.get_text(strip=True)
            if name and "UFC" in name.upper():
                full_url = href if href.startswith("http") else BASE_URL + href
                events.append({"event_name": name, "event_url": full_url})

        return events

    def scrape_fighter_odds_history(self, fighter_name: str) -> list[dict[str, Any]]:
        """Scrape historical odds for a specific fighter."""
        search_url = f"{BASE_URL}/search?query={fighter_name.replace(' ', '+')}"
        html = self.fetch(search_url)
        soup = BeautifulSoup(html, "lxml")

        fighter_link = soup.select_one("a[href*='/fighters/']")
        if not fighter_link:
            return []

        fighter_url = fighter_link.get("href", "")
        if not fighter_url.startswith("http"):
            fighter_url = BASE_URL + fighter_url

        return self._scrape_fighter_page(fighter_url)

    def _scrape_fighter_page(self, url: str) -> list[dict[str, Any]]:
        """Parse odds history from a fighter's BFO page."""
        html = self.fetch(url)
        soup = BeautifulSoup(html, "lxml")

        fights = []
        table = soup.select_one("table.content-list")
        if not table:
            return fights

        rows = table.select("tr")
        for row in rows:
            cols = row.select("td")
            if len(cols) < 4:
                continue

            opponent_el = cols[0].select_one("a")
            if not opponent_el:
                continue

            odds_el = cols[1] if len(cols) > 1 else None
            event_el = cols[2] if len(cols) > 2 else None
            date_el = cols[3] if len(cols) > 3 else None

            odds_text = odds_el.get_text(strip=True) if odds_el else ""
            try:
                closing_odds = int(odds_text.replace("+", ""))
            except (ValueError, AttributeError):
                closing_odds = None

            fight = {
                "opponent": opponent_el.get_text(strip=True),
                "closing_odds": closing_odds,
                "event": event_el.get_text(strip=True) if event_el else "",
                "date": date_el.get_text(strip=True) if date_el else "",
            }
            if closing_odds is not None:
                fight["implied_prob"] = american_to_implied_prob(closing_odds)
            fights.append(fight)

        return fights

    def scrape_all(self) -> pd.DataFrame:
        """Scrape odds for all available UFC events."""
        events = self.scrape_event_list()
        all_fights = []

        for event in events:
            try:
                fights = self.scrape_event_odds(event["event_url"])
                for fight in fights:
                    fight["event_name"] = event["event_name"]
                all_fights.extend(fights)
            except Exception as e:
                logger.warning(f"Failed to scrape odds for {event['event_name']}: {e}")

        df = pd.DataFrame(all_fights)
        if not df.empty:
            df["source"] = "bestfightodds"
        logger.info(f"Scraped odds for {len(df)} fights")
        return df
