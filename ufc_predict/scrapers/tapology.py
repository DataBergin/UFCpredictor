"""Scraper for Tapology.com — scheduled fights, short-notice flags, weight data."""

from __future__ import annotations

import re
import logging
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

from .base import BaseScraper
from ..utils import parse_date, normalize_name

logger = logging.getLogger(__name__)

BASE_URL = "https://www.tapology.com"


class TapologyScraper(BaseScraper):
    SOURCE_NAME = "tapology"

    def scrape_upcoming_events(self) -> list[dict[str, Any]]:
        """Scrape upcoming UFC events from Tapology."""
        url = f"{BASE_URL}/fightcenter?group=ufc&schedule=upcoming"
        html = self.fetch(url)
        soup = BeautifulSoup(html, "lxml")

        events = []
        event_items = soup.select("div.fightcenterEvents li, section.fcListing")

        for item in event_items:
            link = item.select_one("a")
            if not link:
                continue
            event_url = link.get("href", "")
            if not event_url.startswith("http"):
                event_url = BASE_URL + event_url
            event_name = link.get_text(strip=True)

            date_el = item.select_one("span.datetime, span.date")
            event_date = date_el.get_text(strip=True) if date_el else ""

            location_el = item.select_one("span.location, span.venue")
            location = location_el.get_text(strip=True) if location_el else ""

            events.append({
                "event_url": event_url,
                "event_name": event_name,
                "event_date": event_date,
                "location": location,
            })

        return events

    def scrape_event_fights(self, event_url: str) -> list[dict[str, Any]]:
        """Scrape fight card for a specific event."""
        html = self.fetch(event_url)
        soup = BeautifulSoup(html, "lxml")

        fights = []
        fight_cards = soup.select("li.fightCard, div.matchup, ul.fightCard li")

        for i, card in enumerate(fight_cards):
            fighters = card.select("a.link_underline, span.name a, div.fighter a")
            if len(fighters) < 2:
                continue

            fighter_a = fighters[0].get_text(strip=True)
            fighter_b = fighters[1].get_text(strip=True)

            weight_el = card.select_one("span.weight_class, span.pointed, span.pointed_gray")
            weight_class = weight_el.get_text(strip=True) if weight_el else ""

            bout_type_el = card.select_one("span.boutType, span.division")
            bout_type = bout_type_el.get_text(strip=True) if bout_type_el else ""

            is_main = "main" in bout_type.lower() or i == 0
            is_comain = "co-main" in bout_type.lower() or i == 1

            short_notice = self._detect_short_notice(card)

            fight = {
                "fighter_a": fighter_a,
                "fighter_b": fighter_b,
                "fighter_a_norm": normalize_name(fighter_a),
                "fighter_b_norm": normalize_name(fighter_b),
                "weight_class": weight_class,
                "bout_type": bout_type,
                "is_main_event": is_main,
                "is_co_main": is_comain,
                "card_position": i,
                "short_notice": short_notice,
            }
            fights.append(fight)

        return fights

    def _detect_short_notice(self, card_el) -> bool:
        """Check if a fight has short-notice replacement indicators."""
        text = card_el.get_text().lower()
        indicators = ["short notice", "late replacement", "replacement", "stepped in"]
        return any(ind in text for ind in indicators)

    def scrape_fighter_profile(self, fighter_name: str) -> dict[str, Any] | None:
        """Scrape fighter profile from Tapology for weight/camp data."""
        search_url = f"{BASE_URL}/search?term={fighter_name.replace(' ', '+')}&mainSearchFilter=fighters"
        html = self.fetch(search_url)
        soup = BeautifulSoup(html, "lxml")

        first_result = soup.select_one("td.altCol a, div.searchResult a")
        if not first_result:
            return None

        profile_url = first_result.get("href", "")
        if not profile_url.startswith("http"):
            profile_url = BASE_URL + profile_url

        return self._scrape_profile_page(profile_url)

    def _scrape_profile_page(self, url: str) -> dict[str, Any]:
        """Parse a fighter's Tapology profile page."""
        html = self.fetch(url)
        soup = BeautifulSoup(html, "lxml")

        info: dict[str, Any] = {"tapology_url": url}

        name_el = soup.select_one("h1, div.fighterUpcomingHeader h2")
        if name_el:
            info["name"] = name_el.get_text(strip=True)

        details = soup.select("li.detail, span.pointed")
        for detail in details:
            text = detail.get_text(strip=True).lower()
            if "weight" in text and "miss" in text:
                info["weight_miss"] = True
            if "gym" in text or "camp" in text or "team" in text:
                info["camp"] = detail.get_text(strip=True)
            if "country" in text or "nation" in text:
                info["nationality"] = detail.get_text(strip=True)

        weight_miss_section = soup.select("span.pointed_red, span.miss")
        info["weight_misses_count"] = len(weight_miss_section)

        return info

    def scrape_all(self) -> pd.DataFrame:
        """Scrape upcoming events and their fight cards."""
        events = self.scrape_upcoming_events()
        all_fights = []

        for event in events:
            try:
                fights = self.scrape_event_fights(event["event_url"])
                for fight in fights:
                    fight["event_name"] = event["event_name"]
                    fight["event_date"] = event["event_date"]
                    fight["location"] = event.get("location", "")
                all_fights.extend(fights)
            except Exception as e:
                logger.warning(f"Failed to scrape Tapology event {event['event_name']}: {e}")

        df = pd.DataFrame(all_fights)
        if not df.empty:
            df["source"] = "tapology"
        return df
