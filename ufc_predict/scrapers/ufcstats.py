"""Scraper for ufcstats.com — official UFC fight statistics."""

from __future__ import annotations

import re
import logging
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from .base import BaseScraper
from ..utils import parse_date, parse_fight_time, fight_duration_seconds, normalize_name

logger = logging.getLogger(__name__)

BASE_URL = "http://ufcstats.com/statistics/events/completed"
EVENT_URL = "http://ufcstats.com/event-details/"
FIGHT_URL = "http://ufcstats.com/fight-details/"
FIGHTER_URL = "http://ufcstats.com/fighter-details/"


class UFCStatsScraper(BaseScraper):
    SOURCE_NAME = "ufcstats"

    def scrape_all_events(self) -> list[dict[str, str]]:
        """Get all completed UFC event URLs and names."""
        events = []
        page = 1
        while True:
            url = f"{BASE_URL}?page={page}"
            html = self.fetch(url)
            soup = BeautifulSoup(html, "lxml")
            rows = soup.select("tr.b-statistics__table-row")
            if not rows:
                break

            found_any = False
            for row in rows:
                link = row.select_one("a.b-link")
                if link and link.get("href"):
                    event_url = link["href"].strip()
                    event_name = link.get_text(strip=True)
                    date_td = row.select("td")
                    event_date = None
                    if len(date_td) > 0:
                        date_span = date_td[-1]
                        event_date = date_span.get_text(strip=True)
                    events.append({
                        "event_url": event_url,
                        "event_name": event_name,
                        "event_date": event_date,
                    })
                    found_any = True

            if not found_any:
                break
            page += 1

        logger.info(f"Found {len(events)} events")
        return events

    def scrape_event(self, event_url: str, event_name: str = "",
                     event_date: str = "") -> list[dict[str, Any]]:
        """Scrape all fights from a single event."""
        html = self.fetch(event_url)
        soup = BeautifulSoup(html, "lxml")
        fights = []

        rows = soup.select("tr.b-fight-details__table-row")
        for i, row in enumerate(rows):
            cols = row.select("td")
            if len(cols) < 10:
                continue

            links = row.select("a")
            fight_link = None
            for a in links:
                href = a.get("href", "")
                if "fight-details" in href:
                    fight_link = href.strip()
                    break

            fighter_links = row.select("td:first-child a.b-link")
            if len(fighter_links) < 2:
                continue

            fighter_a = fighter_links[0].get_text(strip=True)
            fighter_b = fighter_links[1].get_text(strip=True)

            fighter_a_url = fighter_links[0].get("href", "").strip()
            fighter_b_url = fighter_links[1].get("href", "").strip()

            method_col = cols[7] if len(cols) > 7 else None
            method = method_col.get_text(strip=True) if method_col else ""

            round_col = cols[8] if len(cols) > 8 else None
            round_num = round_col.get_text(strip=True) if round_col else ""

            time_col = cols[9] if len(cols) > 9 else None
            time_str = time_col.get_text(strip=True) if time_col else ""

            win_marker = cols[0].select_one("i.b-flag__inner")
            winner = None
            if win_marker:
                win_text = win_marker.get_text(strip=True).lower()
                if "win" in win_text:
                    winner = fighter_a

            fight = {
                "event_name": event_name,
                "event_date": event_date,
                "fight_url": fight_link,
                "fighter_a": fighter_a,
                "fighter_b": fighter_b,
                "fighter_a_url": fighter_a_url,
                "fighter_b_url": fighter_b_url,
                "winner": winner if winner else fighter_b,
                "method": method,
                "round": round_num,
                "time": time_str,
                "card_position": i,
            }
            fights.append(fight)

        return fights

    def scrape_fight_details(self, fight_url: str) -> dict[str, Any]:
        """Scrape detailed round-by-round stats for a fight."""
        html = self.fetch(fight_url)
        soup = BeautifulSoup(html, "lxml")

        details: dict[str, Any] = {"fight_url": fight_url}

        method_section = soup.select_one("i.b-fight-details__text-item_first")
        if method_section:
            method_parts = method_section.get_text(strip=True).split(":")
            if len(method_parts) > 1:
                details["method_detail"] = method_parts[1].strip()

        info_items = soup.select("i.b-fight-details__text-item")
        for item in info_items:
            text = item.get_text(strip=True)
            if "Round:" in text:
                details["round"] = text.replace("Round:", "").strip()
            elif "Time:" in text:
                details["time"] = text.replace("Time:", "").strip()
            elif "Time format:" in text:
                details["time_format"] = text.replace("Time format:", "").strip()
            elif "Referee:" in text:
                details["referee"] = text.replace("Referee:", "").strip()

        totals_section = soup.select("section.b-fight-details__section")
        if totals_section:
            details["round_stats"] = self._parse_round_stats(totals_section)

        return details

    def _parse_round_stats(self, sections: list) -> list[dict]:
        """Parse round-by-round statistics tables."""
        rounds = []
        for section in sections:
            tables = section.select("table")
            for table in tables:
                rows = table.select("tr")
                for row in rows:
                    cols = row.select("td")
                    if len(cols) < 9:
                        continue
                    texts = [c.get_text(strip=True) for c in cols]
                    if any(t and not t.replace(" ", "").replace(".", "").replace("-", "").isalpha()
                           for t in texts[2:]):
                        rounds.append(texts)
        return rounds

    def scrape_fighter_details(self, fighter_url: str) -> dict[str, Any]:
        """Scrape fighter physical stats and career stats."""
        html = self.fetch(fighter_url)
        soup = BeautifulSoup(html, "lxml")

        info: dict[str, Any] = {"fighter_url": fighter_url}

        name_el = soup.select_one("span.b-content__title-highlight")
        if name_el:
            info["name"] = name_el.get_text(strip=True)

        record_el = soup.select_one("span.b-content__title-record")
        if record_el:
            record_text = record_el.get_text(strip=True)
            info["record"] = record_text.replace("Record:", "").strip()

        bio_items = soup.select("li.b-list__box-list-item")
        for item in bio_items:
            text = item.get_text(strip=True)
            if "Height:" in text:
                info["height"] = text.replace("Height:", "").strip()
            elif "Weight:" in text:
                info["weight"] = text.replace("Weight:", "").strip()
            elif "Reach:" in text:
                info["reach"] = text.replace("Reach:", "").strip()
            elif "STANCE:" in text:
                info["stance"] = text.replace("STANCE:", "").strip()
            elif "DOB:" in text:
                info["dob"] = text.replace("DOB:", "").strip()

        career_stats = soup.select("li.b-list__box-list-item_type_block")
        for stat in career_stats:
            label_el = stat.select_one("i.b-list__box-item-title")
            if not label_el:
                continue
            label = label_el.get_text(strip=True)
            value = stat.get_text(strip=True).replace(label, "").strip()
            key = re.sub(r'[^a-z0-9]', '_', label.lower()).strip('_')
            info[f"career_{key}"] = value

        return info

    def scrape_all(self) -> pd.DataFrame:
        """Scrape all events and fights into a DataFrame."""
        events = self.scrape_all_events()
        all_fights = []

        for event in tqdm(events, desc="Scraping UFC events", unit="event"):
            try:
                fights = self.scrape_event(
                    event["event_url"],
                    event["event_name"],
                    event.get("event_date", ""),
                )
                all_fights.extend(fights)
            except Exception as e:
                logger.warning(f"Failed to scrape event {event['event_name']}: {e}")

        df = pd.DataFrame(all_fights)
        if not df.empty:
            df["fighter_a_norm"] = df["fighter_a"].apply(normalize_name)
            df["fighter_b_norm"] = df["fighter_b"].apply(normalize_name)
            if "event_date" in df.columns:
                df["date"] = df["event_date"].apply(parse_date)
            df["source"] = "ufcstats"

        logger.info(f"Scraped {len(df)} fights from ufcstats")
        return df
