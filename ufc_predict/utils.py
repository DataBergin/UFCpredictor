"""Shared utilities for scraping, caching, and data manipulation."""

from __future__ import annotations

import hashlib
import json
import time
import random
import logging
from pathlib import Path
from datetime import datetime, date
from typing import Any

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from fake_useragent import UserAgent

logger = logging.getLogger(__name__)

_ua = None


def get_user_agent() -> str:
    global _ua
    if _ua is None:
        try:
            _ua = UserAgent()
        except Exception:
            return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    try:
        return _ua.random
    except Exception:
        return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": get_user_agent(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    })
    return session


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_page(url: str, session: requests.Session | None = None,
               delay_range: tuple[float, float] = (1.5, 3.5)) -> str:
    if session is None:
        session = get_session()
    time.sleep(random.uniform(*delay_range))
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def cache_path(cache_dir: str | Path, key: str) -> Path:
    h = hashlib.md5(key.encode()).hexdigest()
    return Path(cache_dir) / f"{h}.json"


def load_cache(cache_dir: str | Path, key: str) -> Any | None:
    p = cache_path(cache_dir, key)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def save_cache(cache_dir: str | Path, key: str, data: Any) -> None:
    p = cache_path(cache_dir, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f, default=str)


def parse_fight_time(time_str: str) -> int | None:
    """Parse time string like '4:35' to total seconds."""
    if not time_str or time_str == "--":
        return None
    parts = time_str.strip().split(":")
    if len(parts) == 2:
        try:
            return int(parts[0]) * 60 + int(parts[1])
        except ValueError:
            return None
    return None


def fight_duration_seconds(round_num: int, time_in_round: str,
                           round_length: int = 300) -> int | None:
    """Total fight duration from round number and time in round."""
    t = parse_fight_time(time_in_round)
    if t is None or round_num is None:
        return None
    return (round_num - 1) * round_length + t


def normalize_name(name: str) -> str:
    """Normalize fighter name for matching across sources."""
    name = name.strip().lower()
    replacements = {
        "junior": "jr",
        "senior": "sr",
        "'": "",
        "'": "",
        "-": " ",
        ".": "",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    parts = name.split()
    return " ".join(parts)


def american_to_implied_prob(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def devig_odds(prob_a: float, prob_b: float) -> tuple[float, float]:
    """Remove vigorish from two-way implied probabilities (multiplicative method)."""
    total = prob_a + prob_b
    if total == 0:
        return 0.5, 0.5
    return prob_a / total, prob_b / total


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_date(date_str: str) -> date | None:
    for fmt in ("%B %d, %Y", "%Y-%m-%d", "%b %d, %Y", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue
    return None
