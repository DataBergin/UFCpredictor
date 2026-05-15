"""Base scraper with shared functionality for all sports."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import random
from pathlib import Path
from typing import Any

import requests
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}


class BaseScraper:
    """Base class for all sport scrapers with caching and rate limiting."""

    SOURCE_NAME: str = "base"

    def __init__(self, cache_dir: str | Path = "data/cache",
                 delay_range: tuple[float, float] = (1.5, 3.5)):
        self.cache_dir = Path(cache_dir) / self.SOURCE_NAME
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay_range = delay_range
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)

    def _cache_key(self, url: str) -> Path:
        """Generate a deterministic cache file path for a URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.cache_dir / f"{url_hash}.html"

    def fetch(self, url: str, use_cache: bool = True) -> str:
        """Fetch a URL with caching and rate limiting."""
        if use_cache:
            cache_path = self._cache_key(url)
            if cache_path.exists():
                logger.debug(f"Cache hit: {url}")
                return cache_path.read_text(encoding="utf-8", errors="replace")

        # Rate limit
        delay = random.uniform(*self.delay_range)
        time.sleep(delay)

        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        html = resp.text

        if use_cache:
            cache_path = self._cache_key(url)
            cache_path.write_text(html, encoding="utf-8")

        return html

    def fetch_json(self, url: str, use_cache: bool = True) -> Any:
        """Fetch a URL and return parsed JSON."""
        cache_path = self.cache_dir / f"{hashlib.md5(url.encode()).hexdigest()}.json"

        if use_cache and cache_path.exists():
            logger.debug(f"Cache hit (json): {url}")
            return json.loads(cache_path.read_text(encoding="utf-8"))

        delay = random.uniform(*self.delay_range)
        time.sleep(delay)

        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if use_cache:
            cache_path.write_text(json.dumps(data), encoding="utf-8")

        return data

    def scrape_all(self) -> pd.DataFrame:
        """Scrape all data into a DataFrame. Override in subclass."""
        raise NotImplementedError
