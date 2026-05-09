"""Base scraper with shared functionality."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ..utils import get_session, fetch_page, load_cache, save_cache, ensure_dir

logger = logging.getLogger(__name__)


class BaseScraper:
    """Base class for all scrapers with caching and rate limiting."""

    SOURCE_NAME: str = "base"

    def __init__(self, cache_dir: str | Path = "data/cache",
                 delay_range: tuple[float, float] = (1.5, 3.5)):
        self.cache_dir = ensure_dir(Path(cache_dir) / self.SOURCE_NAME)
        self.delay_range = delay_range
        self.session = get_session()

    def fetch(self, url: str, use_cache: bool = True) -> str:
        if use_cache:
            cached = load_cache(self.cache_dir, url)
            if cached is not None:
                logger.debug(f"Cache hit: {url}")
                return cached

        html = fetch_page(url, self.session, self.delay_range)

        if use_cache:
            save_cache(self.cache_dir, url, html)

        return html

    def scrape_all(self) -> pd.DataFrame:
        raise NotImplementedError

    def scrape_fighter(self, fighter_url: str) -> dict[str, Any]:
        raise NotImplementedError
