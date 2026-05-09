"""Data scrapers for UFC fight data from multiple sources."""

from .ufcstats import UFCStatsScraper
from .sherdog import SherdogScraper
from .tapology import TapologyScraper
from .odds import OddsScraper

__all__ = ["UFCStatsScraper", "SherdogScraper", "TapologyScraper", "OddsScraper"]
