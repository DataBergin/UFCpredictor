"""Soccer data scrapers."""

from .fbref import FBrefScraper
from .fifa_rankings import FIFARankingsScraper

__all__ = ["FBrefScraper", "FIFARankingsScraper"]
