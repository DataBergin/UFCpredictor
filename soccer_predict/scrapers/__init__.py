"""Soccer data scrapers."""

from .fbref import FBrefScraper
from .fifa_rankings import FIFARankingsScraper
from .club_leagues import ClubLeagueScraper

__all__ = ["FBrefScraper", "FIFARankingsScraper", "ClubLeagueScraper"]
