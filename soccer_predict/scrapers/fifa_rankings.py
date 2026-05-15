"""Scraper for FIFA World Rankings — historical ranking data.

FIFA rankings are a strong baseline predictor for international football.
Higher-ranked teams win more often, and ranking deltas correlate with outcomes.
"""

from __future__ import annotations

import csv
import io
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from sports_predict.scrapers.base import BaseScraper

logger = logging.getLogger(__name__)

# FIFA rankings data source — Kaggle dataset / FIFA API
FIFA_RANKINGS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/rankings.csv"


class FIFARankingsScraper(BaseScraper):
    """Scraper for historical FIFA world rankings."""

    SOURCE_NAME = "fifa_rankings"

    def scrape_all(self) -> pd.DataFrame:
        """Download and parse historical FIFA rankings."""
        try:
            text = self.fetch(FIFA_RANKINGS_URL)
            df = pd.read_csv(io.StringIO(text))
        except Exception as e:
            logger.warning(f"Failed to fetch FIFA rankings from URL: {e}")
            # Try local fallback
            local_path = Path("data/raw/fifa_rankings.csv")
            if local_path.exists():
                df = pd.read_csv(local_path)
                logger.info("Loaded FIFA rankings from local file")
            else:
                logger.error("No FIFA rankings data available")
                return pd.DataFrame()

        # Standardize columns
        col_map = {
            "rank": "fifa_rank",
            "country_full": "team",
            "country_abrv": "team_code",
            "total_points": "fifa_points",
            "rank_date": "ranking_date",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        if "ranking_date" in df.columns:
            df["ranking_date"] = pd.to_datetime(df["ranking_date"], errors="coerce")
            df = df.sort_values("ranking_date").reset_index(drop=True)

        df["source"] = "fifa_rankings"
        logger.info(f"Loaded {len(df)} FIFA ranking entries")
        return df

    def get_ranking_at_date(self, df: pd.DataFrame, team: str,
                            date: str) -> dict[str, Any]:
        """Get a team's FIFA ranking closest to (but before) a given date."""
        target = pd.Timestamp(date)
        team_ranks = df[
            (df["team"].str.lower() == team.lower()) &
            (df["ranking_date"] <= target)
        ]

        if team_ranks.empty:
            return {"fifa_rank": 50, "fifa_points": 1300}

        latest = team_ranks.iloc[-1]
        return {
            "fifa_rank": int(latest.get("fifa_rank", 50)),
            "fifa_points": float(latest.get("fifa_points", 1300)),
        }
