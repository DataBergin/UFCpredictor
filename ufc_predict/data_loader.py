"""Data loading, merging, and preprocessing from multiple sources."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import normalize_name, fight_duration_seconds, parse_date

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and merge fight data from multiple sources into a unified format."""

    REQUIRED_COLS = [
        "fighter_a", "fighter_b", "winner", "method", "round",
        "time", "date", "weight_class",
    ]

    def __init__(self, data_dir: str | Path = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

    def load_all(self) -> pd.DataFrame:
        """Load and merge all available data sources."""
        dfs = []

        # UFC Stats data
        ufcstats_path = self.raw_dir / "ufcstats_fights.csv"
        if ufcstats_path.exists():
            df = self._load_ufcstats(ufcstats_path)
            dfs.append(df)
            logger.info(f"Loaded {len(df)} fights from UFC Stats")

        # Kaggle UFC data (common format)
        for kaggle_file in self.raw_dir.glob("*ufc*.csv"):
            if "ufcstats" not in kaggle_file.name:
                df = self._load_kaggle(kaggle_file)
                if df is not None and not df.empty:
                    dfs.append(df)
                    logger.info(f"Loaded {len(df)} fights from {kaggle_file.name}")

        if not dfs:
            logger.warning("No fight data found. Run `ufc-predict scrape` first.")
            return pd.DataFrame(columns=self.REQUIRED_COLS)

        # Merge and deduplicate
        merged = pd.concat(dfs, ignore_index=True)
        merged = self._deduplicate(merged)
        merged = self._enrich(merged)

        # Merge odds if available
        odds_path = self.raw_dir / "odds_data.csv"
        if odds_path.exists():
            merged = self._merge_odds(merged, odds_path)

        logger.info(f"Final dataset: {len(merged)} unique fights")
        return merged

    def _load_ufcstats(self, path: Path) -> pd.DataFrame:
        """Load UFC Stats scraped data."""
        df = pd.read_csv(path)
        col_map = {
            "fighter_a": "fighter_a",
            "fighter_b": "fighter_b",
            "winner": "winner",
            "method": "method",
            "round": "round",
            "time": "time",
            "event_date": "date",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        df["source"] = "ufcstats"
        return df

    def _load_kaggle(self, path: Path) -> pd.DataFrame | None:
        """Load common Kaggle UFC dataset formats."""
        try:
            df = pd.read_csv(path)
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            return None

        # Handle multiple Kaggle format variations
        col_variants = {
            "fighter_a": ["R_fighter", "red_fighter", "fighter_1", "fighter_a", "Red"],
            "fighter_b": ["B_fighter", "blue_fighter", "fighter_2", "fighter_b", "Blue"],
            "winner": ["Winner", "winner", "result", "W/L"],
            "method": ["win_by", "method", "Method", "finish"],
            "round": ["last_round", "round", "Round", "finish_round"],
            "time": ["last_round_time", "time", "Time", "finish_time"],
            "date": ["date", "Date", "event_date"],
            "weight_class": ["weight_class", "WeightClass", "division"],
        }

        renamed = pd.DataFrame()
        for target, variants in col_variants.items():
            for v in variants:
                if v in df.columns:
                    renamed[target] = df[v]
                    break

        if "fighter_a" not in renamed.columns or "fighter_b" not in renamed.columns:
            return None

        # Normalize winner column
        if "winner" in renamed.columns:
            renamed["winner"] = renamed.apply(self._normalize_winner, axis=1)

        renamed["source"] = "kaggle"
        return renamed

    def _normalize_winner(self, row) -> str:
        """Normalize winner field across different data formats."""
        winner = str(row.get("winner", "")).strip()
        fighter_a = str(row.get("fighter_a", ""))
        fighter_b = str(row.get("fighter_b", ""))

        if winner.lower() in ("red", "r", "1", fighter_a.lower()):
            return fighter_a
        elif winner.lower() in ("blue", "b", "2", fighter_b.lower()):
            return fighter_b
        elif normalize_name(winner) == normalize_name(fighter_a):
            return fighter_a
        elif normalize_name(winner) == normalize_name(fighter_b):
            return fighter_b
        return winner

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate fights across sources."""
        df["fighter_a_norm"] = df["fighter_a"].apply(normalize_name)
        df["fighter_b_norm"] = df["fighter_b"].apply(normalize_name)

        if "date" in df.columns:
            df["date_parsed"] = df["date"].apply(
                lambda x: parse_date(str(x)) if pd.notna(x) else None
            )

        # Dedup by normalized fighters + date
        dedup_key = df.apply(
            lambda r: f"{min(r['fighter_a_norm'], r['fighter_b_norm'])}_"
                      f"{max(r['fighter_a_norm'], r['fighter_b_norm'])}_"
                      f"{r.get('date_parsed', '')}",
            axis=1
        )
        df["dedup_key"] = dedup_key

        # Keep the row with the most data (fewest NaN)
        df["data_completeness"] = df.notna().sum(axis=1)
        df = df.sort_values("data_completeness", ascending=False).drop_duplicates(
            subset="dedup_key", keep="first"
        )
        df = df.drop(columns=["dedup_key", "data_completeness"])

        return df.reset_index(drop=True)

    def _enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich with computed columns."""
        # Duration in seconds
        if "round" in df.columns and "time" in df.columns:
            df["duration_seconds"] = df.apply(
                lambda r: fight_duration_seconds(
                    int(r["round"]) if pd.notna(r["round"]) and str(r["round"]).isdigit() else None,
                    str(r["time"]) if pd.notna(r["time"]) else ""
                ),
                axis=1
            )

        # Ensure date column is proper
        if "date_parsed" in df.columns:
            df["date"] = df["date_parsed"]
            df = df.drop(columns=["date_parsed"], errors="ignore")

        # Method classification
        df["method_class"] = df["method"].apply(self._classify_method)

        # Sort chronologically
        df = df.sort_values("date").reset_index(drop=True)

        return df

    def _merge_odds(self, fights_df: pd.DataFrame, odds_path: Path) -> pd.DataFrame:
        """Merge odds data with fight data."""
        odds_df = pd.read_csv(odds_path)
        if odds_df.empty:
            return fights_df

        odds_df["fighter_a_norm"] = odds_df["fighter_a_norm"] if "fighter_a_norm" in odds_df.columns \
            else odds_df.get("fighter_a", pd.Series()).apply(normalize_name)
        odds_df["fighter_b_norm"] = odds_df["fighter_b_norm"] if "fighter_b_norm" in odds_df.columns \
            else odds_df.get("fighter_b", pd.Series()).apply(normalize_name)

        odds_cols = [c for c in odds_df.columns if any(
            k in c for k in ["prob", "odds", "line_move", "range"]
        )]

        merged = fights_df.merge(
            odds_df[["fighter_a_norm", "fighter_b_norm"] + odds_cols],
            on=["fighter_a_norm", "fighter_b_norm"],
            how="left",
            suffixes=("", "_odds"),
        )

        logger.info(f"Merged odds for {merged['closing_prob_a'].notna().sum()} fights")
        return merged

    def _classify_method(self, method: str) -> int:
        """0=KO/TKO, 1=Sub, 2=Decision."""
        if pd.isna(method):
            return 2
        m = str(method).lower()
        if "ko" in m or "tko" in m:
            return 0
        elif "sub" in m:
            return 1
        return 2

    def save_processed(self, df: pd.DataFrame, filename: str = "fights_merged.parquet") -> Path:
        """Save processed fight data."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        path = self.processed_dir / filename
        df.to_parquet(path, index=False)
        logger.info(f"Saved processed data to {path}")
        return path
