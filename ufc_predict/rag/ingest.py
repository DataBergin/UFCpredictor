"""Document ingestion pipeline for fight-related articles and news."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .store import FightDocumentStore

logger = logging.getLogger(__name__)


class ArticleIngester:
    """Ingests text documents into the FightDocumentStore.

    Handles chunking, fighter name detection, and metadata extraction.
    """

    def __init__(self, store: FightDocumentStore,
                 chunk_size: int = 1000, chunk_overlap: int = 200,
                 fighter_roster: list[str] | None = None):
        """
        Args:
            store: FightDocumentStore to ingest into
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between consecutive chunks
            fighter_roster: Known fighter names for matching (optional)
        """
        self.store = store
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.fighter_roster = set(n.lower() for n in (fighter_roster or []))

    def set_roster_from_dataframe(self, fights_df) -> None:
        """Build fighter roster from scraped fight data."""
        names = set()
        for col in ["fighter_a", "fighter_b"]:
            if col in fights_df.columns:
                names.update(fights_df[col].dropna().unique())
        self.fighter_roster = set(n.lower() for n in names)
        logger.info(f"Fighter roster loaded: {len(self.fighter_roster)} fighters")

    def ingest_from_directory(self, dir_path: str | Path,
                              source: str = "local",
                              default_date: str = "") -> int:
        """Ingest all text/HTML files from a directory.

        Expected filename format: YYYY-MM-DD_title.txt (date prefix optional).
        If no date in filename, uses default_date.
        """
        dir_path = Path(dir_path)
        if not dir_path.exists():
            logger.warning(f"Directory not found: {dir_path}")
            return 0

        total = 0
        for file_path in sorted(dir_path.rglob("*")):
            if file_path.suffix.lower() not in (".txt", ".html", ".htm", ".md"):
                continue

            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
                if file_path.suffix.lower() in (".html", ".htm"):
                    text = self._strip_html(text)

                # Extract date from filename if present
                date = self._extract_date_from_filename(file_path.name) or default_date

                # Detect fighter names in text
                fighters = self._detect_fighters(text)

                # Chunk and ingest
                chunks = self.splitter.split_text(text)
                docs = []
                for i, chunk in enumerate(chunks):
                    docs.append({
                        "text": chunk,
                        "fighters": fighters,
                        "date": date,
                        "source": source,
                        "doc_type": self._classify_doc_type(text),
                    })

                count = self.store.add_documents(docs)
                total += count
                logger.debug(f"Ingested {count} chunks from {file_path.name}")

            except Exception as e:
                logger.warning(f"Failed to ingest {file_path}: {e}")

        logger.info(f"Ingested {total} document chunks from {dir_path}")
        return total

    def ingest_text(self, text: str, fighters: list[str],
                    date: str, source: str = "manual",
                    doc_type: str = "article") -> int:
        """Ingest a single text document directly."""
        chunks = self.splitter.split_text(text)
        docs = [{
            "text": chunk,
            "fighters": fighters,
            "date": date,
            "source": source,
            "doc_type": doc_type,
        } for chunk in chunks]

        return self.store.add_documents(docs)

    def ingest_from_rss(self, feed_url: str, source: str = "rss") -> int:
        """Ingest articles from an RSS feed.

        Requires feedparser: pip install feedparser
        """
        try:
            import feedparser
        except ImportError:
            logger.error("feedparser not installed. Run: pip install feedparser")
            return 0

        feed = feedparser.parse(feed_url)
        total = 0

        for entry in feed.entries:
            try:
                title = entry.get("title", "")
                summary = entry.get("summary", "")
                content = entry.get("content", [{}])[0].get("value", "") if entry.get("content") else ""
                text = f"{title}\n\n{content or summary}"

                if len(text.strip()) < 50:
                    continue

                # Extract date
                published = entry.get("published", "")
                date = self._parse_rss_date(published)

                # Strip HTML from content
                text = self._strip_html(text)

                fighters = self._detect_fighters(text)

                count = self.ingest_text(
                    text=text,
                    fighters=fighters,
                    date=date,
                    source=source,
                    doc_type="article",
                )
                total += count

            except Exception as e:
                logger.warning(f"Failed to ingest RSS entry: {e}")

        logger.info(f"Ingested {total} chunks from RSS feed {feed_url}")
        return total

    def _detect_fighters(self, text: str) -> list[str]:
        """Detect fighter names in text using the roster."""
        if not self.fighter_roster:
            return []

        text_lower = text.lower()
        found = []
        for name in self.fighter_roster:
            if name in text_lower:
                found.append(name)

        # Also check last names only (common in articles)
        if not found:
            for name in self.fighter_roster:
                parts = name.split()
                if len(parts) >= 2:
                    last_name = parts[-1]
                    if len(last_name) > 3 and last_name in text_lower:
                        found.append(name)

        return list(set(found))

    def _extract_date_from_filename(self, filename: str) -> str | None:
        """Extract YYYY-MM-DD date from filename."""
        match = re.match(r'(\d{4}-\d{2}-\d{2})', filename)
        return match.group(1) if match else None

    def _parse_rss_date(self, date_str: str) -> str:
        """Parse RSS date to YYYY-MM-DD."""
        from email.utils import parsedate_to_datetime
        try:
            dt = parsedate_to_datetime(date_str)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return ""

    def _strip_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        try:
            from bs4 import BeautifulSoup
            return BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)
        except ImportError:
            return re.sub(r'<[^>]+>', '', text)

    def _classify_doc_type(self, text: str) -> str:
        """Classify document type from content."""
        text_lower = text.lower()
        if any(w in text_lower for w in ["injury", "injured", "pulled out", "withdrew"]):
            return "injury_report"
        if any(w in text_lower for w in ["interview", "said", "told", "speaking to"]):
            return "interview"
        if any(w in text_lower for w in ["prediction", "picks", "odds", "betting"]):
            return "prediction"
        if any(w in text_lower for w in ["camp", "training", "sparring"]):
            return "camp_report"
        return "article"
