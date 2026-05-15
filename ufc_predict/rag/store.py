"""ChromaDB vector store for fight-related documents with temporal filtering."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class FightDocumentStore:
    """Manages a ChromaDB vector store of fight-related articles and news.

    Documents are stored with metadata (fighter names, date, source) and
    retrieved with strict date filtering to prevent information leakage.
    """

    def __init__(self, persist_dir: str = "data/vectorstore",
                 embedding_model: str = "nomic-embed-text",
                 collection_name: str = "fight_docs"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        self.collection_name = collection_name

        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        # Use Ollama embeddings if available, fall back to sentence-transformers
        self._embedding_fn = None
        try:
            from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
            # Quick connectivity test
            import urllib.request
            urllib.request.urlopen("http://localhost:11434", timeout=2)
            self._embedding_fn = OllamaEmbeddingFunction(
                model_name=embedding_model,
                url="http://localhost:11434",
            )
            logger.info("Using Ollama embeddings")
        except Exception:
            try:
                from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
                self._embedding_fn = SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
                logger.info("Ollama not available, using sentence-transformers embeddings")
            except Exception:
                logger.warning("No embedding backend available, using ChromaDB defaults")
                self._embedding_fn = None

        try:
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                embedding_function=self._embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
        except ValueError:
            # Embedding function conflict — delete and recreate
            logger.warning("Embedding function conflict, recreating collection")
            self._client.delete_collection(collection_name)
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                embedding_function=self._embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )

        logger.info(f"Document store initialized: {self._collection.count()} documents "
                     f"in {self.persist_dir}")

    def add_documents(self, documents: list[dict[str, Any]]) -> int:
        """Add documents to the vector store.

        Each document dict should have:
            - text: str — the document content
            - fighters: list[str] — fighter names mentioned
            - date: str — publication date (YYYY-MM-DD)
            - source: str — source name (e.g., "mmajunkie", "espn")
            - doc_type: str — "article", "interview", "injury_report", etc.
        """
        if not documents:
            return 0

        ids = []
        texts = []
        metadatas = []

        for doc in documents:
            text = doc["text"]
            doc_id = hashlib.md5(text.encode()).hexdigest()

            # Store each fighter as a separate metadata entry for filtering
            fighters = doc.get("fighters", [])
            # Store date as int (YYYYMMDD) for ChromaDB numeric filtering
            date_str = doc.get("date", "")
            date_int = self._date_to_int(date_str)

            metadata = {
                "date_int": date_int,
                "date_str": date_str,
                "source": doc.get("source", "unknown"),
                "doc_type": doc.get("doc_type", "article"),
                "fighters": ",".join(fighters),  # comma-separated for filtering
                "fighter_count": len(fighters),
            }

            ids.append(doc_id)
            texts.append(text)
            metadatas.append(metadata)

        # Upsert to handle duplicates
        self._collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )

        logger.info(f"Added/updated {len(ids)} documents to store")
        return len(ids)

    def retrieve_for_fight(self, fighter_a: str, fighter_b: str,
                           fight_date: str, k: int = 10) -> list[dict[str, Any]]:
        """Retrieve relevant documents for a fight prediction.

        CRITICAL: Only returns documents dated BEFORE fight_date to prevent leakage.

        Args:
            fighter_a: First fighter name
            fighter_b: Second fighter name
            fight_date: Fight date (YYYY-MM-DD) — only docs before this date returned
            k: Maximum number of documents to return

        Returns:
            List of dicts with 'text', 'metadata', and 'distance' keys
        """
        if self._collection.count() == 0:
            return []

        query_text = f"{fighter_a} vs {fighter_b} fight prediction analysis"

        # ChromaDB where filter: date must be strictly before fight date
        fight_date_int = self._date_to_int(fight_date)
        where_filter = {"date_int": {"$lt": fight_date_int}}

        try:
            results = self._collection.query(
                query_texts=[query_text],
                n_results=min(k * 3, self._collection.count()),  # over-fetch then filter
                where=where_filter,
            )
        except Exception as e:
            logger.warning(f"ChromaDB query failed: {e}")
            return []

        if not results or not results["documents"] or not results["documents"][0]:
            return []

        # Post-filter: keep only docs mentioning at least one fighter
        fighter_a_lower = fighter_a.lower()
        fighter_b_lower = fighter_b.lower()
        filtered = []

        for doc_text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            fighters_str = metadata.get("fighters", "").lower()
            doc_text_lower = doc_text.lower()

            # Check if document is relevant to either fighter
            a_match = fighter_a_lower in fighters_str or fighter_a_lower in doc_text_lower
            b_match = fighter_b_lower in fighters_str or fighter_b_lower in doc_text_lower

            if a_match or b_match:
                filtered.append({
                    "text": doc_text,
                    "metadata": metadata,
                    "distance": distance,
                    "relevant_to_a": a_match,
                    "relevant_to_b": b_match,
                })

            if len(filtered) >= k:
                break

        return filtered

    def retrieve_for_entity(self, entity_name: str, as_of_date: str,
                            k: int = 5) -> list[dict[str, Any]]:
        """Retrieve documents for a single entity (fighter/team) as of a date.

        Used for general entity context, not fight-specific.
        """
        if self._collection.count() == 0:
            return []

        try:
            results = self._collection.query(
                query_texts=[f"{entity_name} recent news analysis"],
                n_results=min(k * 2, self._collection.count()),
                where={"date_int": {"$lt": self._date_to_int(as_of_date)}},
            )
        except Exception as e:
            logger.warning(f"Entity retrieval failed: {e}")
            return []

        if not results or not results["documents"] or not results["documents"][0]:
            return []

        entity_lower = entity_name.lower()
        filtered = []
        for doc_text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            if entity_lower in doc_text.lower() or entity_lower in metadata.get("fighters", "").lower():
                filtered.append({
                    "text": doc_text,
                    "metadata": metadata,
                    "distance": distance,
                })
            if len(filtered) >= k:
                break

        return filtered

    def count(self) -> int:
        """Return total number of documents in the store."""
        return self._collection.count()

    def clear(self) -> None:
        """Delete all documents from the store."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Document store cleared")

    @staticmethod
    def _date_to_int(date_str: str) -> int:
        """Convert YYYY-MM-DD date string to integer YYYYMMDD for ChromaDB filtering."""
        try:
            return int(date_str.replace("-", ""))
        except (ValueError, AttributeError):
            return 0
