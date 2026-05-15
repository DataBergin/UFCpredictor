"""RAG-based feature source for the fight prediction pipeline."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class RAGFeatures:
    """Feature source that extracts signals from unstructured text via RAG.

    Integrates with the existing feature pipeline as a plug-in source.
    When disabled or unavailable, returns zero-filled defaults so the
    model degrades gracefully.
    """

    def __init__(self, config: dict[str, Any]):
        self.enabled = config.get("enabled", False)
        self._store = None
        self._extractor = None

        if self.enabled:
            try:
                from ..rag.store import FightDocumentStore
                from ..rag.extractor import RAGFeatureExtractor

                self._store = FightDocumentStore(
                    persist_dir=config.get("vectorstore_dir", "data/vectorstore"),
                    embedding_model=config.get("embedding_model", "nomic-embed-text"),
                )
                self._extractor = RAGFeatureExtractor(
                    ollama_model=config.get("ollama_model", "mistral"),
                    sport=config.get("sport", "ufc"),
                    cache_dir=config.get("cache_dir", "data/cache/rag_features"),
                )
                logger.info(f"RAG features enabled: {self._store.count()} docs in store")
            except Exception as e:
                logger.warning(f"RAG initialization failed (features will be zero-filled): {e}")
                self.enabled = False

    def get_features(self, fighter_a: str, fighter_b: str,
                     fight_date: str) -> dict[str, float]:
        """Get RAG-derived features for a fight.

        Args:
            fighter_a: First fighter name
            fighter_b: Second fighter name
            fight_date: Fight date (YYYY-MM-DD) for temporal filtering

        Returns:
            Dict of rag_* prefixed numerical features
        """
        if not self.enabled or self._store is None or self._extractor is None:
            return self._extractor._empty_features() if self._extractor else self._default_empty()

        # Retrieve relevant pre-fight documents
        docs = self._store.retrieve_for_fight(
            fighter_a, fighter_b, fight_date,
            k=10,
        )

        if not docs:
            return self._extractor._empty_features()

        # Extract structured features via LLM
        return self._extractor.extract_features(
            fighter_a, fighter_b, docs, fight_date
        )

    def _default_empty(self) -> dict[str, float]:
        """Fallback empty features when extractor isn't initialized."""
        from ..rag.extractor import EXTRACTION_SCHEMAS
        features = {}
        schema = EXTRACTION_SCHEMAS.get("ufc", {})
        for key in schema:
            features[f"rag_{key}_a"] = 0.0
            features[f"rag_{key}_b"] = 0.0
            features[f"rag_{key}_diff"] = 0.0
        features["rag_doc_count_a"] = 0
        features["rag_doc_count_b"] = 0
        features["rag_doc_count_diff"] = 0
        features["rag_expert_consensus_a"] = 0.5
        features["rag_expert_consensus_diff"] = 0.0
        return features
