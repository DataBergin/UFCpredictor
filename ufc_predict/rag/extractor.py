"""LLM-based feature extraction from retrieved documents using Ollama."""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Extraction schemas per sport — defines what features to pull from text
EXTRACTION_SCHEMAS: dict[str, dict[str, dict[str, Any]]] = {
    "ufc": {
        "confidence": {"type": "float", "range": [-1, 1],
                       "desc": "fighter's expressed confidence level"},
        "injury": {"type": "int", "range": [0, 1],
                   "desc": "whether an injury is mentioned"},
        "injury_severity": {"type": "float", "range": [0, 1],
                            "desc": "estimated injury impact on performance"},
        "camp_quality": {"type": "float", "range": [-1, 1],
                         "desc": "quality of training camp reports"},
        "media_sentiment": {"type": "float", "range": [-1, 1],
                            "desc": "overall media tone about this fighter"},
    },
    "soccer": {
        "squad_fitness": {"type": "float", "range": [0, 1],
                          "desc": "overall squad health and availability"},
        "manager_confidence": {"type": "float", "range": [-1, 1],
                               "desc": "manager's expressed confidence"},
        "key_player_available": {"type": "int", "range": [0, 1],
                                 "desc": "whether key/star player is available"},
        "tactical_edge": {"type": "float", "range": [-1, 1],
                          "desc": "tactical advantage from narrative"},
    },
    "mlb": {
        "pitcher_health": {"type": "float", "range": [0, 1],
                           "desc": "starting pitcher's health signals"},
        "clubhouse_morale": {"type": "float", "range": [-1, 1],
                             "desc": "team chemistry and morale"},
        "trade_impact": {"type": "float", "range": [-1, 1],
                         "desc": "impact of recent trades or roster moves"},
        "lineup_confidence": {"type": "float", "range": [-1, 1],
                              "desc": "stability and confidence in lineup"},
    },
}


class RAGFeatureExtractor:
    """Extracts structured numerical features from retrieved text using a local LLM.

    Uses Ollama (local, no cloud API) to analyze retrieved documents and produce
    features that can be consumed by the existing ensemble models.
    """

    def __init__(self, ollama_model: str = "mistral",
                 sport: str = "ufc",
                 cache_dir: str = "data/cache/rag_features"):
        self.ollama_model = ollama_model
        self.sport = sport
        self.schema = EXTRACTION_SCHEMAS.get(sport, EXTRACTION_SCHEMAS["ufc"])
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._llm = None

    def _get_llm(self):
        """Lazy-load the Ollama LLM."""
        if self._llm is None:
            try:
                from langchain_ollama import ChatOllama
                self._llm = ChatOllama(
                    model=self.ollama_model,
                    temperature=0,
                    num_predict=512,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Ollama LLM: {e}")
                return None
        return self._llm

    def extract_features(self, entity_a: str, entity_b: str,
                         documents: list[dict[str, Any]],
                         fight_date: str = "") -> dict[str, float]:
        """Extract numerical features from retrieved documents.

        Args:
            entity_a: First entity (fighter/team) name
            entity_b: Second entity name
            documents: Retrieved documents from FightDocumentStore
            fight_date: For cache key

        Returns:
            Dict of rag_* prefixed features
        """
        # Check cache
        cache_key = self._cache_key(entity_a, entity_b, fight_date)
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        llm = self._get_llm()
        if llm is None:
            return self._empty_features()

        # Split docs by relevance to each entity
        docs_a = [d for d in documents if d.get("relevant_to_a", True)]
        docs_b = [d for d in documents if d.get("relevant_to_b", True)]

        # Extract features for each entity
        feats_a = self._extract_for_entity(entity_a, docs_a, llm)
        feats_b = self._extract_for_entity(entity_b, docs_b, llm)

        # Build final feature dict with rag_ prefix
        features = {}
        for key in self.schema:
            val_a = feats_a.get(key, 0.0)
            val_b = feats_b.get(key, 0.0)
            features[f"rag_{key}_a"] = val_a
            features[f"rag_{key}_b"] = val_b
            features[f"rag_{key}_diff"] = val_a - val_b

        # Meta features
        features["rag_doc_count_a"] = len(docs_a)
        features["rag_doc_count_b"] = len(docs_b)
        features["rag_doc_count_diff"] = len(docs_a) - len(docs_b)

        # Extract matchup-level features (expert consensus)
        matchup_docs = [d for d in documents
                        if d.get("relevant_to_a") and d.get("relevant_to_b")]
        if matchup_docs:
            consensus = self._extract_consensus(entity_a, entity_b, matchup_docs, llm)
            features["rag_expert_consensus_a"] = consensus
            features["rag_expert_consensus_diff"] = consensus - 0.5
        else:
            features["rag_expert_consensus_a"] = 0.5
            features["rag_expert_consensus_diff"] = 0.0

        # Cache results
        self._save_cache(cache_key, features)
        return features

    def _extract_for_entity(self, entity: str,
                            docs: list[dict[str, Any]],
                            llm) -> dict[str, float]:
        """Extract features for a single entity from relevant documents."""
        if not docs:
            return {k: 0.0 for k in self.schema}

        # Combine document texts (limit context length)
        context = "\n---\n".join(d["text"][:500] for d in docs[:5])

        # Build prompt from schema
        fields_desc = "\n".join(
            f"- {key}: {info['desc']} ({info['range'][0]} to {info['range'][1]})"
            for key, info in self.schema.items()
        )

        prompt = (
            f"Analyze the following pre-fight/pre-game context about {entity}. "
            f"Extract these signals as a JSON object with numeric values only:\n\n"
            f"{fields_desc}\n\n"
            f"Context:\n{context}\n\n"
            f"Respond with ONLY a valid JSON object, no other text. "
            f"Use 0 as default if information is not available."
        )

        try:
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            parsed = self._parse_json_response(content)

            # Validate and clamp values
            result = {}
            for key, info in self.schema.items():
                val = float(parsed.get(key, 0.0))
                lo, hi = info["range"]
                result[key] = max(lo, min(hi, val))

            return result

        except Exception as e:
            logger.warning(f"LLM extraction failed for {entity}: {e}")
            return {k: 0.0 for k in self.schema}

    def _extract_consensus(self, entity_a: str, entity_b: str,
                           docs: list[dict[str, Any]], llm) -> float:
        """Extract expert consensus probability for entity_a winning."""
        context = "\n---\n".join(d["text"][:500] for d in docs[:3])

        prompt = (
            f"Based on the following expert analysis and predictions about "
            f"{entity_a} vs {entity_b}, what is the implied probability that "
            f"{entity_a} wins? Respond with ONLY a single number between 0.0 and 1.0."
            f"\n\nContext:\n{context}"
        )

        try:
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            # Extract first float from response
            import re
            match = re.search(r'(0?\.\d+|1\.0|0|1)', content)
            if match:
                return max(0.0, min(1.0, float(match.group(1))))
        except Exception as e:
            logger.warning(f"Consensus extraction failed: {e}")

        return 0.5

    def _parse_json_response(self, text: str) -> dict:
        """Parse JSON from LLM response, handling common formatting issues."""
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON block from markdown
        import re
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding first { ... } block
        brace_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group())
            except json.JSONDecodeError:
                pass

        logger.warning(f"Could not parse JSON from LLM response: {text[:200]}")
        return {}

    def _empty_features(self) -> dict[str, float]:
        """Return zero-filled features when extraction is unavailable."""
        features = {}
        for key in self.schema:
            features[f"rag_{key}_a"] = 0.0
            features[f"rag_{key}_b"] = 0.0
            features[f"rag_{key}_diff"] = 0.0
        features["rag_doc_count_a"] = 0
        features["rag_doc_count_b"] = 0
        features["rag_doc_count_diff"] = 0
        features["rag_expert_consensus_a"] = 0.5
        features["rag_expert_consensus_diff"] = 0.0
        return features

    def _cache_key(self, entity_a: str, entity_b: str, fight_date: str) -> str:
        """Generate cache key from fight parameters."""
        raw = f"{entity_a}|{entity_b}|{fight_date}|{self.sport}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _load_cache(self, key: str) -> dict[str, float] | None:
        """Load cached features if available."""
        cache_path = self.cache_dir / f"{key}.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        return None

    def _save_cache(self, key: str, features: dict[str, float]) -> None:
        """Save extracted features to cache."""
        cache_path = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(features, f)
        except Exception as e:
            logger.warning(f"Failed to cache features: {e}")
