"""
recommender.py — Core Recommendation Pipeline
===============================================
Orchestrates the full pipeline:

  User query (text or image bytes)
          ↓
  TextEncoder / ImageEncoder
          ↓
  VectorStore (FAISS nearest-neighbour search)
          ↓
  Top-K candidates
          ↓
  LLMAdvisor (optional re-ranking + styling advice)
          ↓
  Final top-N recommendations

Usage example::

    rec = Recommender()
    results, advice = rec.recommend("blue printed kurta for summer")
    for item in results:
        print(item["name"], item["score"])
    print(advice)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from src.encoder import TextEncoder, ImageEncoder
from src.vector_store import VectorStore
from src.llm_advisor import LLMAdvisor

logger = logging.getLogger(__name__)


class Recommender:
    """End-to-end clothing recommender."""

    def __init__(
        self,
        top_k: int = config.TOP_K_RETRIEVAL,
        top_n: int = config.TOP_N_FINAL,
    ) -> None:
        self._top_k = top_k
        self._top_n = top_n

        logger.info("Initialising Recommender …")
        self._encoder = TextEncoder()
        self._image_encoder = ImageEncoder(self._encoder)
        self._store = VectorStore.load()
        self._advisor = LLMAdvisor()
        logger.info("Recommender ready. Index size: %d products.", len(self._store))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(
        self,
        query: str,
        image_bytes: Optional[bytes] = None,
    ) -> tuple[list[dict[str, Any]], str]:
        """Return top-N product recommendations for *query*.

        Parameters
        ----------
        query:
            Free-text description of the desired clothing item.
        image_bytes:
            Optional raw bytes of an uploaded image.  When provided, the
            image embedding is fused with the text embedding (average).

        Returns
        -------
        (items, styling_advice)
            *items* — list of product dicts sorted by relevance.
            *styling_advice* — LLM-generated (or fallback) styling tips.
        """
        if not query.strip() and image_bytes is None:
            raise ValueError("Please provide a text query or an image.")

        # ------------------------------------------------------------------
        # 1. Encode the query
        # ------------------------------------------------------------------
        text_emb = self._encoder.encode(query) if query.strip() else None

        if image_bytes is not None:
            img_emb = self._image_encoder.encode_uploaded_image(
                image_bytes, fallback_query=query or "clothing item"
            )
            if text_emb is not None:
                # Simple average fusion of text and image embeddings
                import numpy as np
                query_emb = (text_emb + img_emb) / 2.0
            else:
                query_emb = img_emb
        else:
            query_emb = text_emb

        # ------------------------------------------------------------------
        # 2. Retrieve candidates from FAISS
        # ------------------------------------------------------------------
        candidates = self._store.search(query_emb, top_k=self._top_k)
        logger.debug("Retrieved %d candidates from FAISS.", len(candidates))

        if not candidates:
            return [], "No matching products found. Try a different query."

        # ------------------------------------------------------------------
        # 3. Re-rank with LLM and generate styling advice
        # ------------------------------------------------------------------
        ranked, advice = self._advisor.rerank_and_advise(
            query=query,
            candidates=candidates,
            top_n=self._top_n,
        )
        return ranked, advice

    # ------------------------------------------------------------------
    # Convenience helper for the Streamlit app
    # ------------------------------------------------------------------

    @staticmethod
    def index_exists() -> bool:
        """Return True if the FAISS index has already been built."""
        return (
            config.FAISS_INDEX_PATH.exists() and config.METADATA_PATH.exists()
        )
