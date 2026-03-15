"""
vector_store.py — FAISS Vector Database
=========================================
Wraps a FAISS index together with product metadata so that the
recommender can perform fast approximate-nearest-neighbour search and
immediately look up the matching product records.

Usage
-----
Build & save (run once, from preprocessing.ipynb or a script)::

    store = VectorStore(dim=384)
    store.add(embeddings, metadata_list)
    store.save()

Load & query (in the app)::

    store = VectorStore.load()
    results = store.search(query_embedding, top_k=5)
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-backed vector store with attached metadata."""

    def __init__(self, dim: int = config.EMBEDDING_DIM) -> None:
        self._dim = dim
        # IndexFlatIP with pre-normalised vectors ↔ cosine similarity
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(dim)
        self._metadata: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Building the index
    # ------------------------------------------------------------------

    def add(
        self, embeddings: np.ndarray, metadata: list[dict[str, Any]]
    ) -> None:
        """Add *embeddings* and associated *metadata* to the index.

        Parameters
        ----------
        embeddings:
            Float32 array of shape ``(n, dim)`` with L2-normalised rows.
        metadata:
            List of ``n`` dicts, one per embedding, containing product
            fields (name, price, img URL, …).
        """
        if len(embeddings) != len(metadata):
            raise ValueError(
                f"embeddings ({len(embeddings)}) and metadata ({len(metadata)}) "
                "must have the same length."
            )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        self._index.add(embeddings)
        self._metadata.extend(metadata)
        logger.info(
            "Added %d vectors; index now contains %d.", len(embeddings), self._index.ntotal
        )

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def search(
        self, query_embedding: np.ndarray, top_k: int = config.TOP_K_RETRIEVAL
    ) -> list[dict[str, Any]]:
        """Return the *top_k* most similar products to *query_embedding*.

        Parameters
        ----------
        query_embedding:
            Float32 array of shape ``(1, dim)`` or ``(dim,)``, L2-normalised.
        top_k:
            Number of nearest neighbours to retrieve.

        Returns
        -------
        List of dicts, each being a product metadata record enriched with
        a ``score`` key (cosine similarity, higher = more similar).
        """
        vec = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        k = min(top_k, self._index.ntotal)
        if k == 0:
            logger.warning("Vector store is empty — returning no results.")
            return []

        scores, indices = self._index.search(vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue  # FAISS pads with -1 when fewer than k items exist
            # Shallow-copy to avoid mutating the stored metadata record
            item = dict(self._metadata[idx])
            item["score"] = float(score)
            results.append(item)
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self,
        index_path: Path | str = config.FAISS_INDEX_PATH,
        meta_path: Path | str = config.METADATA_PATH,
    ) -> None:
        """Persist the FAISS index and metadata to disk."""
        index_path = Path(index_path)
        meta_path = Path(meta_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(index_path))
        with open(meta_path, "wb") as f:
            pickle.dump(self._metadata, f)
        logger.info("Saved index → %s, metadata → %s.", index_path, meta_path)

    @classmethod
    def load(
        cls,
        index_path: Path | str = config.FAISS_INDEX_PATH,
        meta_path: Path | str = config.METADATA_PATH,
    ) -> "VectorStore":
        """Load a previously saved VectorStore from disk.

        Raises
        ------
        FileNotFoundError
            If either file is missing (run preprocessing.ipynb first).
        """
        index_path = Path(index_path)
        meta_path = Path(meta_path)
        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at '{index_path}'. "
                "Please run preprocessing.ipynb to build the index first."
            )
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found at '{meta_path}'. "
                "Please run preprocessing.ipynb to build the index first."
            )

        store = cls.__new__(cls)
        store._index = faiss.read_index(str(index_path))
        store._dim = store._index.d
        with open(meta_path, "rb") as f:
            store._metadata = pickle.load(f)
        logger.info(
            "Loaded index with %d vectors from '%s'.", store._index.ntotal, index_path
        )
        return store

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._index.ntotal

    @property
    def is_empty(self) -> bool:
        return self._index.ntotal == 0
