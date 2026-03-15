"""
encoder.py — Text & Image Encoder
==================================
Converts free-text queries (and optionally product images) into dense
vector embeddings that can be stored in and queried against the FAISS
vector database.

Text encoding
-------------
Uses a SentenceTransformers model (default: all-MiniLM-L6-v2) to
produce a fixed-size embedding from a combined textual representation
of a product or from a user query.

Image encoding
--------------
Optionally downloads a product thumbnail and encodes it using the same
text model via a CLIP-style text–image bridge, or simply returns None
when image support is disabled.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

import numpy as np
import requests
from PIL import Image
from sentence_transformers import SentenceTransformer

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

logger = logging.getLogger(__name__)


class TextEncoder:
    """Encodes text strings into dense embeddings using SentenceTransformers."""

    def __init__(self, model_name: str = config.TEXT_EMBEDDING_MODEL) -> None:
        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, texts: list[str] | str, batch_size: int = 64) -> np.ndarray:
        """Return L2-normalised embeddings for *texts*.

        Parameters
        ----------
        texts:
            A single string or a list of strings.
        batch_size:
            How many texts to encode per forward pass.

        Returns
        -------
        np.ndarray of shape ``(len(texts), embedding_dim)`` with float32 dtype.
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine similarity ≡ dot-product on unit vectors
        )
        return embeddings.astype(np.float32)

    # ------------------------------------------------------------------
    # Helpers for building product text representations
    # ------------------------------------------------------------------

    @staticmethod
    def build_product_text(row: dict) -> str:
        """Combine product fields into a single descriptive sentence.

        The sentence is designed to be rich enough for semantic search
        while staying concise so that short user queries still match.

        Parameters
        ----------
        row:
            A dict (or pandas Series) with the Myntra dataset columns:
            name, brand, colour, description, p_attributes, price, avg_rating.
        """
        parts: list[str] = []

        if name := str(row.get("name", "") or "").strip():
            parts.append(name)
        if brand := str(row.get("brand", "") or "").strip():
            parts.append(f"by {brand}")
        if colour := str(row.get("colour", "") or "").strip():
            parts.append(f"in {colour}")
        if desc := str(row.get("description", "") or "").strip():
            parts.append(desc[:300])  # cap very long descriptions
        if attrs := str(row.get("p_attributes", "") or "").strip():
            # p_attributes is stored as a Python dict repr; just include it verbatim
            parts.append(attrs[:200])

        return ". ".join(filter(None, parts))


class ImageEncoder:
    """Downloads a product image from a URL and encodes it.

    Currently uses the *text* encoder on an image-derived caption
    (colour + name) as a lightweight fallback when CLIP is not available.
    A full CLIP-based implementation can be dropped in here later.
    """

    def __init__(self, text_encoder: TextEncoder) -> None:
        self._text_encoder = text_encoder

    def encode_from_url(self, url: str, product_text: str) -> Optional[np.ndarray]:
        """Fetch *url* and return an embedding.

        Falls back gracefully to ``product_text`` embedding if the image
        cannot be fetched (e.g. the URL is unreachable in the current env).
        """
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            Image.open(io.BytesIO(response.content)).verify()
            # TODO: integrate a real vision model (e.g. OpenAI CLIP) here.
            # For now fall through to text-based embedding.
        except Exception as exc:
            logger.debug("Image fetch failed (%s), using text embedding.", exc)

        return self._text_encoder.encode(product_text)

    def encode_uploaded_image(
        self, image_bytes: bytes, fallback_query: str
    ) -> np.ndarray:
        """Encode a user-uploaded image.

        Parameters
        ----------
        image_bytes:
            Raw bytes of the uploaded image.
        fallback_query:
            Text to embed if image encoding is not yet available.
        """
        try:
            Image.open(io.BytesIO(image_bytes)).verify()
            # TODO: plug in a real CLIP model for image-to-embedding.
        except Exception as exc:
            logger.warning("Could not decode uploaded image: %s", exc)

        return self._text_encoder.encode(fallback_query)
