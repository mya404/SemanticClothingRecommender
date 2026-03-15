"""
llm_advisor.py — LLM Re-ranking & Styling Advice
==================================================
Uses an OpenAI chat model to:
1. Re-rank the FAISS candidates based on the original user query.
2. Generate a short, personalised styling tip for the final selection.

The module degrades gracefully: if no API key is configured the
re-ranking step is skipped and the FAISS ordering is used as-is.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

logger = logging.getLogger(__name__)

# Lazy import so that the rest of the app works without openai installed
try:
    from openai import OpenAI
    _openai_available = True
except ImportError:
    _openai_available = False


class LLMAdvisor:
    """Wraps OpenAI to re-rank candidates and produce styling advice."""

    def __init__(
        self,
        api_key: str = config.OPENAI_API_KEY,
        model: str = config.OPENAI_MODEL,
    ) -> None:
        self._model = model
        self._enabled = bool(api_key) and _openai_available
        if self._enabled:
            self._client = OpenAI(api_key=api_key)
            logger.info("LLM advisor enabled (model=%s).", model)
        else:
            logger.info(
                "LLM advisor disabled — no API key or openai not installed. "
                "FAISS ranking will be used as-is."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank_and_advise(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int = config.TOP_N_FINAL,
    ) -> tuple[list[dict[str, Any]], str]:
        """Re-rank *candidates* and return styling advice.

        Parameters
        ----------
        query:
            The original user query.
        candidates:
            Product dicts retrieved from FAISS (already sorted by similarity).
        top_n:
            How many items to return in the final list.

        Returns
        -------
        (ranked_items, styling_advice)
            *ranked_items* — list of up to *top_n* product dicts, each
            enriched with an ``llm_score`` key (1-based rank).
            *styling_advice* — a markdown-formatted string with outfit tips.
        """
        if not self._enabled or not candidates:
            advice = self._fallback_advice(query, candidates[:top_n])
            return candidates[:top_n], advice

        try:
            ranked, advice = self._call_llm(query, candidates, top_n)
            return ranked, advice
        except Exception as exc:
            logger.warning("LLM call failed (%s); falling back to FAISS order.", exc)
            advice = self._fallback_advice(query, candidates[:top_n])
            return candidates[:top_n], advice

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int,
    ) -> tuple[list[dict[str, Any]], str]:
        """Send candidates to the LLM and parse its response."""
        # Build a compact catalogue string for the model
        catalogue_lines = []
        for i, item in enumerate(candidates):
            catalogue_lines.append(
                f"{i}: {item.get('name', 'Unknown')} | {item.get('brand', '')} | "
                f"{item.get('colour', '')} | ₹{item.get('price', 'N/A')} | "
                f"{str(item.get('description', ''))[:150]}"
            )
        catalogue = "\n".join(catalogue_lines)

        system_prompt = (
            "You are an expert fashion stylist. "
            "Given a user query and a list of clothing options, you must:\n"
            "1. Select the best matching items from the list.\n"
            "2. Return a JSON object with two keys:\n"
            "   - \"ranked_indices\": list of item indices (0-based) in priority order, "
            f"     at most {top_n} items.\n"
            "   - \"styling_advice\": a friendly, markdown-formatted paragraph with "
            "     outfit-building tips relevant to the query and the selected items.\n"
            "Output ONLY valid JSON — no markdown code fences."
        )
        user_prompt = f"User query: \"{query}\"\n\nAvailable items:\n{catalogue}"

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=512,
        )
        raw = response.choices[0].message.content.strip()

        # Parse JSON (handle minor formatting issues)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON block if the model wrapped it
            import re
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            data = json.loads(match.group()) if match else {}

        ranked_indices: list[int] = data.get("ranked_indices") or list(range(len(candidates)))
        styling_advice: str = data.get(
            "styling_advice",
            self._fallback_advice(query, candidates[:top_n]),
        )

        ranked_items = []
        seen = set()
        for rank, idx in enumerate(ranked_indices[:top_n], start=1):
            if 0 <= idx < len(candidates) and idx not in seen:
                item = dict(candidates[idx])
                item["llm_score"] = rank
                ranked_items.append(item)
                seen.add(idx)

        return ranked_items, styling_advice

    @staticmethod
    def _fallback_advice(query: str, items: list[dict[str, Any]]) -> str:
        """Simple rule-based advice when the LLM is unavailable."""
        names = [item.get("name", "item") for item in items[:3]]
        listed = ", ".join(names) if names else "these items"
        return (
            f"Based on your search for **\"{query}\"**, we found {listed} "
            f"and more. Style them with complementary accessories and footwear "
            f"to complete your look!"
        )
