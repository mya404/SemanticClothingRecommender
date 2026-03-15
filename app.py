"""
app.py — Streamlit Web Application
====================================
Launches the SemanticClothingRecommender UI.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

import logging
import os

import streamlit as st

import config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Helper — lazy-load the Recommender (cached across reruns)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading recommendation engine …")
def load_recommender():
    from src.recommender import Recommender

    return Recommender()


# ---------------------------------------------------------------------------
# Sidebar — settings & info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")

    top_n = st.slider(
        "Number of recommendations",
        min_value=1,
        max_value=20,
        value=config.TOP_N_FINAL,
        step=1,
    )

    st.markdown("---")
    st.markdown(
        "**📖 How it works**\n\n"
        "1. Type a query describing the item you want.\n"
        "2. *(Optional)* Upload an image for visual search.\n"
        "3. The system encodes your input, searches the vector database, "
        "   and optionally re-ranks results with an LLM.\n"
        "4. Styled recommendations are shown with images and details."
    )
    st.markdown("---")
    st.markdown(
        "**🔑 LLM (optional)**\n\n"
        "Set `OPENAI_API_KEY` in your environment or a `.env` file to enable "
        "AI-powered re-ranking and personalised styling advice."
    )

    llm_status = "✅ Enabled" if config.OPENAI_API_KEY else "⚠️ Disabled (no API key)"
    st.info(f"LLM advisor: {llm_status}")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.title(config.APP_TITLE)
st.caption("Semantic fashion search powered by SentenceTransformers + FAISS")

# Check that the index exists
if not config.FAISS_INDEX_PATH.exists():
    st.error(
        "⚠️ **Vector index not found.**\n\n"
        "Please run **`preprocessing.ipynb`** first to build the FAISS index "
        "from the Myntra Fashion Dataset."
    )
    st.stop()

# Load the recommender (cached)
recommender = load_recommender()
# Override top_n from the sidebar slider
recommender._top_n = top_n

# ---------------------------------------------------------------------------
# Helper — renders a single product card inside a Streamlit column
# ---------------------------------------------------------------------------
def _render_product_card(col, item: dict) -> None:
    """Render a single product card inside a Streamlit column."""
    img_url = str(item.get("img", "") or "").strip()

    with col:
        if img_url:
            try:
                st.image(img_url, use_container_width=True)
            except Exception:
                st.image(
                    "https://via.placeholder.com/300x400?text=No+Image",
                    use_container_width=True,
                )
        else:
            st.image(
                "https://via.placeholder.com/300x400?text=No+Image",
                use_container_width=True,
            )

        name = item.get("name", "Unknown product")
        st.markdown(f"**{name}**")

        brand = item.get("brand", "")
        colour = item.get("colour", "")
        price = item.get("price", "")
        rating = item.get("avg_rating", "")

        meta_parts = []
        if brand:
            meta_parts.append(f"🏷️ {brand}")
        if colour:
            meta_parts.append(f"🎨 {colour}")
        if price:
            meta_parts.append(f"💰 ₹{price}")
        if rating:
            meta_parts.append(f"⭐ {rating}")

        st.caption("  |  ".join(meta_parts))

        desc = str(item.get("description", "") or "")
        if desc:
            with st.expander("Description"):
                st.write(desc[:400])

        score = item.get("score")
        if score is not None:
            st.progress(
                min(float(score), 1.0),
                text=f"Similarity: {score:.2%}",
            )


# ---------------------------------------------------------------------------
# Query input
# ---------------------------------------------------------------------------
col_text, col_img = st.columns([3, 1])

with col_text:
    query = st.text_input(
        "🔍 Describe what you are looking for",
        placeholder="e.g. red floral kurta for a summer wedding",
    )

with col_img:
    uploaded_file = st.file_uploader(
        "📷 Upload an image (optional)",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a reference image for visual similarity search.",
    )

image_bytes = uploaded_file.read() if uploaded_file else None

search_clicked = st.button("🔎 Find Recommendations", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Run search
# ---------------------------------------------------------------------------
if search_clicked:
    if not query.strip() and image_bytes is None:
        st.warning("Please enter a search query or upload an image.")
    else:
        with st.spinner("Searching …"):
            try:
                items, styling_advice = recommender.recommend(
                    query=query,
                    image_bytes=image_bytes,
                )
            except Exception as exc:
                st.error(f"Search failed: {exc}")
                logger.exception("Search error")
                items, styling_advice = [], ""

        if not items:
            st.info("No results found. Try a different query.")
        else:
            # ------------------------------------------------------------------
            # Styling advice banner
            # ------------------------------------------------------------------
            st.markdown("---")
            st.subheader("✨ Styling Advice")
            st.markdown(styling_advice)
            st.markdown("---")

            # ------------------------------------------------------------------
            # Product cards
            # ------------------------------------------------------------------
            st.subheader(f"🛍️ Top {len(items)} Recommendations")

            cols_per_row = 3
            for row_start in range(0, len(items), cols_per_row):
                row_items = items[row_start : row_start + cols_per_row]
                cols = st.columns(cols_per_row)
                for col, item in zip(cols, row_items):
                    with col:
                        _render_product_card(col, item)
