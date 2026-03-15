"""
Global configuration for SemanticClothingRecommender.

All paths and tuneable hyper-parameters are defined here so that
the rest of the code-base never hard-codes them.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables from an optional .env file
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "index"

# Raw dataset (downloaded from Kaggle)
DATASET_CSV = DATA_DIR / "Fashion Dataset.csv"

# Persisted FAISS index and product metadata
FAISS_INDEX_PATH = INDEX_DIR / "fashion.index"
METADATA_PATH = INDEX_DIR / "metadata.pkl"

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
# Any sentence-transformers compatible model name or local path.
# "all-MiniLM-L6-v2" is fast and works well for fashion text.
TEXT_EMBEDDING_MODEL = os.getenv(
    "TEXT_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIM = 384  # Must match the model above; all-MiniLM-L6-v2 → 384

# ---------------------------------------------------------------------------
# FAISS / retrieval
# ---------------------------------------------------------------------------
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "20"))  # candidates from FAISS
TOP_N_FINAL = int(os.getenv("TOP_N_FINAL", "5"))           # items shown to user

# ---------------------------------------------------------------------------
# LLM (optional – used for re-ranking & styling advice)
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
APP_TITLE = "👗 Semantic Clothing Recommender"
