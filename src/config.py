"""Central configuration loaded from environment / .env.

Import settings from here instead of hardcoding connection strings, model names,
or paths anywhere else (see CLAUDE.md §7).
"""

import os

from dotenv import load_dotenv

load_dotenv()

# --- Neo4j ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "changeme")

# --- Qdrant ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "papers")

# --- Ollama ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")  # smoke model; production pick = M8.3

# --- Embeddings ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# --- OpenAlex (data source; see docs/data-source.md) ---
OPENALEX_BASE_URL = os.getenv("OPENALEX_BASE_URL", "https://api.openalex.org/works")
OPENALEX_MAILTO = os.getenv("OPENALEX_MAILTO", "shayanx13@gmail.com")  # polite pool

# --- Data paths ---
DATA_RAW = os.getenv("DATA_RAW", "data/raw")
DATA_INTERIM = os.getenv("DATA_INTERIM", "data/interim")
DATA_PROCESSED = os.getenv("DATA_PROCESSED", "data/processed")
