"""
CourseMate — central configuration.

All tunable values live here and are overridable via environment variables
(or a local .env file, see .env.example). Nothing in app.py or ingest.py
should hardcode a model name, path, or chunk size — they should import
from here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # loads .env if present; no-op in production if env vars are set another way

# ── Paths ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "faiss_index"

DATA_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

# ── LLM provider toggle ─────────────────────────────────────────────────
# "ollama" -> fully local, offline, free, needs Ollama installed
# "groq"   -> cloud, fast, free-tier API, needs GROQ_API_KEY
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# ── Embeddings ───────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "thenlper/gte-small")

# ── Chunking ─────────────────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "75"))

# ── Retrieval ────────────────────────────────────────────────────────────
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))

# ── App metadata ─────────────────────────────────────────────────────────
APP_TITLE = os.getenv("APP_TITLE", "CourseMate")
APP_ICON = os.getenv("APP_ICON", "🎓")
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "CourseMate")
