"""
config.py — Central configuration for ragprobe.

All magic numbers, model names, and thresholds live here.
Override any value by setting the corresponding environment variable before
importing this module.  No .env loading is performed here — call
``load_dotenv()`` at the entry point if needed.
"""

from __future__ import annotations

import os
import pathlib

# ── Models ────────────────────────────────────────────────────────────────────

JUDGE_MODEL: str = os.getenv("RAGPROBE_JUDGE_MODEL", "claude-haiku-4-5-20251001")
"""Anthropic model used as LLM judge for faithfulness/relevance/context recall."""

EMBED_MODEL: str = os.getenv("RAGPROBE_EMBED_MODEL", "text-embedding-3-small")
"""OpenAI embedding model — used ONLY for FAISS index construction and query embedding."""

RAG_MODEL: str = os.getenv("RAGPROBE_RAG_MODEL", "claude-haiku-4-5-20251001")
"""Anthropic model used for RAG response generation (baseline and hardened pipelines)."""

# ── Chunking ──────────────────────────────────────────────────────────────────

CHUNK_TOKENS: int = int(os.getenv("RAGPROBE_CHUNK_TOKENS", "512"))
"""Target chunk size in tokens for FAISS index construction."""

CHUNK_OVERLAP: int = int(os.getenv("RAGPROBE_CHUNK_OVERLAP", "64"))
"""Token overlap between consecutive chunks."""

# ── Retrieval ─────────────────────────────────────────────────────────────────

TOP_K_BASELINE: int = int(os.getenv("RAGPROBE_TOP_K_BASELINE", "5"))
"""Number of chunks retrieved in baseline mode."""

TOP_K_HARDENED: int = int(os.getenv("RAGPROBE_TOP_K_HARDENED", "10"))
"""Number of chunks retrieved before re-ranking in hardened mode."""

TOP_K_RERANK: int = int(os.getenv("RAGPROBE_TOP_K_RERANK", "5"))
"""Number of chunks kept after re-ranking in hardened mode."""

CONFIDENCE_THRESHOLD: float = float(os.getenv("RAGPROBE_CONFIDENCE_THRESHOLD", "0.4"))
"""Minimum re-rank score (0–1) to include a chunk in hardened mode response."""

EMBED_DIMENSION: int = 1536
"""Output dimension of text-embedding-3-small. Not overridable — tied to model."""

# ── Scoring thresholds ────────────────────────────────────────────────────────

FAITHFULNESS_THRESHOLD: float = float(os.getenv("RAGPROBE_FAITHFULNESS_THRESHOLD", "0.75"))
"""Score below this = faithfulness failure."""

RELEVANCE_THRESHOLD: float = float(os.getenv("RAGPROBE_RELEVANCE_THRESHOLD", "0.75"))
"""Score below this = relevance failure."""

CONTEXT_RECALL_THRESHOLD: float = float(os.getenv("RAGPROBE_CONTEXT_RECALL_THRESHOLD", "0.70"))
"""Score below this = context recall failure."""

# ── Paths ─────────────────────────────────────────────────────────────────────

DB_PATH: pathlib.Path = pathlib.Path(
    os.getenv("RAGPROBE_DB_PATH", str(pathlib.Path.home() / ".ragprobe" / "ragprobe.db"))
)
"""SQLite database file. Defaults to ~/.ragprobe/ragprobe.db."""

FAISS_PATH: pathlib.Path = pathlib.Path(
    os.getenv("RAGPROBE_FAISS_PATH", "faiss_index")
)
"""Directory containing index.faiss + metadata.json."""

REPORTS_DIR: pathlib.Path = pathlib.Path(
    os.getenv("RAGPROBE_REPORTS_DIR", "reports")
)
"""Output directory for JSON/CSV/text report files (relative to CWD)."""

EDGAR_DATA_DIR: pathlib.Path = pathlib.Path(
    os.getenv("RAGPROBE_EDGAR_DIR", "sec_edgar_filings")
)
"""Root directory where sec-edgar-downloader stores downloaded filings."""

# ── Internal paths (not overridable — relative to this package) ───────────────

_PACKAGE_DIR: pathlib.Path = pathlib.Path(__file__).parent
PROMPTS_DIR: pathlib.Path = _PACKAGE_DIR / "prompts"
JUDGE_PROMPT_FILE: pathlib.Path = PROMPTS_DIR / "judge_prompt_v1.txt"
PROMPTS_FILE: pathlib.Path = PROMPTS_DIR / "prompts.json"

# ── SEC EDGAR ─────────────────────────────────────────────────────────────────

EDGAR_USER_AGENT_EMAIL: str = os.getenv(
    "RAGPROBE_EDGAR_EMAIL", "ragprobe@example.com"
)
"""Email required by SEC EDGAR fair-access policy for the User-Agent header."""
