"""
rag_pipeline.py — Baseline and hardened RAG pipelines over FAISS-indexed SEC 10-K chunks.

Two modes
---------
baseline:
  - Retrieve top-5 chunks from FAISS (cosine similarity).
  - Generic system prompt.
  - Generate response with claude-haiku-4-5-20251001.

hardened:
  - Retrieve top-10 chunks from FAISS.
  - Re-rank using a Claude scoring pass; keep top-3.
  - Refuse to answer if no chunk scores above CONFIDENCE_THRESHOLD.
  - Anti-hallucination system prompt; requires source citation.

Both modes return::

    {
      "query":          str,
      "context_chunks": list[str],
      "response":       str,
      "mode":           str,
      "latency_ms":     int,
    }

Usage::

    from ragprobe.rag_pipeline import run

    result = run("What was Apple's net income in FY2023?", mode="baseline")
    print(result["response"])
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ragprobe.config import (
    CONFIDENCE_THRESHOLD,
    EMBED_DIMENSION,
    EMBED_MODEL,
    FAISS_PATH,
    RAG_MODEL,
    TOP_K_BASELINE,
    TOP_K_HARDENED,
    TOP_K_RERANK,
)

logger = logging.getLogger(__name__)

# ── Client singletons ─────────────────────────────────────────────────────────

_anthropic_client = None
_openai_client    = None
_faiss_index      = None
_chunk_metadata: list[dict] = []


def _get_anthropic():
    """Return shared Anthropic client."""
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import Anthropic
        key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("API_KEY")
        if not key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY or API_KEY must be set for RAG pipeline generation."
            )
        _anthropic_client = Anthropic(api_key=key)
    return _anthropic_client


def _get_openai():
    """Return shared OpenAI client (used only for embeddings)."""
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError(
                "OPENAI_API_KEY is required for query embedding. "
                "Export it before running ragprobe."
            )
        _openai_client = OpenAI(api_key=key)
    return _openai_client


# ── FAISS index loader ────────────────────────────────────────────────────────

def _load_index(faiss_path: Path = FAISS_PATH) -> tuple:
    """
    Load FAISS index and chunk metadata from disk.

    Parameters
    ----------
    faiss_path : Path
        Directory containing ``index.faiss`` and ``metadata.json``.

    Returns
    -------
    tuple[faiss.Index, list[dict]]
        Loaded FAISS index and metadata list.

    Raises
    ------
    FileNotFoundError
        If the index files do not exist.
    """
    global _faiss_index, _chunk_metadata

    if _faiss_index is not None:
        return _faiss_index, _chunk_metadata

    try:
        import faiss
    except ImportError:
        raise ImportError(
            "faiss-cpu not installed. Run: pip install faiss-cpu"
        )

    index_file    = faiss_path / "index.faiss"
    metadata_file = faiss_path / "metadata.json"

    if not index_file.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_file}. "
            "Run 'ragprobe ingest' to build the index first."
        )
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"FAISS metadata not found at {metadata_file}. "
            "Run 'ragprobe ingest' to build the index first."
        )

    _faiss_index   = faiss.read_index(str(index_file))
    _chunk_metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    logger.info("Loaded FAISS index (%d vectors) from %s", _faiss_index.ntotal, faiss_path)
    return _faiss_index, _chunk_metadata


# ── Embedding helper ──────────────────────────────────────────────────────────

def _embed_query(query: str) -> np.ndarray:
    """
    Embed a query using OpenAI text-embedding-3-small.

    Parameters
    ----------
    query : str

    Returns
    -------
    np.ndarray
        Normalized float32 vector of shape (1, EMBED_DIMENSION).
    """
    res = _get_openai().embeddings.create(
        input          = [query],
        model          = EMBED_MODEL,
        encoding_format= "float",
    )
    vec = np.array(res.data[0].embedding, dtype=np.float32).reshape(1, -1)
    # Normalize for cosine similarity on L2 index
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


# ── Retrieval ─────────────────────────────────────────────────────────────────

def _retrieve(query: str, k: int, faiss_path: Path = FAISS_PATH) -> list[str]:
    """
    Retrieve the top-k most relevant chunks for a query.

    Parameters
    ----------
    query : str
    k : int
        Number of chunks to retrieve.
    faiss_path : Path

    Returns
    -------
    list[str]
        Retrieved chunk texts in ranked order.
    """
    index, metadata = _load_index(faiss_path)
    q_vec = _embed_query(query)
    distances, indices = index.search(q_vec, k)
    chunks = []
    for idx in indices[0]:
        if 0 <= idx < len(metadata):
            chunks.append(metadata[idx].get("text", ""))
    return chunks


# ── Re-ranker (hardened mode) ─────────────────────────────────────────────────

_RERANK_PROMPT = """Rate the relevance of this document chunk to the query on a scale of 0.0 to 1.0.

Query: {query}

Chunk:
{chunk}

Return ONLY a JSON object: {{"score": <float>}}
No other text."""


def _rerank(query: str, chunks: list[str], top_n: int) -> list[tuple[float, str]]:
    """
    Score each chunk's relevance to the query using Claude.

    Parameters
    ----------
    query : str
    chunks : list[str]
        Candidate chunks from initial retrieval.
    top_n : int
        Number of top-scoring chunks to keep.

    Returns
    -------
    list[tuple[float, str]]
        ``(score, chunk_text)`` pairs for the top_n highest-scoring chunks,
        in descending score order.
    """
    scored = []
    for chunk in chunks:
        try:
            resp = _get_anthropic().messages.create(
                model      = RAG_MODEL,
                max_tokens = 64,
                temperature= 0,
                messages   = [{
                    "role":    "user",
                    "content": _RERANK_PROMPT.format(query=query, chunk=chunk[:800]),
                }],
            )
            raw   = resp.content[0].text.strip()
            raw   = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
            data  = json.loads(raw)
            score = max(0.0, min(1.0, float(data.get("score", 0.0))))
        except Exception as exc:
            logger.warning("Re-rank scoring failed for chunk: %s", exc)
            score = 0.0
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_n]


# ── System prompts ────────────────────────────────────────────────────────────

_BASELINE_SYSTEM = (
    "You are a financial document analyst. "
    "Answer the question using only the provided context. "
    "If the context does not contain the answer, say so clearly."
)

_HARDENED_SYSTEM = (
    "You are a precise financial document analyst. "
    "Rules you must follow:\n"
    "1. Answer ONLY using claims explicitly supported by the provided context chunks.\n"
    "2. Cite the source chunk number for every factual claim, e.g. '[Chunk 2]'.\n"
    "3. If the context does not contain sufficient information to answer the question, "
    "respond with: 'I cannot answer this question based on the available documents.'\n"
    "4. Do not infer, extrapolate, or speculate beyond what the context states.\n"
    "5. Do not fabricate figures, dates, names, or any other details."
)


# ── Main pipeline entry point ─────────────────────────────────────────────────

def run(
    query:      str,
    mode:       str = "baseline",
    faiss_path: Path = FAISS_PATH,
) -> dict:
    """
    Run the RAG pipeline in the specified mode.

    Parameters
    ----------
    query : str
        User question to answer.
    mode : str
        ``"baseline"`` or ``"hardened"``.
    faiss_path : Path
        Directory containing the FAISS index.  Defaults to ``config.FAISS_PATH``.

    Returns
    -------
    dict
        Keys: ``query``, ``context_chunks``, ``response``, ``mode``, ``latency_ms``.

    Raises
    ------
    ValueError
        If ``mode`` is not ``"baseline"`` or ``"hardened"``.
    FileNotFoundError
        If the FAISS index has not been built yet.
    """
    if mode not in ("baseline", "hardened"):
        raise ValueError(f"mode must be 'baseline' or 'hardened', got: {mode!r}")

    start = time.perf_counter()

    # ── Retrieval ─────────────────────────────────────────────────────────────
    if mode == "baseline":
        chunks       = _retrieve(query, k=TOP_K_BASELINE, faiss_path=faiss_path)
        system_prompt = _BASELINE_SYSTEM
    else:
        candidates = _retrieve(query, k=TOP_K_HARDENED, faiss_path=faiss_path)
        ranked     = _rerank(query, candidates, top_n=TOP_K_RERANK)

        # Filter by confidence threshold
        chunks = [chunk for score, chunk in ranked if score >= CONFIDENCE_THRESHOLD]

        if not chunks:
            latency_ms = int((time.perf_counter() - start) * 1000)
            return {
                "query":          query,
                "context_chunks": [],
                "response":       (
                    "I cannot answer this question based on the available documents. "
                    "No retrieved chunks met the confidence threshold."
                ),
                "mode":       mode,
                "latency_ms": latency_ms,
            }
        system_prompt = _HARDENED_SYSTEM

    # ── Context formatting ────────────────────────────────────────────────────
    context_block = "\n\n---\n\n".join(
        f"[Chunk {i + 1}]\n{chunk}" for i, chunk in enumerate(chunks)
    )
    user_message = f"Context:\n{context_block}\n\nQuestion: {query}"

    # ── Generation ────────────────────────────────────────────────────────────
    try:
        resp = _get_anthropic().messages.create(
            model      = RAG_MODEL,
            max_tokens = 1024,
            system     = system_prompt,
            messages   = [{"role": "user", "content": user_message}],
        )
        response_text = resp.content[0].text.strip()
    except Exception as exc:
        logger.error("RAG generation failed: %s", exc)
        response_text = f"[Pipeline error: {exc}]"

    latency_ms = int((time.perf_counter() - start) * 1000)

    return {
        "query":          query,
        "context_chunks": chunks,
        "response":       response_text,
        "mode":           mode,
        "latency_ms":     latency_ms,
    }
