"""
ingest.py — SEC EDGAR 10-K fetcher, chunker, embedder, and FAISS indexer.

Pipeline
--------
1. Download 10-K filings from SEC EDGAR for given tickers/CIKs and fiscal years.
2. Parse HTML filings with BeautifulSoup, strip XBRL boilerplate, extract plain text.
3. Chunk text at ~512 tokens per chunk with 64-token overlap using tiktoken.
4. Embed chunks with OpenAI text-embedding-3-small (batched).
5. Normalize and insert into a FAISS flat L2 index; save to disk.
6. Write chunk metadata (source, company, fiscal_year, chunk_idx, text) to
   ``metadata.json`` alongside the FAISS index file.

Requirements
------------
    OPENAI_API_KEY  — required for embedding.
    pip install faiss-cpu tiktoken beautifulsoup4 requests sec-edgar-downloader

Usage::

    from ragprobe.ingest import build_index

    build_index(tickers=["AAPL", "MSFT"], years=[2022, 2023, 2024])

CLI::

    ragprobe ingest --tickers AAPL MSFT AMZN --years 2022 2023 2024
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
    CHUNK_OVERLAP,
    CHUNK_TOKENS,
    EDGAR_DATA_DIR,
    EDGAR_USER_AGENT_EMAIL,
    EMBED_DIMENSION,
    EMBED_MODEL,
    FAISS_PATH,
)

logger = logging.getLogger(__name__)

# ── OpenAI client (lazy singleton) ───────────────────────────────────────────

_openai_client = None


def _get_openai():
    """Return shared OpenAI client for embeddings."""
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError(
                "OPENAI_API_KEY is required for embedding. "
                "Export it before running ragprobe ingest."
            )
        _openai_client = OpenAI(api_key=key)
    return _openai_client


# ── EDGAR downloader ──────────────────────────────────────────────────────────

def download_filings(
    tickers: list[str],
    years:   list[int],
    data_dir: Path = EDGAR_DATA_DIR,
) -> list[Path]:
    """
    Download 10-K filings from SEC EDGAR for the given tickers and fiscal years.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols (e.g. ``["AAPL", "MSFT"]``).
    years : list[int]
        Fiscal years to target (e.g. ``[2022, 2023, 2024]``).
    data_dir : Path
        Root directory where filings are saved.

    Returns
    -------
    list[Path]
        Paths to all downloaded filing directories.
    """
    try:
        from sec_edgar_downloader import Downloader
    except ImportError:
        raise ImportError(
            "sec-edgar-downloader not installed. Run: pip install sec-edgar-downloader"
        )

    data_dir.mkdir(parents=True, exist_ok=True)
    dl = Downloader("ragprobe", EDGAR_USER_AGENT_EMAIL, data_dir)

    filing_dirs: list[Path] = []

    for ticker in tickers:
        logger.info("Downloading 10-K filings for %s (years: %s)", ticker, years)
        try:
            # Download one filing per year; SEC EDGAR stores most recent first
            dl.get("10-K", ticker, limit=len(years))
        except Exception as exc:
            logger.error("Failed to download 10-K for %s: %s", ticker, exc)
            continue

        # Collect all filing directories for this ticker
        ticker_dir = data_dir / "sec-edgar-filings" / ticker / "10-K"
        if ticker_dir.exists():
            for d in sorted(ticker_dir.iterdir()):
                if d.is_dir():
                    filing_dirs.append(d)
        else:
            # Older downloader version layout
            alt_dir = data_dir / ticker / "10-K"
            if alt_dir.exists():
                for d in sorted(alt_dir.iterdir()):
                    if d.is_dir():
                        filing_dirs.append(d)

    logger.info("Downloaded %d filing directories", len(filing_dirs))
    return filing_dirs


# ── HTML/text parser ──────────────────────────────────────────────────────────

# XBRL boilerplate patterns to strip
_XBRL_PATTERNS: list[re.Pattern] = [
    re.compile(r"<ix:[^>]+>.*?</ix:[^>]+>", re.DOTALL | re.I),
    re.compile(r"<xbrl[^>]*>.*?</xbrl>",    re.DOTALL | re.I),
    re.compile(r"<!--.*?-->",                re.DOTALL),
]

# Noise patterns to remove from extracted text
_NOISE_RE = re.compile(
    r"(Table of Contents|EDGAR Filing|Filed\s+(?:via|pursuant|with)|"
    r"Item \d+[A-Z]?\.\s*\n|^\s*\d+\s*$)",
    re.MULTILINE | re.I,
)


def extract_text_from_filing(filing_dir: Path) -> Optional[str]:
    """
    Extract plain text from a 10-K filing directory.

    Tries HTML files first (most 10-Ks are HTML), then falls back to any
    .txt file that looks like the primary document.

    Parameters
    ----------
    filing_dir : Path
        Directory containing downloaded filing files.

    Returns
    -------
    str | None
        Extracted plain text, or None if no parseable file found.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError(
            "beautifulsoup4 not installed. Run: pip install beautifulsoup4"
        )

    # Look for HTML files; prefer the primary document (largest file)
    html_files = sorted(
        list(filing_dir.glob("*.htm")) + list(filing_dir.glob("*.html")),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )

    if not html_files:
        # Try plain text
        txt_files = sorted(filing_dir.glob("*.txt"), key=lambda p: p.stat().st_size, reverse=True)
        if not txt_files:
            logger.warning("No parseable files in %s", filing_dir)
            return None
        return txt_files[0].read_text(encoding="utf-8", errors="replace")

    html_file = html_files[0]
    raw_html  = html_file.read_text(encoding="utf-8", errors="replace")

    # Strip XBRL boilerplate before parsing
    for pat in _XBRL_PATTERNS:
        raw_html = pat.sub(" ", raw_html)

    soup = BeautifulSoup(raw_html, "html.parser")

    # Remove non-content tags
    for tag in soup(["script", "style", "head", "meta", "link",
                     "ix:header", "xbrl"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    # Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = _NOISE_RE.sub("", text)
    text = text.strip()

    if len(text) < 500:
        logger.warning("Very short text extracted from %s (%d chars)", html_file, len(text))

    return text


# ── Token-based chunker ───────────────────────────────────────────────────────

def chunk_text(
    text:     str,
    n_tokens: int = CHUNK_TOKENS,
    overlap:  int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping token-based chunks.

    Parameters
    ----------
    text : str
        Input document text.
    n_tokens : int
        Target chunk size in tokens.
    overlap : int
        Number of tokens to overlap between consecutive chunks.

    Returns
    -------
    list[str]
        List of text chunk strings.
    """
    try:
        import tiktoken
    except ImportError:
        raise ImportError("tiktoken not installed. Run: pip install tiktoken")

    enc    = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    if not tokens:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end        = min(start + n_tokens, len(tokens))
        chunk_tok  = tokens[start:end]
        chunk_text = enc.decode(chunk_tok)
        if chunk_text.strip():
            chunks.append(chunk_text.strip())
        if end >= len(tokens):
            break
        start += n_tokens - overlap

    return chunks


# ── Embedding ─────────────────────────────────────────────────────────────────

_EMBED_BATCH = 512   # Max inputs per OpenAI embeddings request


def embed_chunks(texts: list[str]) -> np.ndarray:
    """
    Embed a list of text chunks using OpenAI text-embedding-3-small.

    Batches requests to stay within the API's per-request limit.

    Parameters
    ----------
    texts : list[str]
        Text strings to embed.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(len(texts), EMBED_DIMENSION)``.
    """
    if not texts:
        return np.zeros((0, EMBED_DIMENSION), dtype=np.float32)

    client     = _get_openai()
    embeddings = []

    for i in range(0, len(texts), _EMBED_BATCH):
        batch = texts[i : i + _EMBED_BATCH]
        logger.info("Embedding batch %d–%d of %d", i, i + len(batch), len(texts))
        try:
            resp = client.embeddings.create(
                input          = batch,
                model          = EMBED_MODEL,
                encoding_format= "float",
            )
            batch_vecs = [d.embedding for d in resp.data]
            embeddings.extend(batch_vecs)
        except Exception as exc:
            logger.error("Embedding batch %d failed: %s", i, exc)
            # Fill with zeros for failed batch so indices stay aligned
            embeddings.extend([[0.0] * EMBED_DIMENSION] * len(batch))
        # Respect rate limits
        time.sleep(0.2)

    arr = np.array(embeddings, dtype=np.float32)

    # L2-normalize for cosine similarity
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    arr   = arr / norms

    return arr


# ── FAISS index builder ───────────────────────────────────────────────────────

def build_faiss_index(
    embeddings: np.ndarray,
    output_dir: Path = FAISS_PATH,
) -> None:
    """
    Build a FAISS flat L2 index from pre-computed embeddings and save to disk.

    Parameters
    ----------
    embeddings : np.ndarray
        Float32 matrix of shape ``(n_chunks, EMBED_DIMENSION)``.
    output_dir : Path
        Directory to write ``index.faiss``.
    """
    try:
        import faiss
    except ImportError:
        raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")

    output_dir.mkdir(parents=True, exist_ok=True)

    index = faiss.IndexFlatL2(EMBED_DIMENSION)
    index.add(embeddings)
    faiss.write_index(index, str(output_dir / "index.faiss"))
    logger.info("FAISS index built: %d vectors → %s/index.faiss", index.ntotal, output_dir)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def build_index(
    tickers:   list[str],
    years:     list[int],
    data_dir:  Path = EDGAR_DATA_DIR,
    index_dir: Path = FAISS_PATH,
) -> dict:
    """
    Full ingest pipeline: download → parse → chunk → embed → index.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols to ingest (e.g. ``["AAPL", "MSFT", "AMZN"]``).
    years : list[int]
        Target fiscal years (e.g. ``[2022, 2023, 2024]``).
    data_dir : Path
        Where EDGAR filings are downloaded.
    index_dir : Path
        Where the FAISS index and metadata are written.

    Returns
    -------
    dict
        Summary: ``{"tickers": ..., "years": ..., "total_chunks": ..., "index_dir": ...}``
    """
    logger.info("Starting ingest for tickers=%s years=%s", tickers, years)

    # ── Step 1: Download ──────────────────────────────────────────────────────
    filing_dirs = download_filings(tickers, years, data_dir)
    if not filing_dirs:
        raise RuntimeError(
            "No filings downloaded. Check EDGAR_USER_AGENT_EMAIL and connectivity."
        )

    # ── Step 2: Parse + chunk ─────────────────────────────────────────────────
    all_chunks:   list[str]  = []
    all_metadata: list[dict] = []

    for filing_dir in filing_dirs:
        # Extract ticker from path heuristic: …/<TICKER>/10-K/<accession>/
        parts = filing_dir.parts
        ticker = "UNKNOWN"
        for i, p in enumerate(parts):
            if p == "10-K" and i > 0:
                ticker = parts[i - 1].upper()
                break

        text = extract_text_from_filing(filing_dir)
        if not text:
            continue

        chunks = chunk_text(text)
        logger.info(
            "%s: extracted %d chunks from %s", ticker, len(chunks), filing_dir.name
        )

        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                "ticker":      ticker,
                "fiscal_year": None,   # Could be parsed from filing if needed
                "source":      str(filing_dir),
                "chunk_idx":   idx,
                "text":        chunk,
            })

    if not all_chunks:
        raise RuntimeError("No text extracted from any downloaded filing.")

    logger.info("Total chunks to embed: %d", len(all_chunks))

    # ── Step 3: Embed ─────────────────────────────────────────────────────────
    embeddings = embed_chunks(all_chunks)

    # ── Step 4: Build FAISS index ─────────────────────────────────────────────
    build_faiss_index(embeddings, index_dir)

    # ── Step 5: Save metadata ─────────────────────────────────────────────────
    index_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = index_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(all_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Metadata written to %s", metadata_path)

    return {
        "tickers":      tickers,
        "years":        years,
        "total_chunks": len(all_chunks),
        "index_dir":    str(index_dir),
    }
