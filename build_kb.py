"""
build_kb.py — Build the kb_out/ knowledge base required by hw11.

Downloads SEC 10-K filings, parses, chunks, and embeds them using ragprobe's
ingest pipeline, then saves the result in the format expected by hw11.py:

    kb_out/embeddings.npy   — float32 numpy array (N, 1536)
    kb_out/metadata.json    — list of {ticker, text, ...} dicts

Run:
    python build_kb.py

Requires:
    OPENAI_API_KEY in .env
    pip install ragprobe  (or install from source)

Customise:
    Edit TICKERS and YEARS below to change what gets ingested.
    Downloading all 8 tickers × 3 years ≈ 15–20 min and costs ~$0.50–$1 in embeddings.
    Start small (2 tickers, 1 year) for a quick test.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

TICKERS = ["AAPL", "MSFT"]          # add more: NVDA AMZN GOOGL META TSLA NFLX
YEARS   = [2024]                    # add more years for broader coverage
KB_DIR  = Path(__file__).resolve().parent / "kb_out"
DATA_DIR = Path(__file__).resolve().parent / "sec_edgar_filings"

# ── Import ragprobe ingest helpers ────────────────────────────────────────────

try:
    from ragprobe.ingest import (
        download_filings,
        extract_text_from_filing,
        chunk_text,
        embed_chunks,
    )
except ImportError as exc:
    sys.exit(f"[error] ragprobe not installed or import failed: {exc}\n"
             "Run: pip install -e .")

# ── Build ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("[error] OPENAI_API_KEY not set. Add it to your .env file.")

    KB_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Download filings ───────────────────────────────────────────────────
    logger.info("Downloading 10-K filings for %s (years: %s)", TICKERS, YEARS)
    filing_dirs = download_filings(TICKERS, YEARS, DATA_DIR)

    if not filing_dirs:
        sys.exit("[error] No filings downloaded. Check connectivity and EDGAR email.")

    logger.info("Downloaded %d filing directories.", len(filing_dirs))

    # ── 2. Parse + chunk ──────────────────────────────────────────────────────
    all_chunks:   list[str]  = []
    all_metadata: list[dict] = []

    for filing_dir in filing_dirs:
        # Infer ticker from path: …/sec-edgar-filings/<TICKER>/10-K/<accession>/
        parts  = filing_dir.parts
        ticker = "UNKNOWN"
        for i, p in enumerate(parts):
            if p == "10-K" and i > 0:
                ticker = parts[i - 1].upper()
                break

        text = extract_text_from_filing(filing_dir)
        if not text:
            logger.warning("Skipping %s — no text extracted.", filing_dir)
            continue

        chunks = chunk_text(text)
        logger.info("%s: %d chunks from %s", ticker, len(chunks), filing_dir.name)

        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                "ticker":    ticker,
                "source":    str(filing_dir),
                "chunk_idx": idx,
                "text":      chunk,
            })

    if not all_chunks:
        sys.exit("[error] No text could be extracted from any filing.")

    logger.info("Total chunks to embed: %d", len(all_chunks))

    # ── 3. Embed ──────────────────────────────────────────────────────────────
    logger.info("Embedding chunks with text-embedding-3-small ...")
    embeddings = embed_chunks(all_chunks)   # float32 (N, 1536), L2-normalised

    # ── 4. Save ───────────────────────────────────────────────────────────────
    emb_path  = KB_DIR / "embeddings.npy"
    meta_path = KB_DIR / "metadata.json"

    np.save(str(emb_path), embeddings)
    meta_path.write_text(
        json.dumps(all_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("Saved %d embeddings → %s", len(embeddings), emb_path)
    logger.info("Saved metadata      → %s", meta_path)
    logger.info("Done. You can now run:")
    logger.info("    python examples/hw11_ragprobe_eval.py")


if __name__ == "__main__":
    main()
