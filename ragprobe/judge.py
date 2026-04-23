"""
judge.py — LLM-as-judge scorer for ragprobe probe runs.

Single API call returns faithfulness, relevance, and context_recall scores
plus per-dimension rationale.  Judge prompt is versioned in
``ragprobe/prompts/judge_prompt_v1.txt`` and loaded at import time.

Usage::

    from ragprobe.judge import score

    result = score(
        query    = "What was Apple's revenue in FY2023?",
        context  = "Apple reported total net sales of $383.3 billion...",
        response = "Apple's revenue in FY2023 was $383.3 billion.",
    )
    print(result["faithfulness"])   # 0.95
    print(result["rationale"]["faithfulness"])
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from ragprobe.config import JUDGE_MODEL, JUDGE_PROMPT_FILE

logger = logging.getLogger(__name__)

# ── Load judge prompt at import time ─────────────────────────────────────────

def _load_prompt() -> str:
    """Load the versioned judge system prompt from disk."""
    if not JUDGE_PROMPT_FILE.exists():
        raise FileNotFoundError(
            f"Judge prompt file not found: {JUDGE_PROMPT_FILE}\n"
            "Expected location: ragprobe/prompts/judge_prompt_v1.txt"
        )
    return JUDGE_PROMPT_FILE.read_text(encoding="utf-8").strip()


_SYSTEM_PROMPT: str = _load_prompt()

# ── Anthropic client (lazy singleton) ────────────────────────────────────────

_client = None


def _get_client():
    """Return a shared Anthropic client, initializing it on first call."""
    global _client
    if _client is None:
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            )
        key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("API_KEY")
        if not key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY or API_KEY must be set to use the judge."
            )
        _client = Anthropic(api_key=key)
    return _client


# ── JSON parsing helper ───────────────────────────────────────────────────────

def _parse_json(raw: str) -> dict:
    """
    Parse JSON from a model response, stripping markdown code fences if present.

    Parameters
    ----------
    raw : str
        Raw text from the model response.

    Returns
    -------
    dict
        Parsed JSON object.
    """
    # Strip markdown code fences if present (```json ... ```)
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try extracting the first {...} block
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        logger.error("Could not parse judge response as JSON: %r", raw[:200])
        return {}


# ── Main scoring function ─────────────────────────────────────────────────────

def score(
    query:    str,
    context:  str,
    response: str,
    model:    Optional[str] = None,
) -> dict:
    """
    Score a RAG response on faithfulness, relevance, and context recall.

    Makes a single Anthropic API call and returns all three scores plus
    per-dimension rationale in one dict.

    Parameters
    ----------
    query : str
        The original user question sent to the RAG pipeline.
    context : str
        The retrieved context chunks, separated by ``---`` delimiters.
    response : str
        The RAG pipeline's answer to the query.
    model : str | None
        Override the judge model from ``config.JUDGE_MODEL``.

    Returns
    -------
    dict
        Keys:
        - ``faithfulness``   (float 0–1)
        - ``relevance``      (float 0–1)
        - ``context_recall`` (float 0–1)
        - ``rationale``      (dict with keys: faithfulness, relevance, context_recall)

        On API failure returns a zeroed-out dict with an ``error`` key.
    """
    payload = json.dumps(
        {"query": query, "context": context, "response": response},
        ensure_ascii=False,
    )

    try:
        resp = _get_client().messages.create(
            model      = model or JUDGE_MODEL,
            max_tokens = 1024,
            temperature= 0,
            system     = _SYSTEM_PROMPT,
            messages   = [{"role": "user", "content": payload}],
        )
        raw = resp.content[0].text
    except Exception as exc:
        logger.error("Judge API call failed: %s", exc)
        return {
            "faithfulness":   0.0,
            "relevance":      0.0,
            "context_recall": 0.0,
            "rationale": {
                "faithfulness":   "",
                "relevance":      "",
                "context_recall": "",
            },
            "error": str(exc),
        }

    parsed = _parse_json(raw)

    if not parsed:
        logger.warning(
            "Judge returned empty or unparseable JSON.\nRaw response: %s",
            raw[:500]
        )

    # Clamp scores to [0, 1] and provide defaults
    def _clamp(key: str) -> float:
        try:
            return max(0.0, min(1.0, float(parsed.get(key, 0.0))))
        except (TypeError, ValueError):
            return 0.0

    rationale = parsed.get("rationale", {})
    if not isinstance(rationale, dict):
        rationale = {}

    return {
        "faithfulness":   _clamp("faithfulness"),
        "relevance":      _clamp("relevance"),
        "context_recall": _clamp("context_recall"),
        "rationale": {
            "faithfulness":   str(rationale.get("faithfulness",   "")),
            "relevance":      str(rationale.get("relevance",      "")),
            "context_recall": str(rationale.get("context_recall", "")),
        },
    }


def format_context(chunks: list[str]) -> str:
    """
    Format a list of retrieved chunks into the context string expected by score().

    Parameters
    ----------
    chunks : list[str]
        Retrieved document chunks.

    Returns
    -------
    str
        Chunks joined by ``\\n\\n---\\n\\n`` delimiters.
    """
    return "\n\n---\n\n".join(
        f"[Chunk {i + 1}]\n{chunk}" for i, chunk in enumerate(chunks)
    )
