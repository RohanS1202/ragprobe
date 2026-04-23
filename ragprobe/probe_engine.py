"""
probe_engine.py — Adversarial probe runner for ragprobe.

Loads the prompt suite, runs each prompt through the RAG pipeline,
scores it with the judge, classifies it with the safety classifier,
and writes every result to SQLite immediately (no batching).

Usage::

    from ragprobe.probe_engine import run_session

    session_id = run_session(
        mode     = "baseline",
        category = None,     # None = all categories
        limit    = None,     # None = all prompts
    )
    print(f"Session: {session_id}")

CLI::

    ragprobe run --mode baseline --category hallucination_bait --limit 10
"""

from __future__ import annotations

import datetime
import logging
import uuid
from pathlib import Path
from typing import Callable, Optional

from ragprobe.config import DB_PATH, FAISS_PATH
from ragprobe.db import init_db, insert_result, insert_session, update_session_stats
from ragprobe.judge import format_context, score as judge_score
from ragprobe.prompts import VALID_CATEGORIES, get_by_category, load_prompts
from ragprobe.safety_classifier import classify as safety_classify

logger = logging.getLogger(__name__)

# Pipeline callable type: (query: str) -> dict with keys
# response, context_chunks, latency_ms (and optionally mode).
PipelineFn = Callable[[str], dict]


def _default_pipeline(query: str, mode: str, faiss_path: Path) -> dict:
    """Wrap rag_pipeline.run() as the default callable."""
    from ragprobe.rag_pipeline import run as pipeline_run
    return pipeline_run(query=query, mode=mode, faiss_path=faiss_path)


def run_session(
    mode:        str = "baseline",
    category:    Optional[str] = None,
    limit:       Optional[int] = None,
    db_path:     Path = DB_PATH,
    faiss_path:  Path = FAISS_PATH,
    use_llm_safety: bool = True,
    pipeline_fn: Optional[PipelineFn] = None,
) -> str:
    """
    Run a full adversarial probe session and persist results to SQLite.

    Parameters
    ----------
    mode : str
        ``"baseline"`` or ``"hardened"``.
    category : str | None
        Restrict to one prompt category.  None = run all five categories.
    limit : int | None
        Stop after this many probes.  None = run all prompts.
    db_path : Path
        SQLite database file.
    faiss_path : Path
        FAISS index directory.
    use_llm_safety : bool
        Whether to call the LLM for ambiguous safety classification cases.
    pipeline_fn : callable | None
        Optional custom pipeline.  Signature: ``(query: str) -> dict`` where
        the dict must contain ``response`` (str), ``context_chunks`` (list[str]),
        and ``latency_ms`` (int).  When None, uses ``rag_pipeline.run()``.

    Returns
    -------
    str
        The session_id for this run (UUID string).
    """
    if mode not in ("baseline", "hardened"):
        raise ValueError(f"mode must be 'baseline' or 'hardened', got: {mode!r}")
    if category is not None and category not in VALID_CATEGORIES:
        raise ValueError(
            f"Unknown category {category!r}. Valid: {sorted(VALID_CATEGORIES)}"
        )

    # ── Initialise DB ─────────────────────────────────────────────────────────
    init_db(db_path)

    session_id    = str(uuid.uuid4())
    run_timestamp = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    # Record session upfront (stats updated at end)
    insert_session(
        session_id    = session_id,
        run_timestamp = run_timestamp,
        pipeline_mode = mode,
        db_path       = db_path,
    )

    # ── Load prompts ──────────────────────────────────────────────────────────
    if category is not None:
        prompts = get_by_category(category)
    else:
        prompts = load_prompts()

    if limit is not None:
        prompts = prompts[:limit]

    total = len(prompts)
    logger.info(
        "Session %s | mode=%s | category=%s | %d prompts",
        session_id[:8], mode, category or "all", total,
    )

    # ── Rich progress display ─────────────────────────────────────────────────
    try:
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
        from rich.console import Console
        _console = Console()
        _use_rich = True
    except ImportError:
        _use_rich = False

    def _log_progress(idx: int, cat: str, faith: float, rel: float, ctx: float) -> None:
        msg = (
            f"[{idx:>3}/{total}] {cat:<22} "
            f"faith={faith:.2f}  rel={rel:.2f}  ctx={ctx:.2f}"
        )
        if _use_rich:
            _console.print(msg)
        else:
            print(msg)

    # ── Probe loop ────────────────────────────────────────────────────────────
    if _use_rich:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
        )
        task = progress.add_task(f"Probing ({mode})", total=total)
        progress.start()
    else:
        progress = None
        task     = None

    try:
        for i, prompt_item in enumerate(prompts, 1):
            prompt_text = prompt_item.get("prompt_text", "")
            prompt_cat  = prompt_item.get("category",    "unknown")

            # ── Pipeline ──────────────────────────────────────────────────────
            try:
                if pipeline_fn is not None:
                    result = pipeline_fn(prompt_text)
                else:
                    result = _default_pipeline(prompt_text, mode, faiss_path)
                response_text  = result["response"]
                context_chunks = result["context_chunks"]
                latency_ms     = result.get("latency_ms", 0)
            except Exception as exc:
                logger.error("Pipeline error on prompt %d: %s", i, exc)
                response_text  = f"[Pipeline error: {exc}]"
                context_chunks = []
                latency_ms     = 0

            # ── Judge ─────────────────────────────────────────────────────────
            context_str = format_context(context_chunks)
            try:
                scores = judge_score(
                    query    = prompt_text,
                    context  = context_str,
                    response = response_text,
                )
            except Exception as exc:
                logger.error("Judge error on prompt %d: %s", i, exc)
                scores = {
                    "faithfulness":   0.0,
                    "relevance":      0.0,
                    "context_recall": 0.0,
                    "rationale":      {"faithfulness": "", "relevance": "", "context_recall": ""},
                }

            faith = scores["faithfulness"]
            rel   = scores["relevance"]
            ctx   = scores["context_recall"]

            # ── Safety classifier ─────────────────────────────────────────────
            try:
                flags = safety_classify(
                    query    = prompt_text,
                    response = response_text,
                    context  = context_chunks,
                    use_llm  = use_llm_safety,
                )
            except Exception as exc:
                logger.error("Safety classifier error on prompt %d: %s", i, exc)
                flags = {
                    "injection_compliance":      False,
                    "confidentiality_violation": False,
                    "refusal_evasion":           False,
                }

            # ── Persist ───────────────────────────────────────────────────────
            created_at = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
            insert_result(
                session_id                = session_id,
                prompt_category           = prompt_cat,
                prompt_text               = prompt_text,
                response_text             = response_text,
                context_chunks            = context_chunks,
                faithfulness              = faith,
                relevance                 = rel,
                context_recall            = ctx,
                judge_rationale           = scores.get("rationale"),
                injection_compliance      = flags["injection_compliance"],
                confidentiality_violation = flags["confidentiality_violation"],
                refusal_evasion           = flags["refusal_evasion"],
                latency_ms                = latency_ms,
                created_at                = created_at,
                db_path                   = db_path,
            )

            _log_progress(i, prompt_cat, faith, rel, ctx)

            if _use_rich and progress is not None:
                progress.update(task, advance=1)

    finally:
        if _use_rich and progress is not None:
            progress.stop()

    # ── Update session aggregate stats ────────────────────────────────────────
    update_session_stats(session_id, db_path)

    logger.info("Session %s complete.", session_id[:8])
    return session_id
