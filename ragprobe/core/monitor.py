"""
monitor.py — Production monitoring decorator for RAG pipelines.

Usage:
    from ragprobe import monitor

    @monitor(project="my-rag-app", alert_threshold=0.75)
    def my_rag_pipeline(query: str) -> tuple[str, list]:
        answer   = llm.generate(query)
        contexts = retriever.get(query)
        return answer, contexts

Every call is logged to local SQLite. Scoring is async and non-blocking.
"""

from __future__ import annotations

import functools
import json
import sqlite3
import uuid
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib  import Path
from typing   import Callable, Optional


_DB_PATH = Path.home() / ".ragprobe" / "monitor.db"

# Module-level thread pool — survives program exit, no lost scores
_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ragprobe-scorer")


def _init_db(db_path: Path = _DB_PATH) -> None:
    """Create local SQLite DB if it doesn't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")  # prevents "database is locked" errors
    conn.execute("""
        CREATE TABLE IF NOT EXISTS traces (
            id          TEXT PRIMARY KEY,
            project     TEXT NOT NULL,
            timestamp   TEXT NOT NULL,
            query       TEXT NOT NULL,
            answer      TEXT NOT NULL,
            contexts    TEXT NOT NULL,
            latency_ms  REAL,
            faith_score REAL,
            rel_score   REAL,
            ctx_score   REAL,
            overall     REAL,
            alerted     INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


def _log_trace(
    project:    str,
    query:      str,
    answer:     str,
    contexts:   list[str],
    latency_ms: float,
    db_path:    Path = _DB_PATH,
) -> str:
    """Write a trace to local SQLite. Returns the trace ID."""
    _init_db(db_path)
    trace_id = str(uuid.uuid4())
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """INSERT INTO traces
           (id, project, timestamp, query, answer, contexts, latency_ms)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            trace_id,
            project,
            datetime.utcnow().isoformat(),
            query,
            answer,
            json.dumps(contexts),
            latency_ms,
        )
    )
    conn.commit()
    conn.close()
    return trace_id


def _update_trace_scores(trace_id: str, scores: dict) -> None:
    try:
        conn = sqlite3.connect(str(_DB_PATH))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """UPDATE traces SET faith_score=?, rel_score=?, ctx_score=?, overall=?
               WHERE id=?""",
            (
                scores.get("faithfulness"),
                scores.get("relevance"),
                scores.get("context_recall"),
                scores.get("overall"),
                trace_id,
            )
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def get_traces(project: str, limit: int = 50) -> list[dict]:
    """Retrieve recent traces for a project (used by the dashboard)."""
    _init_db()
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM traces WHERE project = ? ORDER BY timestamp DESC LIMIT ?",
        (project, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def monitor(
    project:         str   = "default",
    alert_threshold: float = 0.75,
    judge:           str   = "openai",
    metrics:         Optional[list[str]] = None,
    verbose:         bool  = False,
):
    """
    Decorator that monitors a RAG pipeline in production.

    Decorated function must return (answer: str, contexts: list[str]).
    Scoring is async — pipeline latency is completely unaffected.

    Args:
        project:         Name for grouping traces in the dashboard
        alert_threshold: Alert if overall score drops below this
        judge:           LLM to use for scoring ("openai" | "anthropic")
        metrics:         Which metrics to score
        verbose:         Print score after each call

    Example:
        @monitor(project="support-bot", alert_threshold=0.80)
        def answer(query: str) -> tuple[str, list[str]]: ...
    """
    if metrics is None:
        metrics = ["faithfulness", "relevance"]

    # Instantiate Scorer once at decoration time — not on every request
    from ragprobe.core.scorer import Scorer
    _scorer = Scorer(judge=judge)

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start      = time.perf_counter()
            result     = fn(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000

            if isinstance(result, tuple) and len(result) == 2:
                answer, contexts = result
            else:
                answer, contexts = str(result), []

            query    = kwargs.get("query") or (args[0] if args else "unknown")
            trace_id = _log_trace(
                project    = project,
                query      = str(query),
                answer     = answer,
                contexts   = contexts,
                latency_ms = elapsed_ms,
            )

            def score_async():
                try:
                    from ragprobe.core.generator import TestCase, AttackType
                    mock_tc = TestCase(
                        id                = trace_id,
                        question          = str(query),
                        attack_type       = AttackType.EDGE_CASE,
                        source_chunk      = contexts[0][:200] if contexts else "",
                        expected_behavior = "Answer accurately based on context",
                    )
                    eval_result = _scorer.score(mock_tc, answer, contexts, metrics)
                    scores = {
                        "faithfulness":   eval_result.faithfulness.score   if eval_result.faithfulness   else None,
                        "relevance":      eval_result.relevance.score      if eval_result.relevance      else None,
                        "context_recall": eval_result.context_recall.score if eval_result.context_recall else None,
                        "overall":        eval_result.overall_score,
                    }
                    _update_trace_scores(trace_id, scores)

                    if verbose:
                        print(
                            f"[ragprobe] {project} | "
                            f"score={eval_result.overall_score:.2f} | "
                            f"latency={elapsed_ms:.0f}ms"
                        )

                    if eval_result.overall_score < alert_threshold:
                        print(
                            f"\n[ragprobe ALERT] ⚠  Quality below threshold in '{project}'\n"
                            f"  Score:  {eval_result.overall_score:.2f}\n"
                            f"  Query:  {str(query)[:80]}\n"
                        )
                except Exception as e:
                    if verbose:
                        print(f"[ragprobe] Scoring error (non-blocking): {e}")

            _pool.submit(score_async)
            return result

        return wrapper
    return decorator
