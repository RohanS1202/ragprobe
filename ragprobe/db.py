"""
db.py — SQLite persistence layer for ragprobe probe runs.

Schema
------
  sessions      — one row per probe run session (summary stats).
  probe_results — one row per individual probe (full detail).

All writes use parameterized queries — no string interpolation.

Usage::

    from ragprobe.db import init_db, insert_session, insert_result, \
                           get_session_summary, get_all_results

    init_db()
    insert_session(session_id="abc", run_timestamp="...", pipeline_mode="baseline",
                   total_probes=50, mean_faithfulness=0.82, mean_relevance=0.87,
                   mean_context_recall=0.74)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Optional

from ragprobe.config import DB_PATH

logger = logging.getLogger(__name__)


# ── Connection factory ────────────────────────────────────────────────────────

def _connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Open a connection with WAL mode and row_factory enabled."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


# ── Schema bootstrap ──────────────────────────────────────────────────────────

def init_db(db_path: Path = DB_PATH) -> None:
    """
    Create all tables if they do not already exist.

    Safe to call on every startup — uses ``CREATE TABLE IF NOT EXISTS``.

    Parameters
    ----------
    db_path : Path
        SQLite file path.  Defaults to ``config.DB_PATH``.
    """
    conn = _connect(db_path)
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id         TEXT PRIMARY KEY,
                run_timestamp      TEXT NOT NULL,
                pipeline_mode      TEXT NOT NULL,
                total_probes       INTEGER NOT NULL DEFAULT 0,
                mean_faithfulness  REAL,
                mean_relevance     REAL,
                mean_context_recall REAL
            );

            CREATE TABLE IF NOT EXISTS probe_results (
                id                        INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id                TEXT NOT NULL,
                prompt_category           TEXT,
                prompt_text               TEXT,
                response_text             TEXT,
                context_chunks            TEXT,
                faithfulness              REAL,
                relevance                 REAL,
                context_recall            REAL,
                judge_rationale           TEXT,
                injection_compliance      INTEGER DEFAULT 0,
                confidentiality_violation INTEGER DEFAULT 0,
                refusal_evasion           INTEGER DEFAULT 0,
                latency_ms                INTEGER,
                created_at                TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );
        """)
        conn.commit()
        logger.debug("init_db complete at %s", db_path)
    finally:
        conn.close()


# ── Write helpers ─────────────────────────────────────────────────────────────

def insert_session(
    session_id:          str,
    run_timestamp:       str,
    pipeline_mode:       str,
    total_probes:        int = 0,
    mean_faithfulness:   Optional[float] = None,
    mean_relevance:      Optional[float] = None,
    mean_context_recall: Optional[float] = None,
    db_path:             Path = DB_PATH,
) -> None:
    """
    Insert or replace a session row.

    Parameters
    ----------
    session_id : str
        Unique identifier for this run.
    run_timestamp : str
        ISO-format timestamp when the run started.
    pipeline_mode : str
        ``"baseline"`` or ``"hardened"``.
    total_probes : int
        Number of probes attempted.
    mean_faithfulness, mean_relevance, mean_context_recall : float | None
        Aggregate mean scores across all probes.  Pass ``None`` when the
        session is still in progress and final stats are not yet known.
    db_path : Path
        SQLite file path.
    """
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO sessions
                (session_id, run_timestamp, pipeline_mode, total_probes,
                 mean_faithfulness, mean_relevance, mean_context_recall)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, run_timestamp, pipeline_mode, total_probes,
             mean_faithfulness, mean_relevance, mean_context_recall),
        )
        conn.commit()
    finally:
        conn.close()


def insert_result(
    session_id:                str,
    prompt_category:           str,
    prompt_text:               str,
    response_text:             str,
    context_chunks:            list[str],
    faithfulness:              Optional[float],
    relevance:                 Optional[float],
    context_recall:            Optional[float],
    judge_rationale:           Optional[dict],
    injection_compliance:      bool,
    confidentiality_violation: bool,
    refusal_evasion:           bool,
    latency_ms:                int,
    created_at:                str,
    db_path:                   Path = DB_PATH,
) -> int:
    """
    Insert a single probe result row.

    Parameters
    ----------
    session_id : str
        Parent session identifier.
    prompt_category : str
        One of the five adversarial categories.
    prompt_text : str
        The adversarial prompt sent to the pipeline.
    response_text : str
        The pipeline's response.
    context_chunks : list[str]
        Retrieved chunks passed to the pipeline (stored as a JSON array).
    faithfulness, relevance, context_recall : float | None
        Judge scores.  None if the judge call failed.
    judge_rationale : dict | None
        Rationale dict from the judge.  Stored as JSON.
    injection_compliance : bool
        True if the response followed an injected instruction.
    confidentiality_violation : bool
        True if the response leaked non-public or PII-like data.
    refusal_evasion : bool
        True if the model answered an out-of-scope question it should refuse.
    latency_ms : int
        End-to-end pipeline latency in milliseconds.
    created_at : str
        ISO-format timestamp.
    db_path : Path
        SQLite file path.

    Returns
    -------
    int
        The ``AUTOINCREMENT`` id of the inserted row.
    """
    conn = _connect(db_path)
    try:
        cursor = conn.execute(
            """
            INSERT INTO probe_results
                (session_id, prompt_category, prompt_text, response_text,
                 context_chunks, faithfulness, relevance, context_recall,
                 judge_rationale, injection_compliance, confidentiality_violation,
                 refusal_evasion, latency_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                prompt_category,
                prompt_text,
                response_text,
                json.dumps(context_chunks),
                faithfulness,
                relevance,
                context_recall,
                json.dumps(judge_rationale) if judge_rationale else None,
                int(injection_compliance),
                int(confidentiality_violation),
                int(refusal_evasion),
                latency_ms,
                created_at,
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


# ── Read helpers ──────────────────────────────────────────────────────────────

def get_session_summary(session_id: str, db_path: Path = DB_PATH) -> Optional[dict]:
    """
    Return a session row as a dict, or None if not found.

    Parameters
    ----------
    session_id : str
    db_path : Path

    Returns
    -------
    dict | None
    """
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_all_results(session_id: str, db_path: Path = DB_PATH) -> list[dict]:
    """
    Return all probe_results rows for a session as a list of dicts.

    ``context_chunks`` and ``judge_rationale`` are deserialized from JSON.

    Parameters
    ----------
    session_id : str
    db_path : Path

    Returns
    -------
    list[dict]
    """
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM probe_results WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            try:
                d["context_chunks"] = json.loads(d["context_chunks"]) if d["context_chunks"] else []
            except (json.JSONDecodeError, TypeError):
                d["context_chunks"] = []
            try:
                d["judge_rationale"] = json.loads(d["judge_rationale"]) if d["judge_rationale"] else {}
            except (json.JSONDecodeError, TypeError):
                d["judge_rationale"] = {}
            results.append(d)
        return results
    finally:
        conn.close()


def list_sessions(db_path: Path = DB_PATH) -> list[dict]:
    """
    Return all session rows ordered by most recent first.

    Parameters
    ----------
    db_path : Path

    Returns
    -------
    list[dict]
    """
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY run_timestamp DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def update_session_stats(session_id: str, db_path: Path = DB_PATH) -> None:
    """
    Recompute and persist aggregate stats for a session from its probe_results.

    Call this after all probes for a session have been inserted.

    Parameters
    ----------
    session_id : str
    db_path : Path
    """
    conn = _connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT
                COUNT(*)                  AS total_probes,
                AVG(faithfulness)         AS mean_faithfulness,
                AVG(relevance)            AS mean_relevance,
                AVG(context_recall)       AS mean_context_recall
            FROM probe_results
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        if row:
            conn.execute(
                """
                UPDATE sessions
                SET total_probes       = ?,
                    mean_faithfulness  = ?,
                    mean_relevance     = ?,
                    mean_context_recall = ?
                WHERE session_id = ?
                """,
                (
                    row["total_probes"],
                    row["mean_faithfulness"],
                    row["mean_relevance"],
                    row["mean_context_recall"],
                    session_id,
                ),
            )
            conn.commit()
    finally:
        conn.close()
