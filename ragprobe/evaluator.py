"""
evaluator.py — Integration core for ragprobe.

Wires SafetyGate, RetrievalDiagnostic, and SessionReport into a single
Evaluator class that the CLI calls and end users can use directly from Python.

Usage::

    from ragprobe import Evaluator, SafetyGate, RetrievalDiagnostic

    gate       = SafetyGate.default(budget_usd=1.0)
    diagnostic = RetrievalDiagnostic()
    evaluator  = Evaluator(gate=gate, diagnostic=diagnostic, model="claude-haiku-4-5-20251001")

    report = evaluator.run(
        queries   = ["What was Apple revenue in FY2024?"],
        corpus    = [...],                 # raw document strings
        references= ["Apple revenue was $391B in FY2024."],
    )
    report.save("results.json")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from ragprobe._utils import _STOPWORDS, tokenize as _tokenize
from ragprobe.safety import SafetyGate
from ragprobe.retrieval import RetrievalDiagnostic
from ragprobe.reporter import SessionReport
from ragprobe.core.cost_guard import BudgetExceededError
from ragprobe.core.rate_limiter import RateLimitExceededError
from ragprobe.core.injection_guard import InjectionDetectedError

logger = logging.getLogger(__name__)

# ── Exceptions caught as safety events (never allowed to crash a run) ─────────
_SAFETY_EXCEPTIONS = (BudgetExceededError, RateLimitExceededError, InjectionDetectedError)


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluator
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Evaluator:
    """
    Integration core: wires SafetyGate, RetrievalDiagnostic, and SessionReport.

    Parameters
    ----------
    gate : SafetyGate
        Safety gate used for corpus scanning and query validation.
    diagnostic : RetrievalDiagnostic
        Diagnostic engine used for recall, redundancy, and coverage scoring.
    model : str
        LLM model name recorded in the output SessionReport and used for cost
        estimation.  Default: ``"claude-haiku-4-5-20251001"``.

    Usage::

        evaluator = Evaluator(
            gate       = SafetyGate.default(budget_usd=2.0),
            diagnostic = RetrievalDiagnostic(),
        )
        report = evaluator.run(queries, corpus, references)
    """

    gate:       SafetyGate
    diagnostic: RetrievalDiagnostic
    model:      str = "claude-haiku-4-5-20251001"

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        queries:         list[str],
        corpus:          list[str],
        references:      Optional[list[str]] = None,
        *,
        semantic_recall: bool = False,
        client:          Any  = None,
    ) -> SessionReport:
        """
        Run the full ragprobe evaluation pipeline.

        Execution order
        ---------------
        1. ``gate.check_corpus(corpus)``  — validate and scan the full corpus.
        2. For each query:
           a. ``gate.check_query(query)`` — validate + injection scan.
           b. ``top_k_retrieve(query, clean_corpus)`` — lexical top-5 retrieval.
           c. ``gate.record_usage(prompt_tokens=approx)`` — track cost.
        3. ``diagnostic.run(all_queries, all_chunks, references)`` — score
           retrieval across the full query set and produce findings.
        4. Pack everything into a ``SessionReport`` and return.

        Any ``BudgetExceededError``, ``RateLimitExceededError``, or
        ``InjectionDetectedError`` raised during per-query processing is logged
        and appended to ``safety_events`` — it never crashes the run.

        Parameters
        ----------
        queries : list[str]
            User queries to evaluate.
        corpus : list[str]
            Raw document strings forming the retrieval corpus.
        references : list[str] | None
            Ground-truth reference answers (one per query).  When ``None``,
            recall scores default to ``0.0`` in the returned report.
        semantic_recall : bool
            When ``True``, compute semantic recall using embedding cosine
            similarity.  Requires ``client``.
        client : Any
            OpenAI-compatible embedding client.  Only needed when
            ``semantic_recall=True``.

        Returns
        -------
        SessionReport
        """
        safety_events: list[dict] = []

        # ── 1. Corpus safety gate ─────────────────────────────────────────────
        try:
            clean_corpus, corpus_report = self.gate.check_corpus(corpus)
        except Exception as exc:
            logger.error("corpus check failed: %s", exc)
            clean_corpus  = list(corpus)
            corpus_report = {
                "original_count": len(corpus), "clean_count": len(corpus),
                "validation_errors": [], "injection_flagged": 0,
                "injection_blocked": 0, "injection_risk": "SAFE",
            }

        if corpus_report["injection_flagged"] > 0:
            safety_events.append({
                "type":              "corpus_scan",
                "document_index":    "",
                "pattern":           "injection_pattern",
                "risk_level":        corpus_report["injection_risk"],
                "message": (
                    f"{corpus_report['injection_flagged']} document(s) contained "
                    "injection patterns; "
                    f"{corpus_report['injection_blocked']} blocked before evaluation."
                ),
            })

        # ── 2. Per-query retrieval ────────────────────────────────────────────
        retrieved_per_query: list[list[str]] = []

        for i, query in enumerate(queries):
            try:
                q_report = self.gate.check_query(query)
                if not q_report["is_valid"]:
                    safety_events.append({
                        "type":           "query_validation",
                        "document_index": i,
                        "pattern":        ", ".join(q_report.get("injection_matches", [])),
                        "risk_level":     q_report["injection_risk"],
                        "message":        "; ".join(q_report["validation_errors"]),
                    })

                chunks = self.top_k_retrieve(query, clean_corpus)
                retrieved_per_query.append(chunks)

                # Approximate token usage: query * 4 + chunk words
                approx_tokens = (
                    len(query.split()) * 4
                    + sum(len(c.split()) for c in chunks)
                )
                self.gate.record_usage(
                    prompt_tokens=approx_tokens,
                    model=self.model,
                )

            except _SAFETY_EXCEPTIONS as exc:
                logger.warning("Safety event at query %d: %s", i, exc)
                safety_events.append({
                    "type":           type(exc).__name__,
                    "document_index": i,
                    "pattern":        "",
                    "risk_level":     "HIGH",
                    "message":        str(exc),
                })
                retrieved_per_query.append([])

        # ── 3. Full diagnostic ────────────────────────────────────────────────
        try:
            diag = self.diagnostic.run(
                queries,
                retrieved_per_query,
                references,
                semantic_recall=semantic_recall,
                client=client,
            )
        except Exception as exc:
            logger.error("diagnostic run failed: %s", exc)
            diag = {
                "per_query": [
                    {"query": q, "chunk_count": len(retrieved_per_query[i])}
                    for i, q in enumerate(queries)
                ],
                "aggregate": {},
                "findings":  [],
            }

        per_query = diag["per_query"]
        aggregate = diag["aggregate"]
        findings  = diag["findings"]

        # Ensure mean_recall is always present (defaults 0.0 when no references)
        aggregate.setdefault("mean_recall", 0.0)
        for pq in per_query:
            pq.setdefault("recall", 0.0)

        # ── 4. Pack into SessionReport ────────────────────────────────────────
        cost_sum = self.gate.cost_summary()

        return SessionReport.new(
            model              = self.model,
            total_queries      = len(queries),
            cost_summary       = cost_sum,
            retrieval_findings = findings,
            per_query          = per_query,
            aggregate          = aggregate,
            safety_events      = safety_events,
        )

    # ── Default retriever ─────────────────────────────────────────────────────

    @staticmethod
    def top_k_retrieve(query: str, corpus: list[str], k: int = 5) -> list[str]:
        """
        Lexical top-k retrieval using token overlap scoring.

        Each corpus chunk is scored by the fraction of significant query tokens
        it contains.  Chunks are returned in descending score order.

        This is the default retriever.  Users can swap in their own by
        subclassing ``Evaluator`` and overriding this method.

        Parameters
        ----------
        query : str
            The user query.
        corpus : list[str]
            Document strings to search.
        k : int
            Maximum number of chunks to return.  Default: 5.

        Returns
        -------
        list[str]
            Up to *k* chunks most relevant to *query*.
        """
        if not corpus:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return list(corpus[:k])

        scored: list[tuple[float, str]] = []
        for chunk in corpus:
            chunk_tokens = _tokenize(chunk)
            score = len(query_tokens & chunk_tokens) / len(query_tokens)
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:k]]
