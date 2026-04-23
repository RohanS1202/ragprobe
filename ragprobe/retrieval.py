"""
retrieval.py — Retrieval diagnostics for RAG pipelines.

Three classes:

  ContextRecallScorer — Measures how much of a reference answer is present
                        in retrieved chunks.  Two modes:
                          token_recall    — lexical overlap, stdlib only.
                          semantic_recall — embedding cosine similarity via
                                           any OpenAI-compatible client.

  ChunkAnalyzer       — Structural analysis of retrieved chunks:
                          redundancy_score    — pairwise Jaccard overlap.
                          length_distribution — word-count statistics.
                          coverage_gaps       — query terms absent from chunks.

  RetrievalDiagnostic — Orchestrator; runs both components across a full eval
                        set and returns a ranked findings dict with severity
                        levels (CRITICAL / WARNING / INFO) and actionable
                        recommendations.
"""

from __future__ import annotations

import logging
import math
import statistics as _stats
from typing import Any, Optional

from ragprobe._utils import _STOPWORDS, tokenize_ordered as _sig_tokens

logger = logging.getLogger(__name__)

# ── Severity thresholds ───────────────────────────────────────────────────────
_RECALL_CRITICAL     = 0.3
_RECALL_WARNING      = 0.5
_REDUNDANCY_CRITICAL = 0.6
_COVERAGE_WARNING    = 0.5   # gap fraction, i.e. 1 - coverage_pct

# ── Chunk length thresholds (word count) ─────────────────────────────────────
_SHORT_CHUNK_WORDS = 80
_LONG_CHUNK_WORDS  = 600


# ── Private helpers ───────────────────────────────────────────────────────────

def _sig_token_set(text: str) -> set[str]:
    return set(_sig_tokens(text))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _cosine(a: list[float], b: list[float]) -> float:
    dot    = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ═══════════════════════════════════════════════════════════════════════════════
# ContextRecallScorer
# ═══════════════════════════════════════════════════════════════════════════════

class ContextRecallScorer:
    """
    Measures how much of a reference answer is present in retrieved chunks.

    Stopwords and tokens shorter than 2 characters are filtered before any
    comparison so function words do not inflate scores.
    """

    def token_recall(self, reference: str, chunks: list[str]) -> float:
        """
        Compute lexical recall: fraction of reference significant tokens
        found in the union of all retrieved chunk tokens.

        No API calls required.

        Parameters
        ----------
        reference : str
            The expected / ground-truth answer text.
        chunks : list[str]
            Retrieved document chunks to score against.

        Returns
        -------
        float
            Value in [0.0, 1.0].  Returns 1.0 when the reference contains no
            significant tokens (nothing to miss).  Returns 0.0 when chunks is
            empty and reference has content.
        """
        if not chunks:
            # nothing retrieved: no recall unless reference is also empty
            return 1.0 if not reference.strip() else 0.0

        ref_tokens = _sig_token_set(reference)
        if not ref_tokens:
            return 1.0

        combined: set[str] = set()
        for chunk in chunks:
            combined |= _sig_token_set(chunk)

        matched = ref_tokens & combined
        score   = len(matched) / len(ref_tokens)
        logger.debug(
            "token_recall: %d/%d reference tokens matched → %.3f",
            len(matched), len(ref_tokens), score,
        )
        return score

    def semantic_recall(
        self,
        reference: str,
        chunks:    list[str],
        client:    Any,
        model:     str,
    ) -> float:
        """
        Compute semantic recall as the maximum cosine similarity between the
        reference embedding and any retrieved chunk embedding.

        Parameters
        ----------
        reference : str
            The expected / ground-truth answer text.
        chunks : list[str]
            Retrieved document chunks to score against.
        client : Any
            An OpenAI-compatible client exposing ``client.embeddings.create``.
        model : str
            Embedding model name, e.g. ``"text-embedding-3-small"``.

        Returns
        -------
        float
            Maximum cosine similarity in [0.0, 1.0].  Returns 0.0 immediately
            when chunks is empty (no API call made).
        """
        if not chunks:
            logger.debug("semantic_recall: no chunks provided, returning 0.0")
            return 0.0

        texts    = [reference] + chunks
        response = client.embeddings.create(model=model, input=texts)
        embs     = [item.embedding for item in response.data]

        ref_emb  = embs[0]
        sims     = [_cosine(ref_emb, chunk_emb) for chunk_emb in embs[1:]]
        best     = max(sims)
        logger.debug(
            "semantic_recall: best cosine similarity = %.3f across %d chunks",
            best, len(chunks),
        )
        return best


# ═══════════════════════════════════════════════════════════════════════════════
# ChunkAnalyzer
# ═══════════════════════════════════════════════════════════════════════════════

class ChunkAnalyzer:
    """
    Structural analysis of retrieved document chunks.

    Identifies three categories of problems:
      - Redundancy   — near-duplicate chunks wasting context window space.
      - Length       — chunks that are too short or too long.
      - Coverage     — query terms absent from all retrieved chunks.
    """

    def redundancy_score(self, chunks: list[str]) -> dict:
        """
        Compute pairwise Jaccard token overlap between every pair of chunks.

        Parameters
        ----------
        chunks : list[str]
            Retrieved document chunks to analyse.

        Returns
        -------
        dict
            Keys:
              ``mean``       – mean Jaccard similarity across all pairs (float).
              ``max``        – maximum pairwise Jaccard similarity (float).
              ``worst_pair`` – dict ``{index_a, index_b, score}`` for the most
                               redundant pair, or ``None`` if fewer than two
                               chunks are provided.
        """
        if len(chunks) < 2:
            return {"mean": 0.0, "max": 0.0, "worst_pair": None}

        token_sets = [_sig_token_set(c) for c in chunks]
        pairs: list[tuple[int, int, float]] = []

        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                pairs.append((i, j, _jaccard(token_sets[i], token_sets[j])))

        scores = [p[2] for p in pairs]
        worst  = max(pairs, key=lambda p: p[2])

        return {
            "mean":       round(_stats.mean(scores), 4),
            "max":        round(max(scores), 4),
            "worst_pair": {
                "index_a": worst[0],
                "index_b": worst[1],
                "score":   round(worst[2], 4),
            },
        }

    def length_distribution(self, chunks: list[str]) -> dict:
        """
        Compute word-count statistics across all chunks and flag anomalies.

        Parameters
        ----------
        chunks : list[str]
            Document chunks to measure.  Word count uses whitespace splitting.

        Returns
        -------
        dict
            Keys:
              ``min``            – minimum word count (int).
              ``max``            – maximum word count (int).
              ``mean``           – mean word count (float, rounded to 2 dp).
              ``median``         – median word count (float).
              ``std``            – sample standard deviation (float, 0.0 for
                                   single-chunk inputs).
              ``recommendation`` – actionable string when chunks are too short
                                   (mean < 80 words) or too long (mean > 600),
                                   otherwise ``None``.
        """
        if not chunks:
            return {
                "min": 0, "max": 0, "mean": 0.0,
                "median": 0.0, "std": 0.0, "recommendation": None,
            }

        counts   = [len(c.split()) for c in chunks]
        mean_c   = _stats.mean(counts)
        median_c = float(_stats.median(counts))
        std_c    = _stats.stdev(counts) if len(counts) >= 2 else 0.0

        recommendation: Optional[str] = None
        if mean_c < _SHORT_CHUNK_WORDS:
            target = max(int(mean_c * 2), _SHORT_CHUNK_WORDS + 20)
            recommendation = (
                f"Chunks are too short (mean {mean_c:.0f} words; "
                f"recommended minimum {_SHORT_CHUNK_WORDS}). "
                f"Increase target chunk size to ~{target} words to reduce "
                f"the risk of splitting answers across chunk boundaries."
            )
        elif mean_c > _LONG_CHUNK_WORDS:
            recommendation = (
                f"Chunks are too long (mean {mean_c:.0f} words; "
                f"recommended maximum {_LONG_CHUNK_WORDS}). "
                f"Reduce chunk size to improve retrieval precision and "
                f"reduce context-window dilution."
            )

        return {
            "min":            min(counts),
            "max":            max(counts),
            "mean":           round(mean_c, 2),
            "median":         median_c,
            "std":            round(std_c, 2),
            "recommendation": recommendation,
        }

    def coverage_gaps(self, query: str, chunks: list[str]) -> dict:
        """
        Identify meaningful query terms that are absent from every retrieved chunk.

        Stopwords and tokens shorter than 2 characters are excluded so common
        function words do not produce false gaps.

        Parameters
        ----------
        query : str
            The user query to extract terms from.
        chunks : list[str]
            Retrieved document chunks to search within.

        Returns
        -------
        dict
            Keys:
              ``missing_terms`` – list of significant query terms not found in
                                  any chunk (preserves query order, deduplicated).
              ``covered_terms`` – list of significant query terms found in at
                                  least one chunk.
              ``coverage_pct``  – fraction of query terms covered, in [0.0, 1.0].
        """
        # deduplicate while preserving order
        query_terms = list(dict.fromkeys(_sig_tokens(query)))

        if not query_terms:
            return {"missing_terms": [], "covered_terms": [], "coverage_pct": 1.0}

        combined: set[str] = set()
        for chunk in chunks:
            combined |= _sig_token_set(chunk)

        covered = [t for t in query_terms if t in combined]
        missing = [t for t in query_terms if t not in combined]
        cov_pct = len(covered) / len(query_terms)

        logger.debug(
            "coverage_gaps: %d/%d terms covered (%.0f%%); missing: %s",
            len(covered), len(query_terms), cov_pct * 100, missing,
        )
        return {
            "missing_terms": missing,
            "covered_terms": covered,
            "coverage_pct":  round(cov_pct, 4),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# RetrievalDiagnostic
# ═══════════════════════════════════════════════════════════════════════════════

class RetrievalDiagnostic:
    """
    Orchestrates ContextRecallScorer and ChunkAnalyzer across a full query set
    and produces a structured findings report.

    Severity rules
    --------------
    CRITICAL : mean recall < 0.3  |  mean redundancy > 0.6
    WARNING  : mean recall 0.3–0.5  |  mean coverage gap > 50 %
    INFO     : chunk length anomalies (too short or too long)
    """

    def __init__(self) -> None:
        self._recall_scorer   = ContextRecallScorer()
        self._chunk_analyzer  = ChunkAnalyzer()

    def run(
        self,
        queries:                      list[str],
        retrieved_chunks_per_query:   list[list[str]],
        references:                   Optional[list[str]] = None,
        *,
        semantic_recall:  bool        = False,
        client:           Any         = None,
        model:            str         = "text-embedding-3-small",
    ) -> dict:
        """
        Run the full retrieval diagnostic across all queries.

        No LLM calls are made unless ``semantic_recall=True`` is passed.

        Parameters
        ----------
        queries : list[str]
            Query strings, one per evaluation example.
        retrieved_chunks_per_query : list[list[str]]
            The chunks retrieved for each query.  Must have the same length
            as ``queries``.
        references : list[str] | None
            Ground-truth answers.  When provided, per-query recall scores are
            computed and recall findings are included.  Pass ``None`` to skip.
        semantic_recall : bool
            When ``True``, compute semantic recall in addition to lexical recall
            using ``client`` and ``model``.
        client : Any
            OpenAI-compatible embedding client.  Required when
            ``semantic_recall=True``.
        model : str
            Embedding model name, used only when ``semantic_recall=True``.

        Returns
        -------
        dict
            Keys:
              ``per_query``  – list of per-query result dicts, each containing
                               ``query``, ``chunk_count``, ``redundancy``,
                               ``coverage``, ``length``, and (when references
                               provided) ``recall`` and optionally
                               ``semantic_recall``.
              ``aggregate``  – aggregate stats: ``mean_redundancy``,
                               ``mean_coverage_pct``, and (when references
                               provided) ``mean_recall``.
              ``findings``   – ranked list sorted CRITICAL → WARNING → INFO.
                               Each finding has ``severity``, ``metric``,
                               ``value``, and ``recommendation``.
        """
        n = len(queries)
        if len(retrieved_chunks_per_query) != n:
            raise ValueError(
                f"queries length ({n}) must equal "
                f"retrieved_chunks_per_query length "
                f"({len(retrieved_chunks_per_query)})"
            )
        if references is not None and len(references) != n:
            raise ValueError(
                f"references length ({len(references)}) must equal "
                f"queries length ({n})"
            )

        per_query:        list[dict]  = []
        recall_scores:    list[float] = []
        redundancy_means: list[float] = []
        coverage_pcts:    list[float] = []

        for i, (query, chunks) in enumerate(
            zip(queries, retrieved_chunks_per_query)
        ):
            entry: dict = {"query": query, "chunk_count": len(chunks)}

            # Recall (only when references supplied)
            if references is not None:
                lex = self._recall_scorer.token_recall(references[i], chunks)
                entry["recall"] = round(lex, 4)
                recall_scores.append(lex)

                if semantic_recall and client is not None:
                    sem = self._recall_scorer.semantic_recall(
                        references[i], chunks, client, model
                    )
                    entry["semantic_recall"] = round(sem, 4)

            # Structural analysis
            redundancy = self._chunk_analyzer.redundancy_score(chunks)
            coverage   = self._chunk_analyzer.coverage_gaps(query, chunks)
            length     = self._chunk_analyzer.length_distribution(chunks)

            entry["redundancy"] = redundancy
            entry["coverage"]   = coverage
            entry["length"]     = length

            redundancy_means.append(redundancy["mean"])
            coverage_pcts.append(coverage["coverage_pct"])

            logger.debug(
                "query %d/%d  recall=%.3f  redundancy_mean=%.3f  coverage=%.3f",
                i + 1, n,
                entry.get("recall", float("nan")),
                redundancy["mean"],
                coverage["coverage_pct"],
            )

            per_query.append(entry)

        # Aggregates
        mean_recall     = _stats.mean(recall_scores)     if recall_scores     else None
        mean_redundancy = _stats.mean(redundancy_means)  if redundancy_means  else 0.0
        mean_coverage   = _stats.mean(coverage_pcts)     if coverage_pcts     else 1.0

        aggregate: dict = {
            "mean_redundancy":   round(mean_redundancy, 4),
            "mean_coverage_pct": round(mean_coverage, 4),
        }
        if mean_recall is not None:
            aggregate["mean_recall"] = round(mean_recall, 4)

        findings = self._generate_findings(
            mean_recall       = mean_recall,
            mean_redundancy   = mean_redundancy,
            mean_coverage_pct = mean_coverage,
            per_query         = per_query,
        )

        logger.info(
            "Diagnostic: %d queries → %d findings (%d CRITICAL, %d WARNING)",
            n,
            len(findings),
            sum(1 for f in findings if f["severity"] == "CRITICAL"),
            sum(1 for f in findings if f["severity"] == "WARNING"),
        )

        return {"per_query": per_query, "aggregate": aggregate, "findings": findings}

    # ── Findings generation ───────────────────────────────────────────────────

    @staticmethod
    def _generate_findings(
        mean_recall:       Optional[float],
        mean_redundancy:   float,
        mean_coverage_pct: float,
        per_query:         list[dict],
    ) -> list[dict]:
        findings: list[dict] = []

        # ── Recall ────────────────────────────────────────────────────────────
        if mean_recall is not None:
            if mean_recall < _RECALL_CRITICAL:
                findings.append({
                    "severity": "CRITICAL",
                    "metric":   "recall",
                    "value":    round(mean_recall, 4),
                    "recommendation": (
                        f"Mean context recall is {mean_recall:.0%}, below the "
                        f"{_RECALL_CRITICAL:.0%} critical threshold.  "
                        "Priority fixes: (1) increase k to retrieve more chunks, "
                        "(2) re-index with a domain-specific embedding model, "
                        "(3) add BM25 as a first-stage retriever for exact-match terms."
                    ),
                })
            elif mean_recall < _RECALL_WARNING:
                findings.append({
                    "severity": "WARNING",
                    "metric":   "recall",
                    "value":    round(mean_recall, 4),
                    "recommendation": (
                        f"Mean context recall is {mean_recall:.0%} "
                        f"({_RECALL_CRITICAL:.0%}–{_RECALL_WARNING:.0%} range).  "
                        "Consider adding a cross-encoder re-ranker, increasing "
                        "chunk overlap so boundary-spanning answers are captured, "
                        "or tuning the similarity threshold."
                    ),
                })

        # ── Redundancy ────────────────────────────────────────────────────────
        if mean_redundancy > _REDUNDANCY_CRITICAL:
            chunk_counts = [q["chunk_count"] for q in per_query if q["chunk_count"] > 0]
            typical_k    = round(_stats.mean(chunk_counts)) if chunk_counts else "?"
            findings.append({
                "severity": "CRITICAL",
                "metric":   "redundancy",
                "value":    round(mean_redundancy, 4),
                "recommendation": (
                    f"Mean pairwise chunk overlap is {mean_redundancy:.0%}, "
                    f"above the {_REDUNDANCY_CRITICAL:.0%} critical threshold.  "
                    f"Your top-{typical_k} retrieved chunks are near-duplicates, "
                    "wasting context window space.  "
                    "Reduce k or apply Maximum Marginal Relevance (MMR) "
                    "reranking with λ=0.5 to diversify results."
                ),
            })

        # ── Coverage ──────────────────────────────────────────────────────────
        gap_pct = 1.0 - mean_coverage_pct
        if gap_pct > _COVERAGE_WARNING:
            findings.append({
                "severity": "WARNING",
                "metric":   "coverage",
                "value":    round(mean_coverage_pct, 4),
                "recommendation": (
                    f"On average {gap_pct:.0%} of meaningful query terms are "
                    "absent from retrieved chunks.  "
                    "Verify these terms exist in your corpus; if so, add BM25 "
                    "keyword retrieval alongside dense search (hybrid retrieval) "
                    "to surface exact-match content that embedding models miss."
                ),
            })

        # ── Chunk length (INFO) ───────────────────────────────────────────────
        length_entries = [q["length"] for q in per_query if q.get("length")]
        recs = [e["recommendation"] for e in length_entries if e.get("recommendation")]
        if recs:
            mean_len = _stats.mean(e["mean"] for e in length_entries)
            findings.append({
                "severity": "INFO",
                "metric":   "chunk_length",
                "value":    round(mean_len, 2),
                "recommendation": recs[0],
            })

        # Sort: CRITICAL → WARNING → INFO
        _order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
        findings.sort(key=lambda f: _order.get(f["severity"], 3))
        return findings
