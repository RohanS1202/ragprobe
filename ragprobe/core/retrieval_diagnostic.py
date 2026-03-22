"""
retrieval_diagnostic.py — Diagnose retrieval quality problems in RAG pipelines.

Three self-contained components (no new dependencies beyond what ragprobe already
requires — openai, numpy):

  ContextRecallScorer  — per-query recall: how much of the expected answer
                         appears in the retrieved chunks?  Two modes:
                           lexical  — token overlap, no API cost.
                           semantic — embedding cosine similarity (OpenAI).

  ChunkAnalyzer        — structural analysis of retrieved chunks:
                           redundancy    — pairwise Jaccard overlap of top-k chunks.
                           length stats  — short / long chunk distribution.
                           coverage gaps — query terms absent from every chunk.

  RetrievalDiagnostic  — orchestrator: runs both components, produces a
                         ranked Finding list (CRITICAL / WARNING / INFO)
                         with concrete, actionable recommendations.

Integration::

    from ragprobe import RetrievalDiagnostic, RecallMode
    from ragprobe.core.scorer import EvalResult

    # From a completed evaluation (most common):
    report = RetrievalDiagnostic.from_eval_results(eval_results, corpus=chunks)
    RetrievalDiagnostic.print_report(report)

    # From raw triples (pre-evaluation or standalone):
    triples = [(query, expected_answer, retrieved_chunks), ...]
    report  = RetrievalDiagnostic.run(triples, corpus=chunks)
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


# ── Thresholds ────────────────────────────────────────────────────────────────

# Chunk length (word-token count)
_SHORT_CHUNK_TOKENS = 100    # below → too short to hold a full answer
_LONG_CHUNK_TOKENS  = 600    # above → may dilute retrieval relevance

# Recall
_RECALL_CRITICAL = 0.30
_RECALL_WARNING  = 0.50

# Pairwise Jaccard overlap between chunks retrieved for the same query
_REDUNDANCY_CRITICAL = 0.70
_REDUNDANCY_WARNING  = 0.50

# Fraction of query terms present in at least one retrieved chunk
_COVERAGE_CRITICAL = 0.40
_COVERAGE_WARNING  = 0.60

# ── Common English stopwords (no NLTK needed) ─────────────────────────────────
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "or", "and", "but", "if",
    "this", "that", "it", "its", "not", "no", "so", "what", "which",
    "who", "how", "when", "where", "why", "i", "we", "you", "he", "she",
    "they", "me", "us", "him", "her", "them", "my", "your", "his", "their",
    "our", "than", "then", "there", "here", "up", "about", "any", "all",
    "each", "few", "more", "most", "some", "such", "other", "both", "nor",
    "neither", "either", "these", "those", "also", "into", "over", "after",
})


# ── Enums ─────────────────────────────────────────────────────────────────────

class RecallMode(str, Enum):
    LEXICAL  = "lexical"   # token overlap — no API cost
    SEMANTIC = "semantic"  # embedding cosine similarity — requires OpenAI


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    WARNING  = "WARNING"
    INFO     = "INFO"


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class ChunkRecallDetail:
    """Recall contribution of a single retrieved chunk."""
    chunk_index:    int
    chunk_preview:  str          # first 100 chars
    matched_tokens: list[str]    # expected tokens found in this chunk
    recall_score:   float        # matched / total_expected


@dataclass
class QueryRecallResult:
    """Per-query recall breakdown."""
    query:            str
    expected:         str
    recall_score:     float         # best single-chunk recall (lexical) or
                                    # max cosine sim (semantic)
    combined_recall:  float         # coverage across all chunks combined
    mode:             RecallMode
    matched_tokens:   list[str]     # expected tokens found in any chunk
    missed_tokens:    list[str]     # expected tokens absent from all chunks
    best_chunk_index: int           # which chunk gave the highest recall
    per_chunk:        list[ChunkRecallDetail]


@dataclass
class RecallReport:
    """Aggregate recall statistics across all queries."""
    mode:               RecallMode
    total_queries:      int
    avg_recall:         float
    median_recall:      float
    p25_recall:         float
    p75_recall:         float
    low_recall_count:   int          # queries with combined_recall < _RECALL_WARNING
    zero_recall_count:  int          # queries with combined_recall == 0
    per_query:          list[QueryRecallResult]

    @property
    def low_recall_queries(self) -> list[QueryRecallResult]:
        return [r for r in self.per_query if r.combined_recall < _RECALL_WARNING]


@dataclass
class RedundancyResult:
    """Pairwise chunk overlap for a single query."""
    query:                str
    chunk_count:          int
    avg_pairwise_overlap: float                       # mean Jaccard across all pairs
    max_pairwise_overlap: float                       # worst (most redundant) pair
    redundant_pairs:      list[tuple[int, int, float]] # (chunk_i, chunk_j, jaccard)


@dataclass
class LengthStats:
    """Token-count distribution across all chunks analysed."""
    chunk_count:     int
    min_tokens:      int
    max_tokens:      int
    mean_tokens:     float
    median_tokens:   float
    p25_tokens:      float
    p75_tokens:      float
    p95_tokens:      float
    too_short_count: int    # < _SHORT_CHUNK_TOKENS
    too_long_count:  int    # > _LONG_CHUNK_TOKENS
    too_short_pct:   float
    too_long_pct:    float


@dataclass
class CoverageGap:
    """Query-term coverage for a single query."""
    query:             str
    query_terms:       list[str]   # meaningful (non-stopword) query terms
    uncovered_terms:   list[str]   # terms absent from every retrieved chunk
    covered_terms:     list[str]   # terms present in ≥1 chunk
    coverage_score:    float       # covered / total  (0–1)


@dataclass
class ChunkAnalysis:
    """Aggregate structural analysis of all retrieved chunks."""
    length_stats:     LengthStats
    per_query_redundancy: list[RedundancyResult]
    per_query_coverage:   list[CoverageGap]
    avg_redundancy:   float   # mean avg_pairwise_overlap across queries
    avg_coverage:     float   # mean coverage_score across queries
    unique_chunks_analysed: int


@dataclass
class Finding:
    """A single ranked diagnostic observation."""
    severity:         Severity
    category:         str      # "recall" | "redundancy" | "length" | "coverage"
    title:            str      # ≤ 80 chars, human-readable headline
    detail:           str      # what exactly is happening (numbers included)
    recommendation:   str      # concrete, actionable fix
    affected_queries: list[str] = field(default_factory=list)


@dataclass
class DiagnosticReport:
    """Complete diagnostic output from RetrievalDiagnostic.run()."""
    findings:       list[Finding]      # sorted: CRITICAL → WARNING → INFO
    recall_report:  RecallReport
    chunk_analysis: ChunkAnalysis
    total_queries:  int
    summary:        str                # one-line human headline

    @property
    def critical_findings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.CRITICAL]

    @property
    def warning_findings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.WARNING]


# ── Tokenisation helpers ──────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Split into lowercase word tokens, stripping punctuation."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _significant_tokens(text: str) -> list[str]:
    """Return non-stopword tokens (preserves order, keeps duplicates)."""
    return [t for t in _tokenize(text) if t not in _STOPWORDS and len(t) > 1]


def _significant_token_set(text: str) -> set[str]:
    return set(_significant_tokens(text))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _token_recall(expected_tokens: set[str], chunk_tokens: set[str]) -> float:
    """Fraction of expected tokens present in chunk_tokens."""
    if not expected_tokens:
        return 1.0
    return len(expected_tokens & chunk_tokens) / len(expected_tokens)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ContextRecallScorer
# ═══════════════════════════════════════════════════════════════════════════════

class ContextRecallScorer:
    """
    Measures how much of the expected answer is present in retrieved chunks.

    Lexical mode (default, zero API cost):
        Tokenises both the expected answer and each chunk, then computes:
          • per-chunk recall  = |expected_tokens ∩ chunk_tokens| / |expected_tokens|
          • combined recall   = |expected_tokens ∩ ⋃(chunk_tokens)| / |expected_tokens|
        Only non-stopword tokens are used so "the", "is", etc. don't inflate scores.

    Semantic mode (requires OpenAI key):
        Embeds expected answer and each chunk via text-embedding-3-small,
        computes cosine similarity. More robust for paraphrased answers.

    Usage::

        scorer = ContextRecallScorer()
        result = scorer.score_query(
            query    = "What was Apple's revenue in 2023?",
            expected = "Apple reported $383.3 billion in revenue for fiscal 2023.",
            contexts = retrieved_chunks,
        )
        print(result.recall_score, result.missed_tokens)
    """

    def __init__(
        self,
        mode:    RecallMode         = RecallMode.LEXICAL,
        api_key: Optional[str]      = None,
    ) -> None:
        self.mode    = mode
        self._api_key = api_key

    # ── public API ───────────────────────────────────────────────────────────

    def score_query(
        self,
        query:    str,
        expected: str,
        contexts: list[str],
        *,
        mode: Optional[RecallMode] = None,
    ) -> QueryRecallResult:
        """Score recall for a single (query, expected, contexts) triple."""
        effective_mode = mode or self.mode

        if effective_mode == RecallMode.SEMANTIC:
            return self._score_semantic(query, expected, contexts)
        return self._score_lexical(query, expected, contexts)

    def score_batch(
        self,
        triples:   list[tuple[str, str, list[str]]],
        *,
        mode: Optional[RecallMode] = None,
    ) -> RecallReport:
        """
        Score a list of (query, expected, contexts) triples.

        Returns a RecallReport with per-query results and aggregate statistics.
        """
        effective_mode = mode or self.mode

        if effective_mode == RecallMode.SEMANTIC:
            results = self._batch_semantic(triples)
        else:
            results = [
                self._score_lexical(q, e, ctx) for q, e, ctx in triples
            ]

        if not results:
            return RecallReport(
                mode=effective_mode, total_queries=0,
                avg_recall=0.0, median_recall=0.0,
                p25_recall=0.0, p75_recall=0.0,
                low_recall_count=0, zero_recall_count=0, per_query=[],
            )

        scores = [r.combined_recall for r in results]
        return RecallReport(
            mode              = effective_mode,
            total_queries     = len(results),
            avg_recall        = round(float(np.mean(scores)), 3),
            median_recall     = round(float(np.median(scores)), 3),
            p25_recall        = round(float(np.percentile(scores, 25)), 3),
            p75_recall        = round(float(np.percentile(scores, 75)), 3),
            low_recall_count  = sum(1 for s in scores if s < _RECALL_WARNING),
            zero_recall_count = sum(1 for s in scores if s == 0.0),
            per_query         = results,
        )

    # ── lexical scoring ───────────────────────────────────────────────────────

    def _score_lexical(
        self,
        query:    str,
        expected: str,
        contexts: list[str],
    ) -> QueryRecallResult:
        expected_tokens = _significant_token_set(expected)

        per_chunk: list[ChunkRecallDetail] = []
        all_chunk_tokens: set[str] = set()

        for i, chunk in enumerate(contexts):
            chunk_tokens = _significant_token_set(chunk)
            all_chunk_tokens |= chunk_tokens
            matched = sorted(expected_tokens & chunk_tokens)
            score   = _token_recall(expected_tokens, chunk_tokens)
            per_chunk.append(ChunkRecallDetail(
                chunk_index    = i,
                chunk_preview  = chunk[:100].replace("\n", " "),
                matched_tokens = matched,
                recall_score   = round(score, 3),
            ))

        combined_matched = expected_tokens & all_chunk_tokens
        combined_missed  = expected_tokens - all_chunk_tokens
        combined_recall  = _token_recall(expected_tokens, all_chunk_tokens)

        best_idx   = max(range(len(per_chunk)), key=lambda i: per_chunk[i].recall_score) if per_chunk else 0
        best_score = per_chunk[best_idx].recall_score if per_chunk else 0.0

        return QueryRecallResult(
            query            = query,
            expected         = expected,
            recall_score     = round(best_score, 3),
            combined_recall  = round(combined_recall, 3),
            mode             = RecallMode.LEXICAL,
            matched_tokens   = sorted(combined_matched),
            missed_tokens    = sorted(combined_missed),
            best_chunk_index = best_idx,
            per_chunk        = per_chunk,
        )

    # ── semantic scoring ──────────────────────────────────────────────────────

    def _score_semantic(
        self,
        query:    str,
        expected: str,
        contexts: list[str],
    ) -> QueryRecallResult:
        if not contexts:
            return self._empty_semantic_result(query, expected)

        client = self._get_openai_client()
        texts  = [expected] + contexts
        embs   = self._embed(client, texts)

        expected_emb = embs[0]
        chunk_embs   = embs[1:]

        sims: list[float] = [
            float(np.dot(expected_emb, ce) /
                  (np.linalg.norm(expected_emb) * np.linalg.norm(ce) + 1e-10))
            for ce in chunk_embs
        ]

        per_chunk = [
            ChunkRecallDetail(
                chunk_index    = i,
                chunk_preview  = ctx[:100].replace("\n", " "),
                matched_tokens = [],       # N/A for semantic mode
                recall_score   = round(max(0.0, sim), 3),
            )
            for i, (ctx, sim) in enumerate(zip(contexts, sims))
        ]

        best_idx       = int(np.argmax(sims))
        combined_score = round(float(max(sims)), 3) if sims else 0.0

        # Fall back to lexical for token-level detail
        lex = self._score_lexical(query, expected, contexts)

        return QueryRecallResult(
            query            = query,
            expected         = expected,
            recall_score     = round(float(max(sims)), 3),
            combined_recall  = combined_score,
            mode             = RecallMode.SEMANTIC,
            matched_tokens   = lex.matched_tokens,
            missed_tokens    = lex.missed_tokens,
            best_chunk_index = best_idx,
            per_chunk        = per_chunk,
        )

    def _batch_semantic(
        self,
        triples: list[tuple[str, str, list[str]]],
    ) -> list[QueryRecallResult]:
        """Batch all embedding calls to minimise API round-trips."""
        client = self._get_openai_client()

        # Collect all unique texts
        all_texts: list[str] = []
        text_index: dict[str, int] = {}

        def register(t: str) -> int:
            if t not in text_index:
                text_index[t] = len(all_texts)
                all_texts.append(t)
            return text_index[t]

        plan: list[tuple[int, list[int]]] = []  # (expected_idx, [chunk_idx, ...])
        for q, e, ctxs in triples:
            ei  = register(e)
            cis = [register(c) for c in ctxs]
            plan.append((ei, cis))

        embs = self._embed(client, all_texts)

        results: list[QueryRecallResult] = []
        for (q, e, ctxs), (ei, cis) in zip(triples, plan):
            expected_emb = embs[ei]
            chunk_embs   = [embs[ci] for ci in cis]

            sims = [
                float(np.dot(expected_emb, ce) /
                      (np.linalg.norm(expected_emb) * np.linalg.norm(ce) + 1e-10))
                for ce in chunk_embs
            ]

            per_chunk = [
                ChunkRecallDetail(
                    chunk_index    = i,
                    chunk_preview  = ctx[:100].replace("\n", " "),
                    matched_tokens = [],
                    recall_score   = round(max(0.0, s), 3),
                )
                for i, (ctx, s) in enumerate(zip(ctxs, sims))
            ]

            lex      = self._score_lexical(q, e, ctxs)
            best_idx = int(np.argmax(sims)) if sims else 0

            results.append(QueryRecallResult(
                query            = q,
                expected         = e,
                recall_score     = round(float(max(sims)), 3) if sims else 0.0,
                combined_recall  = round(float(max(sims)), 3) if sims else 0.0,
                mode             = RecallMode.SEMANTIC,
                matched_tokens   = lex.matched_tokens,
                missed_tokens    = lex.missed_tokens,
                best_chunk_index = best_idx,
                per_chunk        = per_chunk,
            ))
        return results

    @staticmethod
    def _embed(client, texts: list[str]) -> list[np.ndarray]:
        """Embed a list of texts, returning one numpy vector per text."""
        # OpenAI allows up to 2048 texts per request
        BATCH = 512
        all_embs: list[np.ndarray] = []
        for start in range(0, len(texts), BATCH):
            batch = texts[start : start + BATCH]
            resp  = client.embeddings.create(
                model = "text-embedding-3-small",
                input = batch,
            )
            all_embs.extend(np.array(d.embedding) for d in resp.data)
        return all_embs

    def _get_openai_client(self):
        import os
        from dotenv import load_dotenv
        load_dotenv()
        from openai import OpenAI
        key = self._api_key or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
        if not key:
            raise ValueError(
                "Semantic recall mode requires an OpenAI key. "
                "Set OPENAI_API_KEY or pass api_key= to ContextRecallScorer, "
                "or use mode=RecallMode.LEXICAL for zero-cost scoring."
            )
        return OpenAI(api_key=key)

    @staticmethod
    def _empty_semantic_result(query: str, expected: str) -> QueryRecallResult:
        return QueryRecallResult(
            query=query, expected=expected, recall_score=0.0,
            combined_recall=0.0, mode=RecallMode.SEMANTIC,
            matched_tokens=[], missed_tokens=_significant_tokens(expected),
            best_chunk_index=0, per_chunk=[],
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ChunkAnalyzer
# ═══════════════════════════════════════════════════════════════════════════════

class ChunkAnalyzer:
    """
    Examines the structure of retrieved chunks for systemic retrieval problems.

    Three analyses:

      Redundancy  — pairwise Jaccard token overlap between chunks retrieved for
                    the same query.  High overlap (> 0.7) means you're spending
                    your context window on near-duplicate content.

      Length      — token-count distribution.  Chunks below 100 tokens are often
                    too narrow to contain a complete answer; chunks above 600
                    dilute relevance by mixing in irrelevant sentences.

      Coverage    — for each query, which meaningful terms appear in zero
                    retrieved chunks.  A coverage gap means the retriever
                    failed to surface even adjacent content.

    Usage::

        analyzer = ChunkAnalyzer()
        analysis = analyzer.analyze(
            queries_and_contexts = [(query, contexts), ...],
            corpus               = all_document_chunks,   # optional
        )
        print(analysis.avg_redundancy, analysis.avg_coverage)
    """

    # ── public API ───────────────────────────────────────────────────────────

    def analyze(
        self,
        queries_and_contexts: list[tuple[str, list[str]]],
        corpus:               Optional[list[str]] = None,
    ) -> ChunkAnalysis:
        """
        Run all three analyses.

        Args:
            queries_and_contexts: List of (query, retrieved_chunks) pairs.
            corpus:               Full document corpus.  When provided, length
                                  statistics reflect the whole corpus, not just
                                  the chunks that happened to be retrieved.
        """
        redundancy_results = [
            self._redundancy(q, ctxs)
            for q, ctxs in queries_and_contexts
            if len(ctxs) >= 2
        ]
        coverage_results = [
            self._coverage(q, ctxs)
            for q, ctxs in queries_and_contexts
        ]

        # Length analysis: corpus if provided, else all retrieved chunks
        chunks_for_length = corpus if corpus else [
            c for _, ctxs in queries_and_contexts for c in ctxs
        ]
        length_stats = self._length_stats(chunks_for_length)

        # Collect unique retrieved chunks for the report header
        unique_retrieved = {c for _, ctxs in queries_and_contexts for c in ctxs}

        avg_redundancy = (
            float(np.mean([r.avg_pairwise_overlap for r in redundancy_results]))
            if redundancy_results else 0.0
        )
        avg_coverage = (
            float(np.mean([c.coverage_score for c in coverage_results]))
            if coverage_results else 1.0
        )

        return ChunkAnalysis(
            length_stats          = length_stats,
            per_query_redundancy  = redundancy_results,
            per_query_coverage    = coverage_results,
            avg_redundancy        = round(avg_redundancy, 3),
            avg_coverage          = round(avg_coverage, 3),
            unique_chunks_analysed = len(unique_retrieved),
        )

    # ── redundancy ────────────────────────────────────────────────────────────

    @staticmethod
    def _redundancy(query: str, contexts: list[str]) -> RedundancyResult:
        token_sets = [_significant_token_set(c) for c in contexts]
        pairs: list[tuple[int, int, float]] = []

        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                overlap = _jaccard(token_sets[i], token_sets[j])
                pairs.append((i, j, round(overlap, 3)))

        avg_overlap = float(np.mean([p[2] for p in pairs])) if pairs else 0.0
        max_overlap = max((p[2] for p in pairs), default=0.0)
        redundant   = [p for p in pairs if p[2] >= _REDUNDANCY_WARNING]

        return RedundancyResult(
            query                = query,
            chunk_count          = len(contexts),
            avg_pairwise_overlap = round(avg_overlap, 3),
            max_pairwise_overlap = round(max_overlap, 3),
            redundant_pairs      = redundant,
        )

    # ── coverage ──────────────────────────────────────────────────────────────

    @staticmethod
    def _coverage(query: str, contexts: list[str]) -> CoverageGap:
        query_terms  = list(dict.fromkeys(_significant_tokens(query)))  # deduplicated, ordered
        all_chunk_tokens: set[str] = set()
        for c in contexts:
            all_chunk_tokens |= _significant_token_set(c)

        covered   = [t for t in query_terms if t in all_chunk_tokens]
        uncovered = [t for t in query_terms if t not in all_chunk_tokens]
        score     = len(covered) / len(query_terms) if query_terms else 1.0

        return CoverageGap(
            query           = query,
            query_terms     = query_terms,
            uncovered_terms = uncovered,
            covered_terms   = covered,
            coverage_score  = round(score, 3),
        )

    # ── length stats ──────────────────────────────────────────────────────────

    @staticmethod
    def _length_stats(chunks: list[str]) -> LengthStats:
        if not chunks:
            return LengthStats(
                chunk_count=0, min_tokens=0, max_tokens=0,
                mean_tokens=0.0, median_tokens=0.0,
                p25_tokens=0.0, p75_tokens=0.0, p95_tokens=0.0,
                too_short_count=0, too_long_count=0,
                too_short_pct=0.0, too_long_pct=0.0,
            )

        counts = [len(_tokenize(c)) for c in chunks]
        arr    = np.array(counts, dtype=float)

        too_short = sum(1 for n in counts if n < _SHORT_CHUNK_TOKENS)
        too_long  = sum(1 for n in counts if n > _LONG_CHUNK_TOKENS)

        return LengthStats(
            chunk_count     = len(counts),
            min_tokens      = int(np.min(arr)),
            max_tokens      = int(np.max(arr)),
            mean_tokens     = round(float(np.mean(arr)), 1),
            median_tokens   = round(float(np.median(arr)), 1),
            p25_tokens      = round(float(np.percentile(arr, 25)), 1),
            p75_tokens      = round(float(np.percentile(arr, 75)), 1),
            p95_tokens      = round(float(np.percentile(arr, 95)), 1),
            too_short_count = too_short,
            too_long_count  = too_long,
            too_short_pct   = round(too_short / len(counts), 3),
            too_long_pct    = round(too_long / len(counts), 3),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. RetrievalDiagnostic  (orchestrator)
# ═══════════════════════════════════════════════════════════════════════════════

class RetrievalDiagnostic:
    """
    Orchestrates ContextRecallScorer and ChunkAnalyzer, then produces a
    ranked findings report with severity levels and concrete recommendations.

    Primary entry points::

        # From a completed evaluation run (preferred):
        report = RetrievalDiagnostic.from_eval_results(eval_results, corpus=docs)

        # From raw (query, expected, contexts) triples:
        report = RetrievalDiagnostic.run(triples, corpus=docs)

        # Print to terminal:
        RetrievalDiagnostic.print_report(report)
    """

    # ── Entry points ─────────────────────────────────────────────────────────

    @classmethod
    def run(
        cls,
        triples: list[tuple[str, str, list[str]]],
        *,
        corpus:  Optional[list[str]] = None,
        mode:    RecallMode           = RecallMode.LEXICAL,
        api_key: Optional[str]        = None,
    ) -> DiagnosticReport:
        """
        Run the full diagnostic on a list of (query, expected_answer, contexts) triples.

        Args:
            triples: Each tuple is (query_string, expected_answer, retrieved_chunks).
            corpus:  Full document corpus for length analysis.  Omit to use only
                     the retrieved chunks (less informative for length stats).
            mode:    RecallMode.LEXICAL (default) or RecallMode.SEMANTIC.
            api_key: OpenAI key (only required for SEMANTIC mode).

        Returns:
            DiagnosticReport with ranked findings and raw analysis data.
        """
        scorer   = ContextRecallScorer(mode=mode, api_key=api_key)
        analyzer = ChunkAnalyzer()

        recall_report  = scorer.score_batch(triples, mode=mode)
        qac            = [(q, ctx) for q, _, ctx in triples]
        chunk_analysis = analyzer.analyze(qac, corpus=corpus)

        findings = cls._generate_findings(recall_report, chunk_analysis)
        summary  = cls._one_line_summary(recall_report, chunk_analysis, findings)

        return DiagnosticReport(
            findings       = findings,
            recall_report  = recall_report,
            chunk_analysis = chunk_analysis,
            total_queries  = len(triples),
            summary        = summary,
        )

    @classmethod
    def from_eval_results(
        cls,
        results:  list,                    # list[EvalResult] — lazy import avoids circular dep
        *,
        corpus:   Optional[list[str]] = None,
        mode:     RecallMode           = RecallMode.LEXICAL,
        api_key:  Optional[str]        = None,
    ) -> DiagnosticReport:
        """
        Build a DiagnosticReport from a completed ragprobe evaluation.

        Maps EvalResult fields:
          query    ← result.question
          expected ← result.test_case.expected_answer  (falls back to expected_behavior)
          contexts ← result.contexts

        Args:
            results: Output of Scorer.score_batch() or RAGEvaluator.evaluate().
            corpus:  Full document corpus (optional, improves length analysis).
            mode:    RecallMode.LEXICAL or RecallMode.SEMANTIC.
            api_key: OpenAI key (only for SEMANTIC mode).
        """
        triples: list[tuple[str, str, list[str]]] = []
        for r in results:
            expected = (
                r.test_case.expected_answer
                if r.test_case.expected_answer
                else r.test_case.expected_behavior
            )
            triples.append((r.question, expected, r.contexts))
        return cls.run(triples, corpus=corpus, mode=mode, api_key=api_key)

    # ── Findings generation ───────────────────────────────────────────────────

    @classmethod
    def _generate_findings(
        cls,
        rr: RecallReport,
        ca: ChunkAnalysis,
    ) -> list[Finding]:
        findings: list[Finding] = []
        ls = ca.length_stats

        # ── 1. Recall findings ────────────────────────────────────────────────
        if rr.total_queries > 0:
            if rr.zero_recall_count > 0:
                pct = rr.zero_recall_count / rr.total_queries
                findings.append(Finding(
                    severity       = Severity.CRITICAL,
                    category       = "recall",
                    title          = f"{rr.zero_recall_count} quer{'y' if rr.zero_recall_count==1 else 'ies'} have zero recall — retriever found nothing useful",
                    detail         = (
                        f"{rr.zero_recall_count}/{rr.total_queries} queries "
                        f"({pct:.0%}) returned chunks with no tokens from the "
                        f"expected answer.  The retriever is completely missing "
                        f"the relevant content."
                    ),
                    recommendation = (
                        "Check whether these queries' key terms exist in your corpus at all "
                        "(run ChunkAnalyzer.coverage).  If absent: expand the corpus.  "
                        "If present: your embedding model or similarity threshold is filtering "
                        "them out — lower the similarity threshold or switch to hybrid BM25+semantic retrieval."
                    ),
                    affected_queries = [
                        r.query for r in rr.per_query if r.combined_recall == 0.0
                    ],
                ))

            if rr.avg_recall < _RECALL_CRITICAL:
                findings.append(Finding(
                    severity       = Severity.CRITICAL,
                    category       = "recall",
                    title          = f"Critical context recall: {rr.avg_recall:.0%} average (threshold {_RECALL_CRITICAL:.0%})",
                    detail         = (
                        f"Average combined recall across {rr.total_queries} queries is "
                        f"{rr.avg_recall:.0%} (median {rr.median_recall:.0%}, "
                        f"p25={rr.p25_recall:.0%}, p75={rr.p75_recall:.0%}).  "
                        f"{rr.low_recall_count} quer{'y' if rr.low_recall_count==1 else 'ies'} "
                        f"score below {_RECALL_WARNING:.0%}."
                    ),
                    recommendation = (
                        "Recall is critically low — this is the primary driver of hallucination.  "
                        "Priority fixes: (1) increase k (retrieve more chunks), "
                        "(2) re-index with a domain-specific embedding model, "
                        "(3) add BM25 as a first-stage retriever to catch exact-match terms."
                    ),
                    affected_queries = [r.query for r in rr.low_recall_queries],
                ))
            elif rr.avg_recall < _RECALL_WARNING:
                findings.append(Finding(
                    severity       = Severity.WARNING,
                    category       = "recall",
                    title          = f"Below-threshold context recall: {rr.avg_recall:.0%} average",
                    detail         = (
                        f"Average recall is {rr.avg_recall:.0%} against the "
                        f"{_RECALL_WARNING:.0%} warning threshold.  "
                        f"{rr.low_recall_count} quer{'y' if rr.low_recall_count==1 else 'ies'} "
                        f"are below threshold."
                    ),
                    recommendation = (
                        "Consider: (1) increasing k to retrieve more candidates, "
                        "(2) adding a re-ranker (cross-encoder) to re-order retrieved chunks, "
                        "(3) tuning chunk overlap so answers that straddle chunk boundaries "
                        "are captured."
                    ),
                    affected_queries = [r.query for r in rr.low_recall_queries],
                ))

        # ── 2. Redundancy findings ────────────────────────────────────────────
        if ca.per_query_redundancy:
            if ca.avg_redundancy >= _REDUNDANCY_CRITICAL:
                k = ca.per_query_redundancy[0].chunk_count if ca.per_query_redundancy else "?"
                findings.append(Finding(
                    severity       = Severity.CRITICAL,
                    category       = "redundancy",
                    title          = f"High chunk redundancy: {ca.avg_redundancy:.0%} avg pairwise token overlap",
                    detail         = (
                        f"Your top-{k} retrieved chunks share {ca.avg_redundancy:.0%} "
                        f"average pairwise token overlap.  You are filling the context "
                        f"window with near-duplicate content, leaving no room for "
                        f"complementary information."
                    ),
                    recommendation = (
                        f"Reduce k from {k} to {max(1, k-2)} OR add Maximum Marginal Relevance "
                        f"(MMR) reranking with λ=0.5 to diversify retrieved chunks.  "
                        f"Also consider whether your chunking strategy creates too many "
                        f"overlapping windows."
                    ),
                    affected_queries = [
                        r.query for r in ca.per_query_redundancy
                        if r.avg_pairwise_overlap >= _REDUNDANCY_CRITICAL
                    ],
                ))
            elif ca.avg_redundancy >= _REDUNDANCY_WARNING:
                findings.append(Finding(
                    severity       = Severity.WARNING,
                    category       = "redundancy",
                    title          = f"Moderate chunk redundancy: {ca.avg_redundancy:.0%} avg pairwise overlap",
                    detail         = (
                        f"Average pairwise Jaccard overlap between retrieved chunks is "
                        f"{ca.avg_redundancy:.0%}.  Some context window space is being "
                        f"wasted on overlapping content."
                    ),
                    recommendation = (
                        "Apply MMR reranking (λ=0.7) or deduplicate chunks that share "
                        f"> {_REDUNDANCY_WARNING:.0%} token overlap before sending to the LLM."
                    ),
                    affected_queries = [
                        r.query for r in ca.per_query_redundancy
                        if r.avg_pairwise_overlap >= _REDUNDANCY_WARNING
                    ],
                ))

        # ── 3. Length findings ────────────────────────────────────────────────
        if ls.chunk_count > 0:
            if ls.too_short_pct >= 0.50:
                target = max(int(ls.median_tokens * 1.8), _SHORT_CHUNK_TOKENS + 50)
                findings.append(Finding(
                    severity       = Severity.CRITICAL,
                    category       = "length",
                    title          = f"{ls.too_short_pct:.0%} of chunks are too short (< {_SHORT_CHUNK_TOKENS} tokens)",
                    detail         = (
                        f"{ls.too_short_count}/{ls.chunk_count} chunks have fewer than "
                        f"{_SHORT_CHUNK_TOKENS} tokens (median: {ls.median_tokens:.0f} tokens).  "
                        f"Chunks this short rarely contain a complete answer, forcing the LLM "
                        f"to hallucinate the missing context."
                    ),
                    recommendation = (
                        f"Increase chunk size from ~{ls.median_tokens:.0f} to ~{target} tokens.  "
                        f"Also add a 15–20% token overlap between adjacent chunks so answers "
                        f"that straddle boundaries are not split."
                    ),
                ))
            elif ls.too_short_pct >= 0.25:
                target = max(int(ls.median_tokens * 1.5), _SHORT_CHUNK_TOKENS + 50)
                findings.append(Finding(
                    severity       = Severity.WARNING,
                    category       = "length",
                    title          = f"{ls.too_short_pct:.0%} of chunks may be too short (< {_SHORT_CHUNK_TOKENS} tokens)",
                    detail         = (
                        f"{ls.too_short_count} chunks have fewer than {_SHORT_CHUNK_TOKENS} tokens "
                        f"(median: {ls.median_tokens:.0f}).  Short chunks increase the risk of "
                        f"splitting answers across chunk boundaries."
                    ),
                    recommendation = (
                        f"Consider increasing target chunk size from ~{ls.median_tokens:.0f} "
                        f"to ~{target} tokens, or merging very short chunks with their neighbours."
                    ),
                ))

            if ls.too_long_pct >= 0.30:
                findings.append(Finding(
                    severity       = Severity.WARNING,
                    category       = "length",
                    title          = f"{ls.too_long_pct:.0%} of chunks exceed {_LONG_CHUNK_TOKENS} tokens — may dilute relevance",
                    detail         = (
                        f"{ls.too_long_count}/{ls.chunk_count} chunks exceed "
                        f"{_LONG_CHUNK_TOKENS} tokens (p95: {ls.p95_tokens:.0f} tokens).  "
                        f"Very long chunks lower the signal-to-noise ratio of retrieved content."
                    ),
                    recommendation = (
                        f"Split chunks larger than {_LONG_CHUNK_TOKENS} tokens at natural "
                        f"sentence or paragraph boundaries.  Target a p95 below "
                        f"{_LONG_CHUNK_TOKENS} tokens."
                    ),
                ))

            # High length variance → inconsistent retrieval quality
            if ls.chunk_count >= 10:
                iqr = ls.p75_tokens - ls.p25_tokens
                cv  = (ls.p95_tokens - ls.min_tokens) / (ls.mean_tokens + 1e-6)
                if cv > 5.0:
                    findings.append(Finding(
                        severity       = Severity.INFO,
                        category       = "length",
                        title          = "High chunk length variance may cause inconsistent retrieval",
                        detail         = (
                            f"Chunk sizes range from {ls.min_tokens} to {ls.max_tokens} tokens "
                            f"(IQR {ls.p25_tokens:.0f}–{ls.p75_tokens:.0f}).  "
                            f"Extreme variation means embedding distances are not comparable "
                            f"across chunks of very different sizes."
                        ),
                        recommendation = (
                            "Use a fixed-size chunking strategy or normalise embeddings by "
                            "chunk length.  Aim for a p25/p75 ratio > 0.5."
                        ),
                    ))

        # ── 4. Coverage findings ──────────────────────────────────────────────
        if ca.per_query_coverage:
            if ca.avg_coverage < _COVERAGE_CRITICAL:
                worst = sorted(ca.per_query_coverage, key=lambda g: g.coverage_score)[:3]
                findings.append(Finding(
                    severity       = Severity.CRITICAL,
                    category       = "coverage",
                    title          = f"Critical query-term coverage gap: {ca.avg_coverage:.0%} avg coverage",
                    detail         = (
                        f"On average, only {ca.avg_coverage:.0%} of meaningful query terms "
                        f"appear in any retrieved chunk.  The retriever is not surfacing "
                        f"content related to the key concepts in the query."
                    ),
                    recommendation = (
                        "Check whether the uncovered terms exist anywhere in the corpus.  "
                        "If absent → expand the corpus.  "
                        "If present → your embedding similarity is failing for these terms; "
                        "add BM25 keyword search as a first-stage retriever (hybrid retrieval)."
                    ),
                    affected_queries = [g.query for g in worst],
                ))
            elif ca.avg_coverage < _COVERAGE_WARNING:
                findings.append(Finding(
                    severity       = Severity.WARNING,
                    category       = "coverage",
                    title          = f"Below-threshold query-term coverage: {ca.avg_coverage:.0%} avg",
                    detail         = (
                        f"Average query-term coverage is {ca.avg_coverage:.0%} "
                        f"(threshold {_COVERAGE_WARNING:.0%}).  "
                        f"Some meaningful query terms are not found in retrieved chunks."
                    ),
                    recommendation = (
                        "Review the uncovered terms in ChunkAnalysis.per_query_coverage.  "
                        "Consider adding synonyms or domain-specific tokenisation, "
                        "or augmenting with keyword-based (BM25) retrieval."
                    ),
                    affected_queries = [
                        g.query for g in ca.per_query_coverage
                        if g.coverage_score < _COVERAGE_WARNING
                    ],
                ))

        # ── Sort: CRITICAL first, then WARNING, then INFO ─────────────────────
        order = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}
        findings.sort(key=lambda f: order[f.severity])
        return findings

    # ── Summary ───────────────────────────────────────────────────────────────

    @staticmethod
    def _one_line_summary(
        rr: RecallReport,
        ca: ChunkAnalysis,
        findings: list[Finding],
    ) -> str:
        n_critical = sum(1 for f in findings if f.severity == Severity.CRITICAL)
        n_warning  = sum(1 for f in findings if f.severity == Severity.WARNING)

        parts = [
            f"recall={rr.avg_recall:.0%}",
            f"redundancy={ca.avg_redundancy:.0%}",
            f"coverage={ca.avg_coverage:.0%}",
        ]
        flag = (
            f"{n_critical} CRITICAL, {n_warning} WARNING"
            if n_critical or n_warning
            else "no issues found"
        )
        return f"[{flag}]  " + "  ".join(parts)

    # ── Rich terminal output ──────────────────────────────────────────────────

    @staticmethod
    def print_report(report: DiagnosticReport) -> None:
        """Print a structured, colour-coded diagnostic report to the terminal."""
        from rich.console import Console
        from rich.table   import Table
        from rich.panel   import Panel
        from rich         import box

        console = Console()

        # ── Header ────────────────────────────────────────────────────────────
        console.print()
        console.print(Panel(
            f"[bold]ragprobe retrieval diagnostic[/bold]\n"
            f"{report.total_queries} queries  |  "
            f"recall mode: {report.recall_report.mode.value}  |  "
            f"{report.chunk_analysis.unique_chunks_analysed} unique chunks analysed",
            expand=False,
        ))
        console.print(f"[dim]{report.summary}[/dim]\n")

        # ── Metric overview ───────────────────────────────────────────────────
        rr = report.recall_report
        ca = report.chunk_analysis
        ls = ca.length_stats

        table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        table.add_column("Metric",   style="dim")
        table.add_column("Value",    justify="right")
        table.add_column("p25",      justify="right")
        table.add_column("Median",   justify="right")
        table.add_column("p75",      justify="right")

        def _rc(v: float, lo: float, hi: float) -> str:
            c = "green" if v >= hi else "yellow" if v >= lo else "red"
            return f"[{c}]{v:.0%}[/{c}]"

        table.add_row(
            "Context recall (combined)",
            _rc(rr.avg_recall, _RECALL_CRITICAL, _RECALL_WARNING),
            f"{rr.p25_recall:.0%}",
            f"{rr.median_recall:.0%}",
            f"{rr.p75_recall:.0%}",
        )
        table.add_row(
            "Chunk redundancy (Jaccard)",
            _rc(1 - ca.avg_redundancy, 1 - _REDUNDANCY_WARNING, 1 - _REDUNDANCY_CRITICAL),
            "—", "—", "—",
        )
        table.add_row(
            "Query-term coverage",
            _rc(ca.avg_coverage, _COVERAGE_CRITICAL, _COVERAGE_WARNING),
            "—", "—", "—",
        )
        table.add_row(
            "Chunk length (tokens)",
            f"{ls.median_tokens:.0f}",
            f"{ls.p25_tokens:.0f}",
            f"{ls.median_tokens:.0f}",
            f"{ls.p75_tokens:.0f}",
        )
        console.print(table)

        # ── Findings ──────────────────────────────────────────────────────────
        if not report.findings:
            console.print("[green]No issues detected.[/green]\n")
            return

        console.print(f"[bold]Findings ({len(report.findings)} total)[/bold]\n")

        _sev_style = {
            Severity.CRITICAL: ("red",    "✖ CRITICAL"),
            Severity.WARNING:  ("yellow", "⚠ WARNING "),
            Severity.INFO:     ("blue",   "ℹ INFO    "),
        }

        for i, f in enumerate(report.findings, 1):
            color, label = _sev_style[f.severity]
            console.print(
                f"[{color}]{label}[/{color}]  [{color}]{f.title}[/{color}]"
            )
            console.print(f"  [dim]Category:[/dim] {f.category}")
            console.print(f"  [dim]Detail:[/dim]   {f.detail}")
            console.print(
                f"  [dim]Fix:[/dim]      [italic]{f.recommendation}[/italic]"
            )
            if f.affected_queries:
                sample = f.affected_queries[:2]
                more   = len(f.affected_queries) - len(sample)
                qs     = "; ".join(f'"{q[:60]}"' for q in sample)
                suffix = f" (+{more} more)" if more > 0 else ""
                console.print(f"  [dim]Queries:[/dim]   {qs}{suffix}")
            if i < len(report.findings):
                console.print()

        console.print()
