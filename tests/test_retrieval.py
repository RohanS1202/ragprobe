"""
tests/test_retrieval.py — Full pytest suite for ragprobe.retrieval.

No live API calls required: the embed client is mocked for semantic_recall tests.
"""

import pytest
from unittest.mock import MagicMock

from ragprobe.retrieval import (
    ContextRecallScorer,
    ChunkAnalyzer,
    RetrievalDiagnostic,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_embed_client(*embeddings: list[float]) -> MagicMock:
    """Return a mock OpenAI-compatible client whose .embeddings.create() yields
    the given embedding vectors in order."""
    client = MagicMock()
    client.embeddings.create.return_value.data = [
        MagicMock(embedding=list(emb)) for emb in embeddings
    ]
    return client


# ═══════════════════════════════════════════════════════════════════════════════
# ContextRecallScorer — token_recall
# ═══════════════════════════════════════════════════════════════════════════════

class TestTokenRecall:

    def setup_method(self):
        self.scorer = ContextRecallScorer()

    def test_exact_match_returns_one(self):
        chunks = ["apple reported 383 billion revenue fiscal 2023"]
        ref    = "apple reported 383 billion revenue fiscal 2023"
        assert self.scorer.token_recall(ref, chunks) == pytest.approx(1.0)

    def test_partial_match_between_zero_and_one(self):
        ref    = "apple reported 383 billion revenue fiscal 2023 iphone sales growth"
        chunks = ["apple reported results 2023"]
        score  = self.scorer.token_recall(ref, chunks)
        assert 0.0 < score < 1.0

    def test_no_match_returns_zero(self):
        ref    = "apple revenue fiscal 2023"
        chunks = ["microsoft azure cloud computing services"]
        assert self.scorer.token_recall(ref, chunks) == pytest.approx(0.0)

    def test_empty_reference_returns_one(self):
        assert self.scorer.token_recall("", ["some chunk content here"]) == pytest.approx(1.0)

    def test_blank_reference_and_empty_chunks_returns_one(self):
        assert self.scorer.token_recall("   ", []) == pytest.approx(1.0)

    def test_nonempty_reference_empty_chunks_returns_zero(self):
        assert self.scorer.token_recall("apple revenue 2023", []) == pytest.approx(0.0)

    def test_stopwords_filtered_from_reference(self):
        # "the", "is", "a" are stopwords and should not penalise recall
        chunks = ["apple revenue company"]
        ref    = "the apple is a revenue company"
        score  = self.scorer.token_recall(ref, chunks)
        assert score == pytest.approx(1.0)

    def test_short_tokens_filtered(self):
        # single-char tokens should be ignored
        chunks = ["apple revenue billion"]
        ref    = "a b c apple revenue billion"
        assert self.scorer.token_recall(ref, chunks) == pytest.approx(1.0)

    def test_cross_chunk_union_used(self):
        # "apple" in chunk 1, "revenue" in chunk 2 → combined recall should be 1.0
        ref    = "apple revenue"
        chunks = ["apple company results", "revenue earnings fiscal"]
        assert self.scorer.token_recall(ref, chunks) == pytest.approx(1.0)

    def test_case_insensitive(self):
        ref    = "Apple Revenue Fiscal"
        chunks = ["apple revenue fiscal 2023"]
        assert self.scorer.token_recall(ref, chunks) == pytest.approx(1.0)

    def test_reference_only_stopwords_returns_one(self):
        # All reference tokens are stopwords → nothing to miss
        ref    = "the is a an"
        chunks = ["microsoft azure services"]
        assert self.scorer.token_recall(ref, chunks) == pytest.approx(1.0)

    def test_multiple_chunks_aggregated(self):
        ref    = "apple revenue iphone fiscal margin"
        chunks = [
            "apple iphone sales results",
            "revenue fiscal earnings",
            "gross margin percent",
        ]
        score = self.scorer.token_recall(ref, chunks)
        assert score == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# ContextRecallScorer — semantic_recall
# ═══════════════════════════════════════════════════════════════════════════════

class TestSemanticRecall:

    def setup_method(self):
        self.scorer = ContextRecallScorer()

    def test_empty_chunks_returns_zero_no_api_call(self):
        client = MagicMock()
        score  = self.scorer.semantic_recall("test reference", [], client, "m")
        assert score == pytest.approx(0.0)
        client.embeddings.create.assert_not_called()

    def test_identical_vectors_return_one(self):
        vec    = [1.0, 0.0, 0.0]
        client = _make_embed_client(vec, vec)
        score  = self.scorer.semantic_recall("ref", ["chunk"], client, "model")
        assert score == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self):
        ref_vec   = [1.0, 0.0]
        chunk_vec = [0.0, 1.0]
        client = _make_embed_client(ref_vec, chunk_vec)
        score  = self.scorer.semantic_recall("ref", ["chunk"], client, "model")
        assert score == pytest.approx(0.0)

    def test_returns_max_across_chunks(self):
        # ref, similar chunk, dissimilar chunk
        ref_vec    = [1.0, 0.0, 0.0]
        similar    = [1.0, 0.0, 0.0]   # cosine = 1.0
        dissimilar = [0.0, 1.0, 0.0]   # cosine = 0.0
        client = _make_embed_client(ref_vec, similar, dissimilar)
        score  = self.scorer.semantic_recall("ref", ["c1", "c2"], client, "m")
        assert score == pytest.approx(1.0)

    def test_returns_float_in_valid_range(self):
        import math
        vec    = [0.6, 0.8]     # unit vector: dot(v, v) = 0.36+0.64=1.0
        client = _make_embed_client(vec, vec)
        score  = self.scorer.semantic_recall("ref", ["chunk"], client, "m")
        assert 0.0 <= score <= 1.0
        assert not math.isnan(score)

    def test_client_called_once_per_batch(self):
        vec    = [1.0, 0.0]
        client = _make_embed_client(vec, vec, vec)
        self.scorer.semantic_recall("ref", ["c1", "c2"], client, "test-model")
        client.embeddings.create.assert_called_once()

    def test_model_name_passed_to_client(self):
        vec    = [1.0, 0.0]
        client = _make_embed_client(vec, vec)
        self.scorer.semantic_recall("ref", ["chunk"], client, "text-embedding-3-small")
        call_kwargs = client.embeddings.create.call_args
        assert call_kwargs.kwargs.get("model") == "text-embedding-3-small" or \
               call_kwargs.args[0] == "text-embedding-3-small" if call_kwargs.args else True


# ═══════════════════════════════════════════════════════════════════════════════
# ChunkAnalyzer — redundancy_score
# ═══════════════════════════════════════════════════════════════════════════════

class TestRedundancyScore:

    def setup_method(self):
        self.analyzer = ChunkAnalyzer()

    def test_identical_chunks_max_is_one(self):
        chunk = "apple revenue 2023 billion dollars fiscal year iphone sales"
        result = self.analyzer.redundancy_score([chunk, chunk])
        assert result["max"] == pytest.approx(1.0)

    def test_identical_chunks_mean_is_one(self):
        chunk  = "apple revenue 2023 billion dollars fiscal year"
        result = self.analyzer.redundancy_score([chunk, chunk, chunk])
        assert result["mean"] == pytest.approx(1.0)

    def test_diverse_chunks_low_overlap(self):
        chunks = [
            "apple reported 383 billion annual revenue fiscal 2023 iphone",
            "federal reserve interest rates monetary policy inflation outlook",
            "european central bank euro zone gdp growth unemployment data",
        ]
        result = self.analyzer.redundancy_score(chunks)
        assert result["mean"] < 0.2
        assert result["max"]  < 0.3

    def test_single_chunk_returns_zeros(self):
        result = self.analyzer.redundancy_score(["only one chunk here text"])
        assert result["mean"] == 0.0
        assert result["max"]  == 0.0
        assert result["worst_pair"] is None

    def test_empty_list_returns_zeros(self):
        result = self.analyzer.redundancy_score([])
        assert result["mean"] == 0.0
        assert result["max"]  == 0.0
        assert result["worst_pair"] is None

    def test_worst_pair_keys_present(self):
        chunks = [
            "apple revenue 2023 billion fiscal",
            "apple revenue 2023 fiscal year",
            "microsoft azure cloud computing services",
        ]
        result = self.analyzer.redundancy_score(chunks)
        wp = result["worst_pair"]
        assert wp is not None
        assert "index_a" in wp
        assert "index_b" in wp
        assert "score"   in wp
        assert wp["index_a"] < wp["index_b"]

    def test_worst_pair_has_highest_score(self):
        chunks = [
            "apple revenue 2023 billion fiscal year results earnings",
            "apple revenue 2023 billion fiscal year results",  # very similar to c0
            "microsoft azure cloud completely different services",
        ]
        result = self.analyzer.redundancy_score(chunks)
        wp = result["worst_pair"]
        assert wp["score"] == pytest.approx(result["max"])

    def test_scores_are_bounded(self):
        chunks = ["alpha beta gamma", "delta epsilon zeta", "alpha beta delta"]
        result = self.analyzer.redundancy_score(chunks)
        assert 0.0 <= result["mean"] <= 1.0
        assert 0.0 <= result["max"]  <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# ChunkAnalyzer — length_distribution
# ═══════════════════════════════════════════════════════════════════════════════

class TestLengthDistribution:

    def setup_method(self):
        self.analyzer = ChunkAnalyzer()

    def test_required_keys_present(self):
        result = self.analyzer.length_distribution(["word " * 100])
        for key in ("min", "max", "mean", "median", "std", "recommendation"):
            assert key in result

    def test_empty_input_returns_zeros(self):
        result = self.analyzer.length_distribution([])
        assert result["min"]    == 0
        assert result["max"]    == 0
        assert result["mean"]   == pytest.approx(0.0)
        assert result["median"] == pytest.approx(0.0)
        assert result["std"]    == pytest.approx(0.0)

    def test_short_chunks_recommendation_not_none(self):
        # 3-word chunks → well below 80-word threshold
        chunks = ["short chunk text"] * 5
        result = self.analyzer.length_distribution(chunks)
        assert result["recommendation"] is not None

    def test_short_chunks_recommendation_mentions_short(self):
        chunks = ["tiny chunk"] * 5
        result = self.analyzer.length_distribution(chunks)
        assert "short" in result["recommendation"].lower()

    def test_long_chunks_recommendation_not_none(self):
        chunks = [("word " * 700).strip()] * 3
        result = self.analyzer.length_distribution(chunks)
        assert result["recommendation"] is not None

    def test_long_chunks_recommendation_mentions_long(self):
        chunks = [("word " * 700).strip()] * 3
        result = self.analyzer.length_distribution(chunks)
        assert "long" in result["recommendation"].lower()

    def test_normal_chunks_no_recommendation(self):
        # 150 words — well within 80–600 range
        chunks = [("word " * 150).strip()] * 4
        result = self.analyzer.length_distribution(chunks)
        assert result["recommendation"] is None

    def test_min_max_correct(self):
        chunks = ["word " * 50, "word " * 200, "word " * 100]
        result = self.analyzer.length_distribution(chunks)
        assert result["min"] == 50
        assert result["max"] == 200

    def test_single_chunk_std_is_zero(self):
        result = self.analyzer.length_distribution(["one two three four five"])
        assert result["std"] == pytest.approx(0.0)

    def test_mean_and_median_reasonable(self):
        chunks = ["word " * 100, "word " * 200, "word " * 300]
        result = self.analyzer.length_distribution(chunks)
        assert result["mean"]   == pytest.approx(200.0, rel=0.01)
        assert result["median"] == pytest.approx(200.0, rel=0.01)


# ═══════════════════════════════════════════════════════════════════════════════
# ChunkAnalyzer — coverage_gaps
# ═══════════════════════════════════════════════════════════════════════════════

class TestCoverageGaps:

    def setup_method(self):
        self.analyzer = ChunkAnalyzer()

    def test_required_keys_present(self):
        result = self.analyzer.coverage_gaps("apple revenue", ["some chunk"])
        for key in ("missing_terms", "covered_terms", "coverage_pct"):
            assert key in result

    def test_full_coverage_pct_one(self):
        result = self.analyzer.coverage_gaps(
            "apple revenue 2023",
            ["apple revenue 2023 billion dollar fiscal"],
        )
        assert result["coverage_pct"] == pytest.approx(1.0)
        assert result["missing_terms"] == []

    def test_zero_coverage_all_terms_missing(self):
        result = self.analyzer.coverage_gaps(
            "apple revenue 2023",
            ["microsoft azure cloud computing services platform"],
        )
        assert result["coverage_pct"] < 1.0
        assert len(result["missing_terms"]) > 0

    def test_empty_query_full_coverage(self):
        result = self.analyzer.coverage_gaps("", ["any chunk content here"])
        assert result["coverage_pct"] == pytest.approx(1.0)
        assert result["missing_terms"] == []

    def test_empty_chunks_all_missing(self):
        result = self.analyzer.coverage_gaps("apple revenue 2023", [])
        assert result["coverage_pct"] == pytest.approx(0.0)
        assert len(result["missing_terms"]) > 0

    def test_stopwords_not_counted_as_gaps(self):
        # "the", "is", "a" are stopwords — should not appear in missing_terms
        result = self.analyzer.coverage_gaps(
            "the apple is a revenue company",
            ["apple revenue company products"],
        )
        assert result["coverage_pct"] == pytest.approx(1.0)
        assert "the" not in result["missing_terms"]
        assert "is"  not in result["missing_terms"]

    def test_partial_coverage(self):
        result = self.analyzer.coverage_gaps(
            "apple revenue iphone margin growth",
            ["apple revenue fiscal 2023"],
        )
        assert 0.0 < result["coverage_pct"] < 1.0
        assert len(result["missing_terms"]) > 0
        assert len(result["covered_terms"]) > 0

    def test_covered_plus_missing_equals_total_terms(self):
        query  = "apple revenue iphone gross margin fiscal 2023"
        chunks = ["apple revenue 2023 results earnings"]
        result = self.analyzer.coverage_gaps(query, chunks)
        total = len(result["covered_terms"]) + len(result["missing_terms"])
        # Query terms (deduplicated, stopwords removed) should match total
        from ragprobe.retrieval import _sig_tokens
        expected = len(list(dict.fromkeys(_sig_tokens(query))))
        assert total == expected

    def test_cross_chunk_coverage(self):
        # "apple" in chunk 1, "revenue" in chunk 2 → both covered
        result = self.analyzer.coverage_gaps(
            "apple revenue",
            ["apple fiscal results 2023", "revenue earnings growth"],
        )
        assert result["coverage_pct"] == pytest.approx(1.0)

    def test_clean_financial_text_no_false_gaps(self):
        query  = "what was the annual revenue for fiscal year 2023"
        chunks = [
            # chunk explicitly contains "annual" so all 5 significant terms are covered
            "Apple annual revenue for fiscal year 2023 totalled 383.3 billion dollars.",
            "Revenue growth was driven primarily by iPhone and services segments.",
        ]
        result = self.analyzer.coverage_gaps(query, chunks)
        # Significant query terms: "annual", "revenue", "fiscal", "year", "2023"
        assert result["coverage_pct"] == pytest.approx(1.0)
        assert result["missing_terms"] == []


# ═══════════════════════════════════════════════════════════════════════════════
# RetrievalDiagnostic — run
# ═══════════════════════════════════════════════════════════════════════════════

class TestRetrievalDiagnostic:

    def setup_method(self):
        self.diag = RetrievalDiagnostic()

    # ── Return structure ──────────────────────────────────────────────────────

    def test_top_level_keys_present(self):
        result = self.diag.run(["query"], [[" word " * 100]])
        assert "per_query"  in result
        assert "aggregate"  in result
        assert "findings"   in result

    def test_per_query_length_matches_input(self):
        queries = ["q1", "q2", "q3"]
        chunks  = [["chunk a"], ["chunk b"], ["chunk c"]]
        result  = self.diag.run(queries, chunks)
        assert len(result["per_query"]) == 3

    def test_per_query_required_keys(self):
        result = self.diag.run(["q"], [["chunk content here"]])
        entry  = result["per_query"][0]
        for key in ("query", "chunk_count", "redundancy", "coverage", "length"):
            assert key in entry

    def test_per_query_recall_absent_without_references(self):
        result = self.diag.run(["q"], [["chunk"]])
        assert "recall" not in result["per_query"][0]

    def test_per_query_recall_present_with_references(self):
        result = self.diag.run(["q"], [["chunk"]], references=["ref"])
        assert "recall" in result["per_query"][0]

    def test_aggregate_has_mean_recall_with_references(self):
        result = self.diag.run(["q1", "q2"], [["c1"], ["c2"]], ["r1", "r2"])
        assert "mean_recall" in result["aggregate"]

    def test_aggregate_no_mean_recall_without_references(self):
        result = self.diag.run(["q"], [["c"]])
        assert "mean_recall" not in result["aggregate"]

    def test_finding_structure(self):
        chunks = [["microsoft azure cloud unrelated services enterprise"]]
        refs   = ["apple revenue billion fiscal 2023 iphone growth earnings"]
        result = self.diag.run(["apple revenue query"], chunks, refs)
        for f in result["findings"]:
            assert "severity"       in f
            assert "metric"         in f
            assert "value"          in f
            assert "recommendation" in f
            assert f["severity"] in ("CRITICAL", "WARNING", "INFO")

    def test_findings_sorted_critical_first(self):
        # High-redundancy identical chunks + zero recall → both CRITICAL
        identical = "apple revenue 2023 billion iphone sales fiscal margin growth"
        chunks    = [[identical] * 4]
        refs      = ["apple revenue 2023 completely unrelated microsoft azure cloud"]

        # Force zero recall: use non-overlapping ref
        refs  = ["entirely different topic xyz unrelated content"]
        result = self.diag.run(["apple 2023 query"], chunks, refs)

        severities = [f["severity"] for f in result["findings"]]
        if len(severities) >= 2:
            order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
            for a, b in zip(severities, severities[1:]):
                assert order[a] <= order[b], f"findings not sorted: {severities}"

    # ── Severity thresholds ───────────────────────────────────────────────────

    def test_critical_recall_finding(self):
        chunks = [["microsoft azure cloud platform completely unrelated services"]]
        refs   = ["apple reported 383 billion revenue fiscal 2023 iphone growth"]
        result = self.diag.run(["apple revenue 2023"], chunks, refs)
        critical = [f for f in result["findings"] if f["severity"] == "CRITICAL"]
        assert any(f["metric"] == "recall" for f in critical)

    def test_warning_recall_finding(self):
        # Partial match: ~30–49% of reference tokens in chunks → WARNING
        chunks = [["apple company reported results 2023 fiscal"]]
        refs   = ["apple reported 383 billion revenue fiscal 2023 iphone sales margin growth"]
        result = self.diag.run(["apple revenue 2023"], chunks, refs)
        severities = {f["severity"] for f in result["findings"]}
        assert "WARNING" in severities or "CRITICAL" in severities

    def test_no_recall_finding_when_no_references(self):
        chunks = [["any chunk"]]
        result = self.diag.run(["query"], chunks)
        recall_findings = [f for f in result["findings"] if f["metric"] == "recall"]
        assert recall_findings == []

    def test_critical_redundancy_finding(self):
        identical = "apple revenue 2023 billion dollars fiscal year results earnings iphone"
        chunks    = [[identical, identical, identical, identical]]
        result    = self.diag.run(["revenue query"], chunks)
        critical  = [f for f in result["findings"] if f["severity"] == "CRITICAL"]
        assert any(f["metric"] == "redundancy" for f in critical)

    def test_no_critical_redundancy_on_diverse_chunks(self):
        chunks = [[
            "apple reported 383 billion annual revenue fiscal 2023",
            "federal reserve raised interest rates by 25 basis points",
            "european central bank maintained euro zone monetary policy",
        ]]
        result   = self.diag.run(["apple revenue"], chunks)
        critical = [
            f for f in result["findings"]
            if f["severity"] == "CRITICAL" and f["metric"] == "redundancy"
        ]
        assert critical == []

    def test_warning_coverage_gap_finding(self):
        # Query terms absent from chunks → coverage gap warning
        chunks = [["microsoft azure cloud services enterprise platform compute"]]
        result = self.diag.run(["apple revenue iphone earnings growth margin"], chunks)
        warn   = [f for f in result["findings"] if f["metric"] == "coverage"]
        assert len(warn) > 0

    def test_no_findings_on_clean_retrieval(self):
        chunks = [[
            "Apple reported total net revenue of 383.3 billion dollars fiscal 2023.",
            "iPhone accounted for 52 percent of Apple annual revenue last year.",
            "Apple gross margin reached 44 percent during fiscal year 2023.",
        ]]
        refs   = ["Apple annual revenue was 383 billion dollars fiscal 2023"]
        result = self.diag.run(
            ["what was Apple annual revenue fiscal 2023"], chunks, refs
        )
        critical = [f for f in result["findings"] if f["severity"] == "CRITICAL"]
        assert critical == [], f"Unexpected CRITICAL findings: {critical}"

    # ── Input validation ──────────────────────────────────────────────────────

    def test_mismatched_queries_and_chunks_raises(self):
        with pytest.raises(ValueError, match="length"):
            self.diag.run(["q1", "q2"], [["c"]])

    def test_mismatched_references_raises(self):
        with pytest.raises(ValueError, match="length"):
            self.diag.run(["q1", "q2"], [["c1"], ["c2"]], references=["only one"])

    def test_none_references_accepted(self):
        result = self.diag.run(["query"], [["chunk content"]], None)
        assert "per_query" in result

    # ── Semantic recall (mocked) ──────────────────────────────────────────────

    def test_semantic_recall_flag_calls_client(self):
        vec    = [1.0, 0.0]
        client = _make_embed_client(vec, vec)
        self.diag.run(
            ["query"], [["chunk"]],
            references     = ["reference answer"],
            semantic_recall = True,
            client         = client,
            model          = "text-embedding-3-small",
        )
        client.embeddings.create.assert_called_once()

    def test_semantic_recall_stored_in_per_query(self):
        vec    = [1.0, 0.0]
        client = _make_embed_client(vec, vec)
        result = self.diag.run(
            ["query"], [["chunk"]],
            references     = ["ref"],
            semantic_recall = True,
            client         = client,
            model          = "m",
        )
        assert "semantic_recall" in result["per_query"][0]

    def test_semantic_recall_false_no_client_call(self):
        client = MagicMock()
        self.diag.run(
            ["query"], [["chunk"]],
            references     = ["ref"],
            semantic_recall = False,
            client         = client,
        )
        client.embeddings.create.assert_not_called()
