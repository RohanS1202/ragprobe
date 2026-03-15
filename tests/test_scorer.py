"""Tests for Scorer — no API keys needed, all mocked."""

import json
import pytest
from unittest.mock import MagicMock
from ragprobe.core.generator import TestCase, AttackType
from ragprobe.core.scorer    import Scorer, EvalResult, MetricScore


def _make_test_case(question: str = "Is alcohol reimbursable?") -> TestCase:
    return TestCase(
        id                = "tc_0001",
        question          = question,
        attack_type       = AttackType.NEGATION,
        source_chunk      = "Alcohol is NOT reimbursable under any circumstances.",
        expected_behavior = "Say no — alcohol is explicitly excluded",
        expected_answer   = "No",
    )


def _build_scorer(score: float) -> Scorer:
    """Build a Scorer with a mocked LLM that always returns the given score."""
    scorer          = Scorer.__new__(Scorer)
    scorer.judge    = "openai"
    scorer.model    = "gpt-4o-mini"
    scorer.thresholds = Scorer.DEFAULT_THRESHOLDS.copy()
    scorer._client  = MagicMock()
    scorer._client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps({
            "score":      score,
            "violations": [],
            "reasoning":  "test",
        })))]
    )
    return scorer


class TestScorer:

    def test_score_returns_eval_result(self):
        scorer = _build_scorer(0.9)
        tc     = _make_test_case()
        result = scorer.score(tc, answer="No, alcohol is not reimbursable.",
                              contexts=["Alcohol is NOT reimbursable."])
        assert isinstance(result, EvalResult)
        assert 0.0 <= result.overall_score <= 1.0

    def test_each_metric_must_pass_its_own_threshold(self):
        """
        A result with faithfulness=0.30 must FAIL even if the average
        score across all metrics would be above the minimum threshold.
        This is the critical pass/fail logic regression test.
        """
        scorer = Scorer.__new__(Scorer)
        scorer.judge      = "openai"
        scorer.model      = "gpt-4o-mini"
        scorer.thresholds = Scorer.DEFAULT_THRESHOLDS.copy()

        def mock_score_metric(prompt, metric, threshold):
            if metric == "faithfulness":
                return MetricScore(metric=metric, score=0.30,
                                   violations=["hallucination"], reasoning="bad", passed=False)
            return MetricScore(metric=metric, score=0.95,
                               violations=[], reasoning="good", passed=True)

        scorer._score_metric    = mock_score_metric
        scorer._format_contexts = Scorer._format_contexts

        tc     = _make_test_case()
        result = scorer.score(tc, answer="test", contexts=["context"])

        # Average = (0.30 + 0.95 + 0.95) / 3 = 0.73, above min threshold 0.70
        # But faithfulness individually failed — so overall must be FAILED
        assert result.passed is False

    def test_perfect_scores_pass(self):
        scorer = _build_scorer(1.0)
        tc     = _make_test_case()
        result = scorer.score(tc, answer="No.", contexts=["Alcohol is NOT reimbursable."])
        assert result.passed is True
        assert result.overall_score == 1.0

    def test_zero_score_fails(self):
        scorer = _build_scorer(0.0)
        tc     = _make_test_case()
        result = scorer.score(tc, answer="Sure, alcohol is fine.",
                              contexts=["Alcohol is NOT reimbursable."])
        assert result.passed is False

    def test_summarise_totals(self):
        scorer  = _build_scorer(0.9)
        tc      = _make_test_case()
        r1      = scorer.score(tc, "No.", ["context"])
        r2      = scorer.score(tc, "No.", ["context"])
        summary = scorer.summarise([r1, r2])
        assert summary.total_cases == 2
        assert summary.passed + summary.failed == summary.total_cases

    def test_summarise_empty(self):
        scorer  = _build_scorer(0.9)
        summary = scorer.summarise([])
        assert summary.total_cases == 0
        assert summary.pass_rate   == 0.0

    def test_parse_score_handles_bad_json(self):
        scorer = Scorer.__new__(Scorer)
        result = scorer._parse_score("not json at all")
        assert "score" in result
        assert result["score"] == 0.5
