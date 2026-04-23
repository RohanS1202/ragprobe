"""
rag_evaluator.py — High-level facade that combines TestGenerator and Scorer.

Usage::

    from ragprobe import RAGEvaluator

    evaluator = RAGEvaluator(judge="openai")
    suite     = evaluator.generate_tests(documents=docs, n_cases=10)
    results   = evaluator.evaluate(pipeline=my_pipeline, test_suite=suite)
    evaluator.print_summary(results)
    summary   = evaluator.summarise(results)
"""

from __future__ import annotations

from typing import Callable, Optional

from ragprobe.core.generator import AttackType, TestSuite, TestGenerator
from ragprobe.core.scorer import EvalResult, EvalSummary, Scorer


class RAGEvaluator:
    """
    Facade over TestGenerator and Scorer.

    Parameters
    ----------
    judge : str
        "openai" or "anthropic"
    model : str, optional
        Override the default model for both generator and scorer.
    api_key : str, optional
        API key (reads from env if omitted).
    """

    def __init__(
        self,
        judge:   str = "openai",
        model:   Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self._generator = TestGenerator(judge=judge, model=model, api_key=api_key)
        self._scorer    = Scorer(judge=judge, model=model, api_key=api_key)

    def generate_tests(
        self,
        documents:    list[str],
        n_cases:      int = 20,
        attack_types: Optional[list[AttackType]] = None,
        suite_name:   str = "eval_suite",
    ) -> TestSuite:
        """Generate an adversarial test suite from document chunks."""
        return self._generator.generate(
            documents    = documents,
            n_cases      = n_cases,
            attack_types = attack_types,
            suite_name   = suite_name,
        )

    def evaluate(
        self,
        pipeline:   Callable[[str], tuple[str, list[str]]],
        test_suite: TestSuite,
        metrics:    Optional[list[str]] = None,
    ) -> list[EvalResult]:
        """Run pipeline on every test case and score each output."""
        return self._scorer.score_batch(
            pipeline   = pipeline,
            test_cases = test_suite.cases,
            metrics    = metrics,
        )

    def print_summary(self, results: list[EvalResult]) -> None:
        """Print a rich-formatted evaluation summary to the terminal."""
        self._scorer.print_summary(results)

    def summarise(self, results: list[EvalResult]) -> EvalSummary:
        """Return aggregate stats across all results."""
        return self._scorer.summarise(results)
