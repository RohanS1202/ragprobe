"""
evaluator.py — High-level API that ties generator + scorer together.
"""

from __future__ import annotations

from typing import Callable, Optional

from ragprobe.core.generator import TestGenerator, TestCase, TestSuite, AttackType
from ragprobe.core.scorer    import Scorer, EvalResult, EvalSummary


class RAGEvaluator:
    """
    High-level evaluator for RAG pipelines.

    Combines test generation and scoring in one clean interface.

    Args:
        judge:      "openai" | "anthropic"
        model:      specific model string (optional)
        thresholds: dict of metric -> pass/fail threshold
        api_key:    API key (reads from env if not provided)

    Example:
        evaluator = RAGEvaluator(judge="openai")

        suite = evaluator.generate_tests(
            documents=my_chunks,
            n_cases=30,
        )

        results = evaluator.evaluate(
            pipeline=my_rag_fn,
            test_suite=suite,
        )

        evaluator.print_summary(results)
    """

    def __init__(
        self,
        judge:      str = "openai",
        model:      Optional[str] = None,
        thresholds: Optional[dict[str, float]] = None,
        api_key:    Optional[str] = None,
    ):
        self.judge   = judge
        self.model   = model
        self.api_key = api_key

        self._generator = TestGenerator(judge=judge, model=model, api_key=api_key)
        self._scorer    = Scorer(judge=judge, model=model,
                                 thresholds=thresholds, api_key=api_key)

    def generate_tests(
        self,
        documents:    list[str],
        n_cases:      int = 20,
        attack_types: Optional[list[AttackType]] = None,
        suite_name:   str = "eval_suite",
    ) -> TestSuite:
        """Generate adversarial test cases from your document corpus."""
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
        """
        Run the pipeline on every test case and score each output.

        Args:
            pipeline:   fn(query: str) -> (answer: str, contexts: list[str])
            test_suite: output of generate_tests()
            metrics:    which metrics to score (default: all three)
        """
        return self._scorer.score_batch(
            pipeline   = pipeline,
            test_cases = test_suite.cases,
            metrics    = metrics,
        )

    def summarise(self, results: list[EvalResult]) -> EvalSummary:
        return self._scorer.summarise(results)

    def print_summary(self, results: list[EvalResult]) -> None:
        self._scorer.print_summary(results)
