"""
ragprobe — Adversarial testing & evaluation for RAG pipelines and LLM agents.

Quick start:
    from ragprobe import RAGEvaluator

    evaluator = RAGEvaluator(judge="openai")

    suite = evaluator.generate_tests(
        documents=my_docs,
        n_cases=20
    )

    results = evaluator.evaluate(
        pipeline=my_rag_fn,
        test_suite=suite
    )

    evaluator.print_summary(results)
"""

from ragprobe.evaluator import RAGEvaluator
from ragprobe.core.generator import TestGenerator, TestCase, AttackType
from ragprobe.core.scorer import Scorer, EvalResult
from ragprobe.core.monitor import monitor

__version__ = "0.1.0"
__all__ = [
    "RAGEvaluator",
    "TestGenerator",
    "TestCase",
    "AttackType",
    "Scorer",
    "EvalResult",
    "monitor",
]
