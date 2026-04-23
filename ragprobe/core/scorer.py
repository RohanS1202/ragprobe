"""
scorer.py — LLM-as-judge scoring for RAG pipeline outputs.

Three core metrics:
  FAITHFULNESS   — Is every claim grounded in the retrieved context?
  RELEVANCE      — Does the answer address what was actually asked?
  CONTEXT_RECALL — Did the retriever surface the right chunks?
"""

from __future__ import annotations

import json
from typing import Callable, Optional
from pydantic import BaseModel

from ragprobe.core.generator import TestCase


class MetricScore(BaseModel):
    metric:     str
    score:      float
    violations: list[str]
    reasoning:  str
    passed:     bool


class EvalResult(BaseModel):
    test_case:      TestCase
    question:       str
    answer:         str
    contexts:       list[str]
    faithfulness:   Optional[MetricScore] = None
    relevance:      Optional[MetricScore] = None
    context_recall: Optional[MetricScore] = None
    overall_score:  float = 0.0
    passed:         bool  = False


class EvalSummary(BaseModel):
    total_cases:             int
    passed:                  int
    failed:                  int
    pass_rate:               float
    avg_faithfulness:        float
    avg_relevance:           float
    avg_context_recall:      float
    failures_by_attack_type: dict[str, int]
    worst_cases:             list[EvalResult]


# ── Scoring prompts ────────────────────────────────────────────────────────

_FAITHFULNESS_PROMPT = """You are evaluating whether an AI answer is faithful to its source context.

QUESTION: {question}

RETRIEVED CONTEXT:
{context}

AI ANSWER:
{answer}

Score FAITHFULNESS 0.0-1.0: does every factual claim in the answer appear in the context?

Penalise for:
- Facts stated confidently that are NOT in the context (hallucinations)
- Numbers, dates, or names that differ from the context
- Logical inferences that go beyond what the context supports

Do NOT penalise for:
- Reasonable paraphrasing of context content
- General framing language that is not a core factual claim

Return JSON: {{"score": <float>, "violations": [<list of problematic claims>], "reasoning": "<1-2 sentences>"}}"""


_RELEVANCE_PROMPT = """You are evaluating whether an AI answer is relevant to the question asked.

QUESTION: {question}

RETRIEVED CONTEXT:
{context}

AI ANSWER:
{answer}

Score RELEVANCE 0.0-1.0: does the answer directly address what was asked?

Penalise for:
- Answering a different question than what was asked
- Excessive hedging that avoids the actual answer
- Generic boilerplate that doesn't address the specific question
- Refusing to answer when the context clearly contains the information

Return JSON: {{"score": <float>, "violations": [<list of relevance problems>], "reasoning": "<1-2 sentences>"}}"""


_CONTEXT_RECALL_PROMPT = """You are evaluating whether an AI retriever returned useful context.

QUESTION: {question}

EXPECTED ANSWER / BEHAVIOR: {expected_behavior}

RETRIEVED CONTEXT CHUNKS:
{context}

Score CONTEXT RECALL 0.0-1.0: did the retrieved chunks contain the information needed?

Penalise for:
- Retrieved chunks that don't contain the answer at all
- Key information missing from all retrieved chunks
- Chunks topically related but missing the specific detail needed

Return JSON: {{"score": <float>, "violations": [<list of gaps>], "reasoning": "<1-2 sentences>"}}"""


class Scorer:
    """
    Scores RAG pipeline outputs using LLM-as-judge.

    Args:
        judge:      "openai" | "anthropic"
        model:      specific model string (optional)
        thresholds: per-metric pass/fail cutoffs
        api_key:    API key (reads from env if not provided)

    Example:
        scorer = Scorer(judge="openai")
        result = scorer.score(
            test_case=tc,
            answer="No, alcohol is not reimbursable.",
            contexts=["Alcohol is NOT reimbursable under any circumstances."],
        )
        print(result.overall_score)
    """

    _DEFAULTS = {
        "openai":    "gpt-4o-mini",
        "anthropic": "claude-haiku-4-5-20251001",
    }

    DEFAULT_THRESHOLDS = {
        "faithfulness":   0.80,
        "relevance":      0.75,
        "context_recall": 0.70,
    }

    def __init__(
        self,
        judge:      str = "openai",
        model:      Optional[str] = None,
        thresholds: Optional[dict[str, float]] = None,
        api_key:    Optional[str] = None,
    ):
        self.judge      = judge.lower()
        self.model      = model or self._DEFAULTS[self.judge]
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS.copy()
        self._client    = self._init_client(api_key)

    def score(
        self,
        test_case: TestCase,
        answer:    str,
        contexts:  list[str],
        metrics:   Optional[list[str]] = None,
    ) -> EvalResult:
        """Score a single RAG output on the requested metrics."""
        if metrics is None:
            metrics = ["faithfulness", "relevance", "context_recall"]

        context_str = self._format_contexts(contexts)
        result = EvalResult(
            test_case = test_case,
            question  = test_case.question,
            answer    = answer,
            contexts  = contexts,
        )

        scores = []

        if "faithfulness" in metrics:
            result.faithfulness = self._score_metric(
                prompt    = _FAITHFULNESS_PROMPT.format(
                    question = test_case.question,
                    context  = context_str,
                    answer   = answer,
                ),
                metric    = "faithfulness",
                threshold = self.thresholds["faithfulness"],
            )
            scores.append(result.faithfulness.score)

        if "relevance" in metrics:
            result.relevance = self._score_metric(
                prompt    = _RELEVANCE_PROMPT.format(
                    question = test_case.question,
                    context  = context_str,
                    answer   = answer,
                ),
                metric    = "relevance",
                threshold = self.thresholds["relevance"],
            )
            scores.append(result.relevance.score)

        if "context_recall" in metrics:
            result.context_recall = self._score_metric(
                prompt    = _CONTEXT_RECALL_PROMPT.format(
                    question          = test_case.question,
                    expected_behavior = test_case.expected_behavior,
                    context           = context_str,
                ),
                metric    = "context_recall",
                threshold = self.thresholds["context_recall"],
            )
            scores.append(result.context_recall.score)

        result.overall_score = round(sum(scores) / len(scores), 3) if scores else 0.0

        # Each metric must pass its own threshold independently
        result.passed = all([
            (result.faithfulness   is None or result.faithfulness.passed),
            (result.relevance      is None or result.relevance.passed),
            (result.context_recall is None or result.context_recall.passed),
        ])

        return result

    def score_batch(
        self,
        pipeline:   Callable[[str], tuple[str, list[str]]],
        test_cases: list[TestCase],
        metrics:    Optional[list[str]] = None,
    ) -> list[EvalResult]:
        """Run pipeline on every test case and score each output."""
        from rich.progress import track

        results = []
        for tc in track(test_cases, description="Evaluating..."):
            try:
                answer, contexts = pipeline(tc.question)
                result = self.score(tc, answer, contexts, metrics)
                results.append(result)
            except Exception as e:
                print(f"[ragprobe] Pipeline error on '{tc.question[:50]}...': {e}")
        return results

    def summarise(self, results: list[EvalResult]) -> EvalSummary:
        """Compute aggregate stats across all results."""
        if not results:
            return EvalSummary(
                total_cases=0, passed=0, failed=0, pass_rate=0.0,
                avg_faithfulness=0.0, avg_relevance=0.0, avg_context_recall=0.0,
                failures_by_attack_type={}, worst_cases=[],
            )

        passed = sum(1 for r in results if r.passed)

        def avg(attr: str) -> float:
            vals = [getattr(r, attr).score for r in results if getattr(r, attr) is not None]
            return round(sum(vals) / len(vals), 3) if vals else 0.0

        failures_by_type: dict[str, int] = {}
        for r in results:
            if not r.passed:
                t = r.test_case.attack_type.value
                failures_by_type[t] = failures_by_type.get(t, 0) + 1

        worst = sorted(results, key=lambda r: r.overall_score)[:5]

        return EvalSummary(
            total_cases             = len(results),
            passed                  = passed,
            failed                  = len(results) - passed,
            pass_rate               = round(passed / len(results), 3),
            avg_faithfulness        = avg("faithfulness"),
            avg_relevance           = avg("relevance"),
            avg_context_recall      = avg("context_recall"),
            failures_by_attack_type = failures_by_type,
            worst_cases             = worst,
        )

    def print_summary(self, results: list[EvalResult]) -> None:
        """Print a rich-formatted summary to the terminal."""
        from rich.console import Console
        from rich.table   import Table
        from rich         import box

        console = Console()
        summary = self.summarise(results)
        console.print("\n[bold]ragprobe evaluation summary[/bold]\n")

        table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        table.add_column("Metric",  style="dim")
        table.add_column("Score",   justify="right")
        table.add_column("Status",  justify="center")

        def fmt(score: float, threshold: float) -> tuple[str, str]:
            color  = "green" if score >= threshold else "red"
            symbol = "✓" if score >= threshold else "✗"
            return f"[{color}]{score:.2f}[/{color}]", f"[{color}]{symbol}[/{color}]"

        table.add_row("Faithfulness",   *fmt(summary.avg_faithfulness,   self.thresholds["faithfulness"]))
        table.add_row("Relevance",      *fmt(summary.avg_relevance,      self.thresholds["relevance"]))
        table.add_row("Context recall", *fmt(summary.avg_context_recall, self.thresholds["context_recall"]))
        console.print(table)

        color = "green" if summary.pass_rate >= 0.8 else "yellow" if summary.pass_rate >= 0.6 else "red"
        console.print(
            f"Pass rate: [{color}]{summary.passed}/{summary.total_cases} "
            f"({summary.pass_rate*100:.0f}%)[/{color}]\n"
        )

        if summary.failures_by_attack_type:
            console.print("[bold]Failures by attack type:[/bold]")
            for attack, count in sorted(
                summary.failures_by_attack_type.items(), key=lambda x: -x[1]
            ):
                console.print(f"  {attack:<20} {count} failures")
            console.print()

        if summary.worst_cases:
            console.print("[bold]Worst performing cases:[/bold]")
            for r in summary.worst_cases:
                console.print(
                    f"  [{r.test_case.attack_type.value}] "
                    f"score={r.overall_score:.2f}  "
                    f"Q: {r.question[:70]}..."
                )
        console.print()

    def _score_metric(
        self,
        prompt:    str,
        metric:    str,
        threshold: float,
    ) -> MetricScore:
        raw    = self._call_llm(prompt)
        parsed = self._parse_score(raw)
        score  = max(0.0, min(1.0, float(parsed.get("score", 0.5))))
        return MetricScore(
            metric     = metric,
            score      = round(score, 3),
            violations = parsed.get("violations", []),
            reasoning  = parsed.get("reasoning", ""),
            passed     = score >= threshold,
        )

    def _call_llm(self, prompt: str) -> str:
        system = (
            "You are a precise AI evaluator. "
            "Return only valid JSON with the exact keys requested. "
            "No preamble, no markdown."
        )
        if self.judge == "openai":
            resp = self._client.chat.completions.create(
                model           = self.model,
                messages        = [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                temperature     = 0.0,
                response_format = {"type": "json_object"},
            )
            return resp.choices[0].message.content

        elif self.judge == "anthropic":
            resp = self._client.messages.create(
                model      = self.model,
                max_tokens = 512,
                system     = system,
                messages   = [{"role": "user", "content": prompt}],
            )
            return resp.content[0].text

        raise ValueError(f"Unknown judge: {self.judge}")

    def _parse_score(self, raw: str) -> dict:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return {"score": 0.5, "violations": [], "reasoning": "Parse error"}

    @staticmethod
    def _format_contexts(contexts: list[str]) -> str:
        return "\n\n---\n\n".join(
            f"[Chunk {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)
        )

    def _init_client(self, api_key: Optional[str]):
        import os
        from dotenv import load_dotenv
        load_dotenv()

        if self.judge == "openai":
            from openai import OpenAI
            key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
            if not key:
                raise ValueError("Set OPENAI_API_KEY in your .env file.")
            return OpenAI(api_key=key)

        elif self.judge == "anthropic":
            from anthropic import Anthropic
            key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError("Set ANTHROPIC_API_KEY in your .env file.")
            return Anthropic(api_key=key)

        raise ValueError(f"Unknown judge '{self.judge}'.")
