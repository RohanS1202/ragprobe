"""
generator.py — Adversarial test case generator for RAG pipelines.

Attack types:
  - NEGATION:      Questions where the correct answer contradicts naive assumptions
  - OUT_OF_SCOPE:  Plausible questions the corpus cannot answer
  - AMBIGUOUS:     Questions with multiple valid interpretations
  - CONFLICTING:   Questions where two chunks give different answers
  - EDGE_CASE:     Boundary conditions — dates, numbers, names
  - HYPOTHETICAL:  What-if questions requiring reasoning beyond the docs
"""

from __future__ import annotations

import json
import math
import random
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class AttackType(str, Enum):
    NEGATION     = "negation"
    OUT_OF_SCOPE = "out_of_scope"
    AMBIGUOUS    = "ambiguous"
    CONFLICTING  = "conflicting"
    EDGE_CASE    = "edge_case"
    HYPOTHETICAL = "hypothetical"


class TestCase(BaseModel):
    """A single adversarial test case."""
    id:                str
    question:          str
    attack_type:       AttackType
    source_chunk:      str
    expected_behavior: str
    expected_answer:   Optional[str] = None
    metadata:          dict = Field(default_factory=dict)


class TestSuite(BaseModel):
    """A collection of test cases generated from a document corpus."""
    name:    str
    cases:   list[TestCase]
    total:   int
    by_type: dict[str, int]

    def __len__(self) -> int:
        return len(self.cases)

    def filter(self, attack_type: AttackType) -> list[TestCase]:
        return [c for c in self.cases if c.attack_type == attack_type]


# ── Prompts ────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert at adversarial testing of AI systems.
Your job is to generate test cases that expose failure modes in RAG pipelines.
Always return valid JSON matching the exact schema requested. No preamble, no markdown."""


def _build_generation_prompt(
    chunk: str,
    attack_types: list[AttackType],
    n_per_type: int,
) -> str:
    type_descriptions = {
        AttackType.NEGATION: (
            "Questions where the correct answer explicitly contradicts "
            "something a naive model might assume."
        ),
        AttackType.OUT_OF_SCOPE: (
            "Plausible, on-topic questions this specific chunk cannot answer. "
            "The RAG system should say it doesn't know, not hallucinate."
        ),
        AttackType.AMBIGUOUS: (
            "Questions with multiple valid interpretations. "
            "A good system should ask for clarification or hedge its answer."
        ),
        AttackType.CONFLICTING: (
            "Questions where the answer might differ depending on which "
            "part of the document you focus on."
        ),
        AttackType.EDGE_CASE: (
            "Boundary conditions: exact dates, maximum/minimum values, "
            "specific version numbers. These catch hallucinations on precise details."
        ),
        AttackType.HYPOTHETICAL: (
            "What-if questions requiring reasoning beyond the document. "
            "A good system grounds its answer and flags when speculating."
        ),
    }

    types_block = "\n".join(
        f"- {t.value}: {type_descriptions[t]}"
        for t in attack_types
    )
    n_total = n_per_type * len(attack_types)

    return (
        f"Given this document chunk:\n\n---\n{chunk}\n---\n\n"
        f"Generate {n_per_type} test case(s) for EACH of these {len(attack_types)} attack types:\n"
        f"{types_block}\n\n"
        f"Return a JSON object with key 'cases' containing exactly {n_total} items.\n"
        f"Each item must have: question (str), attack_type (str), "
        f"expected_behavior (str), expected_answer (str or null).\n"
        f'Schema: {{"cases": [{{"question":"...","attack_type":"...","expected_behavior":"...","expected_answer":null}}]}}'
    )


# ── Main class ─────────────────────────────────────────────────────────────

class TestGenerator:
    """
    Generates adversarial test cases from document chunks.

    Args:
        judge:   "openai" | "anthropic"
        model:   specific model string (optional)
        api_key: API key (reads from env if not provided)

    Example:
        gen = TestGenerator(judge="openai")
        suite = gen.generate(documents=my_chunks, n_cases=20)
    """

    _DEFAULTS = {
        "openai":    "gpt-4o-mini",
        "anthropic": "claude-haiku-4-5-20251001",
    }

    def __init__(
        self,
        judge:   str = "openai",
        model:   Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.judge   = judge.lower()
        self.model   = model or self._DEFAULTS[self.judge]
        self._client = self._init_client(api_key)

    def generate(
        self,
        documents:    list[str],
        n_cases:      int = 20,
        attack_types: Optional[list[AttackType]] = None,
        suite_name:   str = "eval_suite",
    ) -> TestSuite:
        """
        Generate a test suite from a list of document chunks.

        Args:
            documents:    List of text chunks from your corpus
            n_cases:      Approximate total test cases to generate
            attack_types: Which attack types to include (all 6 by default)
            suite_name:   Name for this suite

        Returns:
            TestSuite with generated test cases
        """
        if attack_types is None:
            attack_types = list(AttackType)

        n_types  = len(attack_types)
        n_chunks = max(1, math.ceil(n_cases / n_types))
        n_chunks = min(n_chunks, len(documents))

        sampled_chunks = random.sample(documents, k=n_chunks)

        all_cases: list[TestCase] = []
        case_id = 0

        for chunk in sampled_chunks:
            raw    = self._call_llm(chunk, attack_types, n_per_type=1)
            parsed = self._parse_response(raw, chunk, case_id)
            all_cases.extend(parsed)
            case_id += len(parsed)

        by_type = {t.value: 0 for t in AttackType}
        for c in all_cases:
            by_type[c.attack_type.value] += 1

        return TestSuite(
            name    = suite_name,
            cases   = all_cases,
            total   = len(all_cases),
            by_type = {k: v for k, v in by_type.items() if v > 0},
        )

    def _call_llm(
        self,
        chunk:        str,
        attack_types: list[AttackType],
        n_per_type:   int,
    ) -> str:
        prompt = _build_generation_prompt(chunk, attack_types, n_per_type)
        if self.judge == "openai":
            return self._call_openai(prompt)
        elif self.judge == "anthropic":
            return self._call_anthropic(prompt)
        raise ValueError(f"Unknown judge: {self.judge}")

    def _call_openai(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model           = self.model,
            messages        = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature     = 0.8,
            response_format = {"type": "json_object"},
        )
        raw    = response.choices[0].message.content
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            for key in ("cases", "test_cases", "tests", "results"):
                if key in parsed and isinstance(parsed[key], list):
                    return json.dumps(parsed[key])
            for val in parsed.values():
                if isinstance(val, list):
                    return json.dumps(val)
        if isinstance(parsed, list):
            return json.dumps(parsed)
        return json.dumps([])

    def _call_anthropic(self, prompt: str) -> str:
        response = self._client.messages.create(
            model      = self.model,
            max_tokens = 2048,
            system     = _SYSTEM_PROMPT,
            messages   = [{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                for key in ("cases", "test_cases", "tests"):
                    if key in parsed and isinstance(parsed[key], list):
                        return json.dumps(parsed[key])
                for val in parsed.values():
                    if isinstance(val, list):
                        return json.dumps(val)
            return json.dumps(parsed) if isinstance(parsed, list) else json.dumps([])
        except json.JSONDecodeError:
            import re
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            return match.group() if match else json.dumps([])

    def _parse_response(
        self,
        raw:      str,
        chunk:    str,
        start_id: int,
    ) -> list[TestCase]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                print("[ragprobe] Warning: could not parse LLM response, skipping chunk.")
                return []

        if not isinstance(data, list):
            data = [data]

        attack_map = {
            "out_of_scope": AttackType.OUT_OF_SCOPE,
            "out-of-scope": AttackType.OUT_OF_SCOPE,
            "negation":     AttackType.NEGATION,
            "ambiguous":    AttackType.AMBIGUOUS,
            "conflicting":  AttackType.CONFLICTING,
            "edge_case":    AttackType.EDGE_CASE,
            "edge-case":    AttackType.EDGE_CASE,
            "hypothetical": AttackType.HYPOTHETICAL,
        }

        cases = []
        for i, item in enumerate(data):
            try:
                attack_str  = item.get("attack_type", "").lower().replace(" ", "_")
                attack_type = attack_map.get(attack_str, AttackType.EDGE_CASE)
                cases.append(TestCase(
                    id                = f"tc_{start_id + i:04d}",
                    question          = item["question"],
                    attack_type       = attack_type,
                    source_chunk      = chunk[:500],
                    expected_behavior = item.get("expected_behavior", ""),
                    expected_answer   = item.get("expected_answer"),
                ))
            except (KeyError, ValueError) as e:
                print(f"[ragprobe] Warning: skipping malformed test case: {e}")
        return cases

    def _init_client(self, api_key: Optional[str]):
        import os
        from dotenv import load_dotenv
        load_dotenv()

        if self.judge == "openai":
            from openai import OpenAI
            key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
            if not key:
                raise ValueError(
                    "OpenAI API key not found. Set OPENAI_API_KEY in your .env file."
                )
            return OpenAI(api_key=key)

        elif self.judge == "anthropic":
            from anthropic import Anthropic
            key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError(
                    "Anthropic API key not found. Set ANTHROPIC_API_KEY in your .env file."
                )
            return Anthropic(api_key=key)

        raise ValueError(f"Unknown judge '{self.judge}'. Choose 'openai' or 'anthropic'.")
