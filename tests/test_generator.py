"""Tests for TestGenerator — no API keys needed, all mocked."""

import json
import pytest
from unittest.mock import MagicMock, patch
from ragprobe.core.generator import TestGenerator, TestCase, TestSuite, AttackType


SAMPLE_CHUNK = """Remote Work Policy — Section 3.2
Employees may work remotely up to 3 days per week with manager approval.
Remote work is NOT permitted for employees on performance improvement plans.
All remote work must be conducted from within the United States."""


def _mock_openai_response(cases: list[dict]) -> MagicMock:
    msg         = MagicMock()
    msg.content = json.dumps({"cases": cases})
    choice      = MagicMock()
    choice.message = msg
    resp        = MagicMock()
    resp.choices = [choice]
    return resp


FAKE_CASES = [
    {
        "question":          "Can employees on PIPs work remotely?",
        "attack_type":       "negation",
        "expected_behavior": "Say no — PIPs explicitly block remote work",
        "expected_answer":   "No",
    },
    {
        "question":          "What is the company parental leave policy?",
        "attack_type":       "out_of_scope",
        "expected_behavior": "Say it doesn't know — not in this chunk",
        "expected_answer":   None,
    },
]


class TestTestGenerator:

    def test_init_raises_without_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch("dotenv.load_dotenv"):
                with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                    TestGenerator(judge="openai")

    def test_generate_returns_test_suite(self):
        gen          = TestGenerator.__new__(TestGenerator)
        gen.judge    = "openai"
        gen.model    = "gpt-4o-mini"
        gen._client  = MagicMock()
        gen._client.chat.completions.create.return_value = _mock_openai_response(FAKE_CASES)

        suite = gen.generate(documents=[SAMPLE_CHUNK], n_cases=2)

        assert isinstance(suite, TestSuite)
        assert suite.total == 2
        assert len(suite.cases) == 2

    def test_parse_response_maps_attack_types(self):
        gen       = TestGenerator.__new__(TestGenerator)
        gen.judge = "openai"

        raw   = json.dumps(FAKE_CASES)
        cases = gen._parse_response(raw, SAMPLE_CHUNK, start_id=0)

        assert cases[0].attack_type == AttackType.NEGATION
        assert cases[1].attack_type == AttackType.OUT_OF_SCOPE

    def test_parse_response_handles_malformed_json(self):
        gen       = TestGenerator.__new__(TestGenerator)
        gen.judge = "openai"

        result = gen._parse_response("not json at all {{{", SAMPLE_CHUNK, 0)
        assert result == []

    def test_test_case_ids_are_unique(self):
        gen         = TestGenerator.__new__(TestGenerator)
        gen.judge   = "openai"
        gen.model   = "gpt-4o-mini"
        gen._client = MagicMock()
        gen._client.chat.completions.create.return_value = _mock_openai_response(FAKE_CASES)

        suite = gen.generate(documents=[SAMPLE_CHUNK, SAMPLE_CHUNK], n_cases=4)
        ids   = [c.id for c in suite.cases]
        assert len(ids) == len(set(ids)), "Duplicate TestCase IDs found"

    def test_filter_by_attack_type(self):
        gen         = TestGenerator.__new__(TestGenerator)
        gen.judge   = "openai"
        gen.model   = "gpt-4o-mini"
        gen._client = MagicMock()
        gen._client.chat.completions.create.return_value = _mock_openai_response(FAKE_CASES)

        suite    = gen.generate(documents=[SAMPLE_CHUNK], n_cases=2)
        negation = suite.filter(AttackType.NEGATION)
        assert all(c.attack_type == AttackType.NEGATION for c in negation)
