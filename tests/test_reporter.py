"""
tests/test_reporter.py — Unit tests for ragprobe/reporter.py.

Covers:
  - SessionReport construction, to_dict/from_dict round-trip, save/load persistence
  - JSONReporter: valid JSON output, findings sort order
  - HTMLReporter: self-contained HTML, structural guarantees, no external URLs
  - MarkdownReporter: emoji severity indicators, GFM tables
  - ReporterFactory: dispatch by format name, error on unknown format
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragprobe.reporter import (
    HTMLReporter,
    JSONReporter,
    MarkdownReporter,
    ReporterFactory,
    SessionReport,
)


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def make_report() -> SessionReport:
    """
    A fully-populated SessionReport with:
      - 2 queries
      - 1 CRITICAL finding, 1 WARNING finding, 1 INFO finding (in mixed order)
      - 1 safety event
      - realistic cost_summary dict
    """
    return SessionReport.new(
        model="gpt-4o-mini",
        total_queries=2,
        cost_summary={
            "total_tokens": 4_200,
            "total_cost_usd": 0.0042,
            "budget_used_pct": 21.0,
            "calls": 2,
        },
        retrieval_findings=[
            # Deliberately mixed order to test sorting
            {
                "severity": "WARNING",
                "metric": "redundancy",
                "value": 0.76,
                "recommendation": "Consider deduplicating chunks.",
            },
            {
                "severity": "CRITICAL",
                "metric": "recall",
                "value": 0.42,
                "recommendation": "Recall is critically low; add more context.",
            },
            {
                "severity": "INFO",
                "metric": "coverage",
                "value": 0.91,
                "recommendation": "Coverage looks good.",
            },
        ],
        per_query=[
            {
                "query": "What was AAPL revenue in FY2024?",
                "chunk_count": 3,
                "recall": 0.42,
                "redundancy": {"mean": 0.76, "max": 0.82, "worst_pair": None},
                "coverage": {
                    "coverage_pct": 0.91,
                    "missing_terms": ["quarterly"],
                    "covered_terms": ["aapl", "revenue", "fy2024"],
                },
                "length": {
                    "min": 18, "max": 55, "mean": 36.5,
                    "median": 35.0, "std": 12.3, "recommendation": None,
                },
            },
            {
                "query": "How did NVDA gross margin change year over year?",
                "chunk_count": 2,
                "recall": 0.88,
                "redundancy": {"mean": 0.11, "max": 0.11, "worst_pair": None},
                "coverage": {
                    "coverage_pct": 1.0,
                    "missing_terms": [],
                    "covered_terms": ["nvda", "gross", "margin"],
                },
                "length": {
                    "min": 30, "max": 60, "mean": 45.0,
                    "median": 45.0, "std": 15.0, "recommendation": None,
                },
            },
        ],
        aggregate={
            "mean_recall": 0.65,
            "mean_redundancy": 0.435,
            "mean_coverage_pct": 0.955,
        },
        safety_events=[
            {
                "type": "injection",
                "document_index": 3,
                "pattern": "ignore_all_previous",
                "risk_level": "HIGH",
                "message": "Prompt injection pattern detected in document 3.",
            }
        ],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SessionReport
# ═══════════════════════════════════════════════════════════════════════════════

class TestSessionReport:

    def test_new_sets_run_id(self, make_report):
        assert isinstance(make_report.run_id, str)
        assert len(make_report.run_id) == 36  # UUID4 string length

    def test_new_sets_timestamp(self, make_report):
        assert isinstance(make_report.timestamp, str)
        assert "T" in make_report.timestamp  # ISO format

    def test_new_sets_model(self, make_report):
        assert make_report.model == "gpt-4o-mini"

    def test_new_sets_total_queries(self, make_report):
        assert make_report.total_queries == 2

    def test_new_sets_ragprobe_version(self, make_report):
        assert isinstance(make_report.ragprobe_version, str)
        assert len(make_report.ragprobe_version) > 0

    def test_to_dict_has_all_keys(self, make_report):
        d = make_report.to_dict()
        for key in (
            "run_id", "timestamp", "model", "total_queries",
            "cost_summary", "retrieval_findings", "per_query",
            "aggregate", "safety_events", "ragprobe_version",
        ):
            assert key in d, f"missing key: {key}"

    def test_to_dict_is_json_serialisable(self, make_report):
        d = make_report.to_dict()
        # Should not raise
        text = json.dumps(d)
        assert len(text) > 0

    def test_from_dict_round_trip(self, make_report):
        d = make_report.to_dict()
        restored = SessionReport.from_dict(d)
        assert restored.run_id == make_report.run_id
        assert restored.model == make_report.model
        assert restored.total_queries == make_report.total_queries
        assert restored.timestamp == make_report.timestamp
        assert restored.ragprobe_version == make_report.ragprobe_version

    def test_from_dict_round_trip_findings(self, make_report):
        d = make_report.to_dict()
        restored = SessionReport.from_dict(d)
        assert len(restored.retrieval_findings) == len(make_report.retrieval_findings)

    def test_from_dict_round_trip_per_query(self, make_report):
        d = make_report.to_dict()
        restored = SessionReport.from_dict(d)
        assert len(restored.per_query) == len(make_report.per_query)

    def test_from_dict_round_trip_safety_events(self, make_report):
        d = make_report.to_dict()
        restored = SessionReport.from_dict(d)
        assert len(restored.safety_events) == 1
        assert restored.safety_events[0]["type"] == "injection"

    def test_to_dict_does_not_sort_findings(self, make_report):
        """to_dict preserves insertion order — sorting is the reporter's job."""
        d = make_report.to_dict()
        severities = [f["severity"] for f in d["retrieval_findings"]]
        # Fixture inserts WARNING first, then CRITICAL — order must be preserved
        assert severities[0] == "WARNING"
        assert severities[1] == "CRITICAL"

    def test_save_and_load_round_trip(self, make_report, tmp_path):
        p = tmp_path / "session.json"
        make_report.save(p)
        assert p.is_file()
        loaded = SessionReport.load(p)
        assert loaded.run_id == make_report.run_id
        assert loaded.model == make_report.model
        assert loaded.total_queries == make_report.total_queries

    def test_save_creates_parent_dirs(self, make_report, tmp_path):
        p = tmp_path / "deep" / "nested" / "session.json"
        make_report.save(p)
        assert p.is_file()

    def test_save_file_is_valid_json(self, make_report, tmp_path):
        p = tmp_path / "session.json"
        make_report.save(p)
        data = json.loads(p.read_text(encoding="utf-8"))
        assert "run_id" in data

    def test_new_with_no_safety_events_defaults_to_empty_list(self):
        r = SessionReport.new(
            model="gpt-4o",
            total_queries=1,
            cost_summary={},
            retrieval_findings=[],
            per_query=[],
            aggregate={},
            safety_events=None,
        )
        assert r.safety_events == []


# ═══════════════════════════════════════════════════════════════════════════════
# JSONReporter
# ═══════════════════════════════════════════════════════════════════════════════

class TestJSONReporter:

    def test_render_returns_valid_json(self, make_report):
        out = JSONReporter().render(make_report)
        parsed = json.loads(out)
        assert isinstance(parsed, dict)

    def test_render_has_required_keys(self, make_report):
        parsed = json.loads(JSONReporter().render(make_report))
        for key in ("run_id", "retrieval_findings", "per_query", "aggregate"):
            assert key in parsed

    def test_findings_sorted_critical_first(self, make_report):
        parsed = json.loads(JSONReporter().render(make_report))
        severities = [f["severity"] for f in parsed["retrieval_findings"]]
        assert severities[0] == "CRITICAL"

    def test_findings_sorted_warning_before_info(self, make_report):
        parsed = json.loads(JSONReporter().render(make_report))
        severities = [f["severity"] for f in parsed["retrieval_findings"]]
        w_idx = severities.index("WARNING")
        i_idx = severities.index("INFO")
        assert w_idx < i_idx

    def test_findings_sorted_full_order(self, make_report):
        parsed = json.loads(JSONReporter().render(make_report))
        severities = [f["severity"] for f in parsed["retrieval_findings"]]
        assert severities == ["CRITICAL", "WARNING", "INFO"]

    def test_render_indent_default(self, make_report):
        out = JSONReporter().render(make_report)
        # pretty-printed JSON has newlines
        assert "\n" in out

    def test_save_writes_nonempty_file(self, make_report, tmp_path):
        p = tmp_path / "report.json"
        JSONReporter().save(make_report, p)
        assert p.is_file()
        assert p.stat().st_size > 0

    def test_save_creates_parent_dirs(self, make_report, tmp_path):
        p = tmp_path / "sub" / "report.json"
        JSONReporter().save(make_report, p)
        assert p.is_file()

    def test_save_file_is_valid_json(self, make_report, tmp_path):
        p = tmp_path / "report.json"
        JSONReporter().save(make_report, p)
        parsed = json.loads(p.read_text(encoding="utf-8"))
        assert "retrieval_findings" in parsed

    def test_render_run_id_present(self, make_report):
        out = JSONReporter().render(make_report)
        assert make_report.run_id in out


# ═══════════════════════════════════════════════════════════════════════════════
# HTMLReporter
# ═══════════════════════════════════════════════════════════════════════════════

class TestHTMLReporter:

    def test_render_starts_with_doctype(self, make_report):
        out = HTMLReporter().render(make_report)
        assert out.startswith("<!DOCTYPE html>")

    def test_render_contains_closing_html_tag(self, make_report):
        out = HTMLReporter().render(make_report)
        assert "</html>" in out

    def test_render_contains_html_open_tag(self, make_report):
        out = HTMLReporter().render(make_report)
        assert "<html" in out

    def test_render_contains_run_id(self, make_report):
        out = HTMLReporter().render(make_report)
        assert make_report.run_id in out

    def test_render_contains_critical_badge(self, make_report):
        out = HTMLReporter().render(make_report)
        assert "CRITICAL" in out

    def test_render_contains_warning_badge(self, make_report):
        out = HTMLReporter().render(make_report)
        assert "WARNING" in out

    def test_render_contains_info_badge(self, make_report):
        out = HTMLReporter().render(make_report)
        assert "INFO" in out

    def test_render_no_external_http_urls(self, make_report):
        out = HTMLReporter().render(make_report)
        assert "http://" not in out
        assert "https://" not in out

    def test_render_no_external_link_tags(self, make_report):
        out = HTMLReporter().render(make_report)
        assert "<link" not in out.lower()

    def test_render_no_external_script_src(self, make_report):
        out = HTMLReporter().render(make_report)
        assert "<script src" not in out.lower()

    def test_render_contains_style_block(self, make_report):
        out = HTMLReporter().render(make_report)
        assert "<style>" in out

    def test_render_contains_findings_section(self, make_report):
        out = HTMLReporter().render(make_report)
        assert "Findings" in out

    def test_render_contains_per_query_accordion(self, make_report):
        out = HTMLReporter().render(make_report)
        assert "<details>" in out
        assert "<summary>" in out

    def test_render_contains_safety_section_when_events_present(self, make_report):
        out = HTMLReporter().render(make_report)
        # fixture has 1 safety event
        assert "Safety Events" in out

    def test_render_no_safety_section_when_events_empty(self):
        r = SessionReport.new(
            model="gpt-4o",
            total_queries=1,
            cost_summary={},
            retrieval_findings=[],
            per_query=[],
            aggregate={},
            safety_events=[],
        )
        out = HTMLReporter().render(r)
        assert "Safety Events" not in out

    def test_render_contains_model_name(self, make_report):
        out = HTMLReporter().render(make_report)
        assert make_report.model in out

    def test_save_writes_nonempty_file(self, make_report, tmp_path):
        p = tmp_path / "report.html"
        HTMLReporter().save(make_report, p)
        assert p.is_file()
        assert p.stat().st_size > 0

    def test_save_creates_parent_dirs(self, make_report, tmp_path):
        p = tmp_path / "sub" / "report.html"
        HTMLReporter().save(make_report, p)
        assert p.is_file()

    def test_save_file_contains_html(self, make_report, tmp_path):
        p = tmp_path / "report.html"
        HTMLReporter().save(make_report, p)
        content = p.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content

    def test_render_query_text_present(self, make_report):
        out = HTMLReporter().render(make_report)
        # First query text should appear (truncated to 120 chars)
        assert "AAPL revenue" in out

    def test_render_no_findings_shows_healthy_message(self):
        r = SessionReport.new(
            model="gpt-4o",
            total_queries=0,
            cost_summary={},
            retrieval_findings=[],
            per_query=[],
            aggregate={},
        )
        out = HTMLReporter().render(r)
        assert "healthy" in out


# ═══════════════════════════════════════════════════════════════════════════════
# MarkdownReporter
# ═══════════════════════════════════════════════════════════════════════════════

class TestMarkdownReporter:

    def test_render_contains_red_circle_emoji(self, make_report):
        out = MarkdownReporter().render(make_report)
        assert "\U0001f534" in out  # 🔴 CRITICAL

    def test_render_contains_yellow_circle_emoji(self, make_report):
        out = MarkdownReporter().render(make_report)
        assert "\U0001f7e1" in out  # 🟡 WARNING

    def test_render_contains_blue_circle_emoji(self, make_report):
        out = MarkdownReporter().render(make_report)
        assert "\U0001f535" in out  # 🔵 INFO

    def test_render_contains_gfm_table_separator(self, make_report):
        out = MarkdownReporter().render(make_report)
        assert "| --- |" in out

    def test_render_contains_h1(self, make_report):
        out = MarkdownReporter().render(make_report)
        assert out.startswith("# ragprobe")

    def test_render_contains_findings_section(self, make_report):
        out = MarkdownReporter().render(make_report)
        assert "## Findings" in out

    def test_render_contains_per_query_section(self, make_report):
        out = MarkdownReporter().render(make_report)
        assert "## Per-query" in out

    def test_render_contains_safety_blockquote(self, make_report):
        out = MarkdownReporter().render(make_report)
        # fixture has 1 safety event → blockquote should appear
        assert "> " in out
        assert "Safety events" in out

    def test_render_no_safety_blockquote_when_empty(self):
        r = SessionReport.new(
            model="gpt-4o",
            total_queries=1,
            cost_summary={},
            retrieval_findings=[],
            per_query=[],
            aggregate={},
            safety_events=[],
        )
        out = MarkdownReporter().render(r)
        assert "Safety events" not in out

    def test_render_contains_model_name(self, make_report):
        out = MarkdownReporter().render(make_report)
        assert make_report.model in out

    def test_render_contains_run_id(self, make_report):
        out = MarkdownReporter().render(make_report)
        assert make_report.run_id in out

    def test_save_writes_nonempty_file(self, make_report, tmp_path):
        p = tmp_path / "report.md"
        MarkdownReporter().save(make_report, p)
        assert p.is_file()
        assert p.stat().st_size > 0

    def test_save_creates_parent_dirs(self, make_report, tmp_path):
        p = tmp_path / "sub" / "report.md"
        MarkdownReporter().save(make_report, p)
        assert p.is_file()

    def test_render_no_findings_shows_healthy_message(self):
        r = SessionReport.new(
            model="gpt-4o",
            total_queries=0,
            cost_summary={},
            retrieval_findings=[],
            per_query=[],
            aggregate={},
        )
        out = MarkdownReporter().render(r)
        assert "healthy" in out

    def test_findings_sorted_in_markdown(self, make_report):
        out = MarkdownReporter().render(make_report)
        crit_pos = out.index("CRITICAL")
        warn_pos = out.index("WARNING")
        info_pos = out.index("INFO")
        # Within the findings table, CRITICAL should appear before WARNING and INFO
        # (Summary section might say "INFO" first, so search within findings block)
        findings_start = out.index("## Findings")
        findings_out = out[findings_start:]
        c = findings_out.index("CRITICAL")
        w = findings_out.index("WARNING")
        i = findings_out.index("INFO")
        assert c < w < i


# ═══════════════════════════════════════════════════════════════════════════════
# ReporterFactory
# ═══════════════════════════════════════════════════════════════════════════════

class TestReporterFactory:

    def test_get_json_returns_json_reporter(self):
        assert isinstance(ReporterFactory.get("json"), JSONReporter)

    def test_get_html_returns_html_reporter(self):
        assert isinstance(ReporterFactory.get("html"), HTMLReporter)

    def test_get_markdown_returns_markdown_reporter(self):
        assert isinstance(ReporterFactory.get("markdown"), MarkdownReporter)

    def test_get_md_alias_returns_markdown_reporter(self):
        assert isinstance(ReporterFactory.get("md"), MarkdownReporter)

    def test_get_case_insensitive_json(self):
        assert isinstance(ReporterFactory.get("JSON"), JSONReporter)

    def test_get_case_insensitive_html(self):
        assert isinstance(ReporterFactory.get("HTML"), HTMLReporter)

    def test_get_case_insensitive_markdown(self):
        assert isinstance(ReporterFactory.get("MARKDOWN"), MarkdownReporter)

    def test_get_unknown_raises_value_error(self):
        with pytest.raises(ValueError):
            ReporterFactory.get("pdf")

    def test_get_unknown_error_message_lists_valid_options(self):
        with pytest.raises(ValueError, match="json"):
            ReporterFactory.get("unknown_format")

    def test_get_empty_string_raises_value_error(self):
        with pytest.raises(ValueError):
            ReporterFactory.get("")

    def test_get_returns_fresh_instance_each_call(self):
        r1 = ReporterFactory.get("json")
        r2 = ReporterFactory.get("json")
        assert r1 is not r2
