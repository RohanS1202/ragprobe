"""
reporter.py — Reporting engine for ragprobe evaluation runs.

Four public classes:

  SessionReport     — Canonical schema for one eval run.  Holds all findings,
                      per-query data, cost info, and safety events.  Supports
                      JSON persistence (save / load) and dict round-trips.

  JSONReporter      — Pretty-printed JSON output with findings sorted by severity.
  HTMLReporter      — Self-contained HTML with dark-navy theme, no external deps.
  MarkdownReporter  — GitHub-Flavored Markdown with emoji severity indicators.

  ReporterFactory   — get("json" | "html" | "markdown" | "md") → reporter.

Usage::

    from ragprobe.reporter import SessionReport, ReporterFactory

    report   = SessionReport.load("results.json")
    reporter = ReporterFactory.get("html")
    reporter.save(report, "report.html")
"""

from __future__ import annotations

import datetime
import html as _html
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

logger = logging.getLogger(__name__)

# ── Package version ───────────────────────────────────────────────────────────
try:
    from importlib.metadata import version as _pkg_version
    _VERSION: str = _pkg_version("ragprobe")
except Exception:
    _VERSION = "0.1.0"

# ── Severity sort order ───────────────────────────────────────────────────────
_SEV_ORDER: dict[str, int] = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}


def _sort_findings(findings: list[dict]) -> list[dict]:
    return sorted(findings, key=lambda f: _SEV_ORDER.get(f.get("severity", "INFO"), 3))


def _e(v: Any) -> str:
    """HTML-escape a value."""
    return _html.escape(str(v))


# ═══════════════════════════════════════════════════════════════════════════════
# SessionReport
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SessionReport:
    """
    Canonical schema produced by every ``ragprobe eval`` run.

    This is the single source of truth written to disk and consumed by all
    three reporters.  Fields map directly to the outputs of the core modules:

    - ``retrieval_findings``  from ``RetrievalDiagnostic.run()["findings"]``
    - ``per_query``           from ``RetrievalDiagnostic.run()["per_query"]``
    - ``aggregate``           from ``RetrievalDiagnostic.run()["aggregate"]``
    - ``cost_summary``        from ``CostGuard.summary()`` serialised to dict
    - ``safety_events``       accumulated during the run from SafetyGate checks
    """

    run_id:             str
    timestamp:          str
    model:              str
    total_queries:      int
    cost_summary:       dict
    retrieval_findings: list[dict]
    per_query:          list[dict]
    aggregate:          dict
    safety_events:      list[dict]
    ragprobe_version:   str = field(default_factory=lambda: _VERSION)

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def new(
        cls,
        model:              str,
        total_queries:      int,
        cost_summary:       dict,
        retrieval_findings: list[dict],
        per_query:          list[dict],
        aggregate:          dict,
        safety_events:      list[dict] | None = None,
    ) -> "SessionReport":
        """
        Create a new SessionReport with an auto-generated run_id and timestamp.

        Parameters
        ----------
        model : str
            LLM model used during the run.
        total_queries : int
            Number of queries evaluated.
        cost_summary : dict
            Serialised output of ``CostGuard.summary()``.
        retrieval_findings : list[dict]
            Findings from ``RetrievalDiagnostic.run()``.
        per_query : list[dict]
            Per-query breakdown from ``RetrievalDiagnostic.run()``.
        aggregate : dict
            Aggregate stats from ``RetrievalDiagnostic.run()``.
        safety_events : list[dict] | None
            Safety events accumulated during the run.

        Returns
        -------
        SessionReport
        """
        return cls(
            run_id             = str(uuid.uuid4()),
            timestamp          = datetime.datetime.now().isoformat(timespec="seconds"),
            model              = model,
            total_queries      = total_queries,
            cost_summary       = cost_summary,
            retrieval_findings = retrieval_findings,
            per_query          = per_query,
            aggregate          = aggregate,
            safety_events      = safety_events or [],
            ragprobe_version   = _VERSION,
        )

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """
        Serialise to a JSON-safe dict.

        The dict is a faithful snapshot; field order is preserved and findings
        are NOT sorted (sorting is the reporter's responsibility).

        Returns
        -------
        dict
        """
        return {
            "run_id":             self.run_id,
            "timestamp":          self.timestamp,
            "model":              self.model,
            "total_queries":      self.total_queries,
            "cost_summary":       self.cost_summary,
            "retrieval_findings": self.retrieval_findings,
            "per_query":          self.per_query,
            "aggregate":          self.aggregate,
            "safety_events":      self.safety_events,
            "ragprobe_version":   self.ragprobe_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionReport":
        """
        Deserialise from a dict (e.g. read from a JSON file).

        Parameters
        ----------
        data : dict
            Dict produced by ``to_dict()`` or loaded from JSON.

        Returns
        -------
        SessionReport
        """
        return cls(
            run_id             = data["run_id"],
            timestamp          = data["timestamp"],
            model              = data["model"],
            total_queries      = data["total_queries"],
            cost_summary       = data.get("cost_summary", {}),
            retrieval_findings = data.get("retrieval_findings", []),
            per_query          = data.get("per_query", []),
            aggregate          = data.get("aggregate", {}),
            safety_events      = data.get("safety_events", []),
            ragprobe_version   = data.get("ragprobe_version", _VERSION),
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """
        Write the report to a JSON file.

        Creates parent directories automatically.

        Parameters
        ----------
        path : str | Path
            Destination file path.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        logger.debug("SessionReport saved → %s", p)

    @classmethod
    def load(cls, path: str | Path) -> "SessionReport":
        """
        Load a SessionReport from a JSON file.

        Parameters
        ----------
        path : str | Path
            Path to a JSON file previously written by ``save()``.

        Returns
        -------
        SessionReport
        """
        p    = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(data)


# ═══════════════════════════════════════════════════════════════════════════════
# JSONReporter
# ═══════════════════════════════════════════════════════════════════════════════

class JSONReporter:
    """
    Renders a SessionReport as pretty-printed JSON.

    Findings are sorted CRITICAL → WARNING → INFO in the output regardless of
    the order they appear in the SessionReport.
    """

    def render(self, report: SessionReport, indent: int = 2) -> str:
        """
        Render the report as a JSON string.

        Parameters
        ----------
        report : SessionReport
        indent : int
            JSON indentation width (default 2).

        Returns
        -------
        str
            Pretty-printed, UTF-8-safe JSON.
        """
        data = report.to_dict()
        data["retrieval_findings"] = _sort_findings(data["retrieval_findings"])
        return json.dumps(data, indent=indent, ensure_ascii=False)

    def save(self, report: SessionReport, path: str | Path) -> None:
        """
        Write the JSON report to *path*.

        Creates parent directories if they do not exist.

        Parameters
        ----------
        report : SessionReport
        path : str | Path
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.render(report), encoding="utf-8")
        logger.debug("JSONReporter saved → %s", p)


# ═══════════════════════════════════════════════════════════════════════════════
# HTMLReporter
# ═══════════════════════════════════════════════════════════════════════════════

# CSS is defined as a module-level constant so it is not inside an f-string
# (which would require escaping every { } in the CSS rules).
_HTML_CSS = (
    ":root{"
    "--bg:#0f172a;--surface:#1e293b;--card:#fff;--text:#334155;"
    "--muted:#94a3b8;--border:#e2e8f0;"
    "--crit:#dc3545;--warn:#f59e0b;--info:#0ea5e9;"
    "}"
    "body{background:var(--bg);color:var(--text);"
    "font-family:system-ui,-apple-system,sans-serif;"
    "margin:0;padding:2rem;line-height:1.5}"
    "h1,h2,h3{margin-top:0}"
    ".hdr{background:var(--surface);color:#f1f5f9;padding:1.5rem 2rem;"
    "border-radius:10px;margin-bottom:1.5rem}"
    ".hdr h1{margin:0;font-size:1.4rem}"
    ".hdr .meta{font-size:.82rem;color:var(--muted);margin-top:.35rem}"
    ".card{background:var(--card);border-radius:10px;padding:1.5rem;"
    "margin-bottom:1.5rem;box-shadow:0 1px 4px rgba(0,0,0,.25)}"
    ".card h2{font-size:.85rem;text-transform:uppercase;letter-spacing:.06em;"
    "color:var(--muted);margin-bottom:1rem;border-bottom:1px solid var(--border);"
    "padding-bottom:.5rem}"
    ".grid{display:grid;gap:1rem}"
    ".grid-4{grid-template-columns:repeat(4,1fr)}"
    ".grid-3{grid-template-columns:repeat(3,1fr)}"
    ".metric{text-align:center;padding:.5rem}"
    ".metric .val{font-size:1.8rem;font-weight:700;color:#0f172a}"
    ".metric .lbl{font-size:.72rem;text-transform:uppercase;color:var(--muted)}"
    ".badge{display:inline-block;font-size:.72rem;font-weight:700;"
    "padding:2px 9px;border-radius:10px;vertical-align:middle}"
    ".b-CRITICAL{background:var(--crit);color:#fff}"
    ".b-WARNING{background:var(--warn);color:#fff}"
    ".b-INFO{background:var(--info);color:#fff}"
    "table{width:100%;border-collapse:collapse}"
    "th{text-align:left;font-size:.78rem;text-transform:uppercase;"
    "color:var(--muted);padding:.5rem .75rem;border-bottom:2px solid var(--border)}"
    "td{padding:.65rem .75rem;border-bottom:1px solid #f1f5f9;"
    "font-size:.85rem;vertical-align:top;word-break:break-word;max-width:380px}"
    "tr:last-child td{border-bottom:none}"
    "details{border-bottom:1px solid var(--border)}"
    "details:last-child{border-bottom:none}"
    "summary{padding:.7rem .75rem;cursor:pointer;font-size:.85rem;list-style:none}"
    "summary::-webkit-details-marker{display:none}"
    "summary::before{content:'▶ ';font-size:.7rem;color:var(--muted)}"
    "details[open] summary::before{content:'▼ '}"
    "summary:hover{background:#f8fafc;border-radius:6px}"
    ".detail-body{padding:.25rem .75rem .75rem 1.5rem;font-size:.82rem;color:#475569}"
    ".safety-banner{border-left:4px solid var(--warn);background:#fffbeb;"
    "padding:1rem 1.5rem;border-radius:0 8px 8px 0;margin-bottom:.75rem}"
    ".ftr{text-align:center;color:var(--muted);font-size:.78rem;margin-top:2rem}"
)


class HTMLReporter:
    """
    Renders a SessionReport as a fully self-contained HTML file.

    - Zero external dependencies (no CDN links, no remote fonts).
    - All CSS is inlined in a single ``<style>`` block.
    - Uses ``<details>``/``<summary>`` for the per-query accordion (no JS).
    - Color scheme: dark navy background, white cards.
    - Severity badges: red = CRITICAL, amber = WARNING, blue = INFO.
    """

    def render(self, report: SessionReport) -> str:
        """
        Render the report to a self-contained HTML string.

        The output is guaranteed to contain ``<!DOCTYPE html>``, ``</html>``,
        the run_id, and the words CRITICAL / WARNING / INFO (if present in
        the findings).

        Parameters
        ----------
        report : SessionReport

        Returns
        -------
        str
            Valid, self-contained HTML.
        """
        parts = [
            "<!DOCTYPE html>\n",
            '<html lang="en">\n',
            "<head>\n",
            '<meta charset="utf-8">\n',
            '<meta name="viewport" content="width=device-width,initial-scale=1">\n',
            f"<title>ragprobe &mdash; {_e(report.run_id[:8])}</title>\n",
            f"<style>{_HTML_CSS}</style>\n",
            "</head>\n",
            "<body>\n",
            self._header(report),
            self._cost_card(report),
            self._metrics_card(report),
            self._findings_card(report),
            self._per_query_card(report),
            self._safety_section(report),
            self._footer(report),
            "</body>\n",
            "</html>\n",
        ]
        return "".join(parts)

    def save(self, report: SessionReport, path: str | Path) -> None:
        """
        Write the HTML report to *path*.

        Creates parent directories if they do not exist.

        Parameters
        ----------
        report : SessionReport
        path : str | Path
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.render(report), encoding="utf-8")
        logger.debug("HTMLReporter saved → %s", p)

    # ── Private section builders ──────────────────────────────────────────────

    @staticmethod
    def _header(r: SessionReport) -> str:
        return (
            '<div class="hdr">\n'
            "<h1>ragprobe diagnostic report</h1>\n"
            '<div class="meta">'
            f"Run&nbsp;ID:&nbsp;<code>{_e(r.run_id)}</code>"
            f"&ensp;|&ensp;{_e(r.timestamp)}"
            f"&ensp;|&ensp;Model:&nbsp;<strong>{_e(r.model)}</strong>"
            f"&ensp;|&ensp;ragprobe&nbsp;v{_e(r.ragprobe_version)}"
            "</div>\n"
            "</div>\n"
        )

    @staticmethod
    def _cost_card(r: SessionReport) -> str:
        cs      = r.cost_summary
        tokens  = cs.get("total_tokens", 0)
        cost    = cs.get("total_cost_usd", 0.0)
        bpct    = cs.get("budget_used_pct")
        bpct_s  = f"{bpct:.1f}%" if bpct is not None else "N/A"

        return (
            '<div class="card">\n'
            "<h2>Cost Summary</h2>\n"
            '<div class="grid grid-4">\n'
            f'<div class="metric"><div class="val">{_e(tokens):}</div><div class="lbl">Total tokens</div></div>\n'
            f'<div class="metric"><div class="val">${cost:.4f}</div><div class="lbl">Estimated cost</div></div>\n'
            f'<div class="metric"><div class="val">{bpct_s}</div><div class="lbl">Budget used</div></div>\n'
            f'<div class="metric"><div class="val">{r.total_queries}</div><div class="lbl">Queries</div></div>\n'
            "</div>\n"
            "</div>\n"
        )

    @staticmethod
    def _metrics_card(r: SessionReport) -> str:
        agg      = r.aggregate
        recall   = agg.get("mean_recall")
        recall_s = f"{recall:.3f}" if isinstance(recall, float) else "N/A"
        redund_s = f"{agg.get('mean_redundancy', 0.0):.3f}"
        cov_s    = f"{agg.get('mean_coverage_pct', 0.0):.1%}"
        n_crit   = sum(1 for f in r.retrieval_findings if f.get("severity") == "CRITICAL")
        n_warn   = sum(1 for f in r.retrieval_findings if f.get("severity") == "WARNING")

        return (
            '<div class="card">\n'
            "<h2>Retrieval Metrics</h2>\n"
            '<div class="grid grid-4">\n'
            f'<div class="metric"><div class="val">{recall_s}</div><div class="lbl">Mean recall</div></div>\n'
            f'<div class="metric"><div class="val">{redund_s}</div><div class="lbl">Mean redundancy</div></div>\n'
            f'<div class="metric"><div class="val">{cov_s}</div><div class="lbl">Coverage</div></div>\n'
            f'<div class="metric"><div class="val">{n_crit}C / {n_warn}W</div><div class="lbl">Findings</div></div>\n'
            "</div>\n"
            "</div>\n"
        )

    @staticmethod
    def _findings_card(r: SessionReport) -> str:
        findings = _sort_findings(r.retrieval_findings)
        if not findings:
            rows = "<tr><td colspan='4'><em>No findings — retrieval looks healthy.</em></td></tr>\n"
        else:
            rows = ""
            for f in findings:
                sev = f.get("severity", "INFO")
                val = f.get("value", 0)
                val_s = f"{val:.3f}" if isinstance(val, (int, float)) else _e(val)
                rows += (
                    "<tr>"
                    f'<td><span class="badge b-{_e(sev)}">{_e(sev)}</span></td>'
                    f"<td>{_e(f.get('metric',''))}</td>"
                    f"<td>{val_s}</td>"
                    f"<td>{_e(f.get('recommendation',''))}</td>"
                    "</tr>\n"
                )

        return (
            f'<div class="card">\n'
            f"<h2>Findings ({len(findings)})</h2>\n"
            "<table>\n"
            "<thead><tr><th>Severity</th><th>Metric</th><th>Value</th><th>Recommendation</th></tr></thead>\n"
            f"<tbody>{rows}</tbody>\n"
            "</table>\n"
            "</div>\n"
        )

    @staticmethod
    def _per_query_card(r: SessionReport) -> str:
        if not r.per_query:
            return ""

        items = ""
        for i, q in enumerate(r.per_query, 1):
            query_text  = q.get("query", "")[:120]
            recall_v    = q.get("recall")
            recall_s    = f"{recall_v:.3f}" if isinstance(recall_v, float) else "N/A"
            cov         = q.get("coverage", {})
            cov_pct     = cov.get("coverage_pct", 1.0)
            missing     = cov.get("missing_terms", [])
            missing_s   = ", ".join(missing) if missing else "<em>none</em>"
            red         = q.get("redundancy", {})
            red_mean    = red.get("mean", 0.0)
            worst       = red.get("worst_pair")
            worst_s     = (
                f"chunks {worst['index_a']}&harr;{worst['index_b']} "
                f"({worst['score']:.3f})"
                if worst else "<em>n/a</em>"
            )
            n_chunks = q.get("chunk_count", 0)

            items += (
                "<details>\n"
                f"<summary><strong>Q{i}:</strong> {_e(query_text)}"
                f"&ensp;<span style='color:#94a3b8'>recall {recall_s} | coverage {cov_pct:.0%}</span>"
                "</summary>\n"
                '<div class="detail-body">\n'
                f"<strong>Chunks retrieved:</strong> {n_chunks}<br>\n"
                f"<strong>Coverage:</strong> {cov_pct:.0%} &mdash; missing terms: {missing_s}<br>\n"
                f"<strong>Redundancy:</strong> mean {red_mean:.3f} &mdash; worst pair: {worst_s}\n"
                "</div>\n"
                "</details>\n"
            )

        return (
            '<div class="card">\n'
            "<h2>Per-query breakdown</h2>\n"
            f"{items}"
            "</div>\n"
        )

    @staticmethod
    def _safety_section(r: SessionReport) -> str:
        if not r.safety_events:
            return ""

        rows = ""
        for ev in r.safety_events:
            rows += (
                "<tr>"
                f"<td>{_e(ev.get('type',''))}</td>"
                f"<td>{_e(ev.get('document_index',''))}</td>"
                f"<td>{_e(ev.get('pattern',''))}</td>"
                f"<td>{_e(ev.get('risk_level',''))}</td>"
                f"<td>{_e(ev.get('message',''))}</td>"
                "</tr>\n"
            )

        return (
            '<div class="card">\n'
            f"<h2>&#9888;&#65039; Safety Events ({len(r.safety_events)})</h2>\n"
            '<div class="safety-banner">Injection patterns or validation errors were detected '
            "and affected documents were removed before evaluation.</div>\n"
            "<table>\n"
            "<thead><tr><th>Type</th><th>Document</th><th>Pattern</th>"
            "<th>Risk</th><th>Message</th></tr></thead>\n"
            f"<tbody>{rows}</tbody>\n"
            "</table>\n"
            "</div>\n"
        )

    @staticmethod
    def _footer(r: SessionReport) -> str:
        generated = datetime.datetime.now().isoformat(timespec="seconds")
        return (
            f'<div class="ftr">Generated by ragprobe&nbsp;v{_e(r.ragprobe_version)} '
            f"&mdash; {_e(generated)}</div>\n"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MarkdownReporter
# ═══════════════════════════════════════════════════════════════════════════════

_MD_EMOJI: dict[str, str] = {
    "CRITICAL": "\U0001f534",   # 🔴
    "WARNING":  "\U0001f7e1",   # 🟡
    "INFO":     "\U0001f535",   # 🔵
}


class MarkdownReporter:
    """
    Renders a SessionReport as GitHub-Flavored Markdown.

    Suitable for pasting into a GitHub issue, PR comment, or any GFM-capable
    markdown renderer.  Includes emoji severity indicators and GFM tables.
    """

    def render(self, report: SessionReport) -> str:
        """
        Render the report as a GFM markdown string.

        Parameters
        ----------
        report : SessionReport

        Returns
        -------
        str
            GitHub-Flavored Markdown text.
        """
        parts = [
            self._header(report),
            self._summary_table(report),
            self._findings_table(report),
            self._per_query_table(report),
            self._safety_blockquote(report),
            self._footer(report),
        ]
        return "\n".join(parts)

    def save(self, report: SessionReport, path: str | Path) -> None:
        """
        Write the Markdown report to *path*.

        Creates parent directories if they do not exist.

        Parameters
        ----------
        report : SessionReport
        path : str | Path
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.render(report), encoding="utf-8")
        logger.debug("MarkdownReporter saved → %s", p)

    # ── Private section builders ──────────────────────────────────────────────

    @staticmethod
    def _header(r: SessionReport) -> str:
        return (
            "# ragprobe diagnostic report\n\n"
            f"**Run ID:** `{r.run_id}`"
            f" &nbsp;|&nbsp; **Model:** `{r.model}`"
            f" &nbsp;|&nbsp; **ragprobe v{r.ragprobe_version}`"
            f" &nbsp;|&nbsp; {r.timestamp}\n"
        )

    @staticmethod
    def _summary_table(r: SessionReport) -> str:
        cs       = r.cost_summary
        cost     = cs.get("total_cost_usd", 0.0)
        tokens   = cs.get("total_tokens", 0)
        agg      = r.aggregate
        recall   = agg.get("mean_recall")
        recall_s = f"{recall:.3f}" if isinstance(recall, float) else "N/A"
        cov_s    = f"{agg.get('mean_coverage_pct', 0.0):.1%}"

        return (
            "## Summary\n\n"
            "| Model | Queries | Tokens | Est. Cost | Mean Recall | Coverage |\n"
            "| --- | --- | --- | --- | --- | --- |\n"
            f"| `{r.model}` | {r.total_queries} | {tokens:,} "
            f"| ${cost:.4f} | {recall_s} | {cov_s} |\n"
        )

    @staticmethod
    def _findings_table(r: SessionReport) -> str:
        findings = _sort_findings(r.retrieval_findings)
        if not findings:
            return "## Findings\n\n_No findings — retrieval looks healthy._\n"

        rows = ""
        for f in findings:
            sev   = f.get("severity", "INFO")
            emoji = _MD_EMOJI.get(sev, "")
            val   = f.get("value", 0)
            val_s = f"{val:.3f}" if isinstance(val, (int, float)) else str(val)
            rec   = f.get("recommendation", "").replace("|", "\\|")
            rows += (
                f"| {emoji} {sev} | {f.get('metric','')} | {val_s} | {rec} |\n"
            )

        return (
            "## Findings\n\n"
            "| Severity | Metric | Value | Recommendation |\n"
            "| --- | --- | --- | --- |\n"
            f"{rows}"
        )

    @staticmethod
    def _per_query_table(r: SessionReport) -> str:
        if not r.per_query:
            return ""

        rows = ""
        for i, q in enumerate(r.per_query, 1):
            query    = q.get("query", "")[:80].replace("|", "\\|")
            recall_v = q.get("recall")
            recall_s = f"{recall_v:.3f}" if isinstance(recall_v, float) else "N/A"
            cov      = q.get("coverage", {})
            cov_pct  = f"{cov.get('coverage_pct', 1.0):.0%}"
            missing  = ", ".join(cov.get("missing_terms", [])) or "—"
            rows += f"| {i} | {query} | {recall_s} | {cov_pct} | {missing} |\n"

        return (
            "## Per-query Recall\n\n"
            "| # | Query | Recall | Coverage | Missing Terms |\n"
            "| --- | --- | --- | --- | --- |\n"
            f"{rows}"
        )

    @staticmethod
    def _safety_blockquote(r: SessionReport) -> str:
        if not r.safety_events:
            return ""
        n = len(r.safety_events)
        return (
            f"> \u26a0\ufe0f **Safety events detected** "
            f"\u2014 {n} event(s) flagged during this run.  "
            "Affected documents were removed before evaluation.\n"
        )

    @staticmethod
    def _footer(r: SessionReport) -> str:
        generated = datetime.datetime.now().isoformat(timespec="seconds")
        return f"\n---\n_Generated by ragprobe v{r.ragprobe_version} \u00b7 {generated}_\n"


# ═══════════════════════════════════════════════════════════════════════════════
# ReporterFactory
# ═══════════════════════════════════════════════════════════════════════════════

class ReporterFactory:
    """
    Factory for obtaining the correct reporter by format name.

    Usage::

        reporter = ReporterFactory.get("html")
        reporter.save(report, "report.html")
    """

    _REGISTRY: dict[str, type] = {
        "json":     JSONReporter,
        "html":     HTMLReporter,
        "markdown": MarkdownReporter,
        "md":       MarkdownReporter,
    }

    @staticmethod
    def get(fmt: str) -> Union[JSONReporter, HTMLReporter, MarkdownReporter]:
        """
        Return a reporter instance for *fmt*.

        Parameters
        ----------
        fmt : str
            One of ``"json"``, ``"html"``, ``"markdown"``, or ``"md"``
            (case-insensitive).

        Returns
        -------
        JSONReporter | HTMLReporter | MarkdownReporter

        Raises
        ------
        ValueError
            If *fmt* is not a recognised format.
        """
        key = fmt.lower().strip()
        cls = ReporterFactory._REGISTRY.get(key)
        if cls is None:
            valid = ", ".join(sorted(ReporterFactory._REGISTRY.keys()))
            raise ValueError(
                f"Unknown report format {fmt!r}. "
                f"Valid options: {valid}"
            )
        return cls()
