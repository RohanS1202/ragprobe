"""
cli.py — Command-line interface for ragprobe.

Three sub-commands:

  ragprobe eval   --config config.yaml [--output results.json]
  ragprobe scan   --corpus ./docs --query "..." [--raise-on-injection]
  ragprobe report --input results.json --format json|html [--output out.html]

Entry point: main()
"""

from __future__ import annotations

import argparse
import datetime
import html as _html_mod
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Argparse with exit-code 1 on usage errors ─────────────────────────────────

class _Parser(argparse.ArgumentParser):
    """ArgumentParser that exits with code 1 (not 2) on usage errors."""

    def error(self, message: str) -> None:  # type: ignore[override]
        self.print_usage(sys.stderr)
        print(f"error: {message}", file=sys.stderr)
        sys.exit(1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _die(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def _ok(msg: str) -> None:
    print(f"\u2713 {msg}")


def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError:
        _die("PyYAML not installed — run: pip install pyyaml")
    if not path.is_file():
        _die(f"config file not found: {path}")
    try:
        with path.open() as fh:
            data = yaml.safe_load(fh)
        return data or {}
    except Exception as exc:
        _die(f"failed to parse config {path}: {exc}")


def _load_corpus(corpus_dir: str | Path) -> list[str]:
    """Load all .txt files from a directory; return list of strings."""
    p = Path(corpus_dir)
    if not p.is_dir():
        _die(f"corpus_dir not found or not a directory: {p}")
    docs = []
    for txt_file in sorted(p.glob("*.txt")):
        try:
            docs.append(txt_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Skipping %s: %s", txt_file, exc)
    return docs


def _retrieve(query: str, corpus: list[str], k: int = 5) -> list[str]:
    """
    Simple lexical retrieval: return the top-k corpus chunks most relevant
    to *query* by non-stopword token overlap.
    """
    if not corpus:
        return []
    _sw = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "or", "and", "but",
        "if", "this", "that", "it", "its", "not", "no", "what", "how",
        "when", "where", "who", "i", "we", "you", "he", "she", "they",
    }
    q_tokens = {
        t for t in re.findall(r"[a-z0-9]+", query.lower())
        if len(t) >= 2 and t not in _sw
    }
    if not q_tokens:
        return corpus[:k]
    scores = [
        len(q_tokens & set(re.findall(r"[a-z0-9]+", c.lower()))) / len(q_tokens)
        for c in corpus
    ]
    ranked = sorted(range(len(corpus)), key=lambda i: scores[i], reverse=True)
    return [corpus[i] for i in ranked[:k]]


def _write_json(data: Any, path: str | Path) -> None:
    out = Path(path)
    try:
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.debug("Wrote JSON to %s", out)
    except Exception as exc:
        _die(f"could not write output file {out}: {exc}")


# ── Sub-command: eval ─────────────────────────────────────────────────────────

def _cmd_eval(args: argparse.Namespace) -> None:
    from ragprobe.safety    import SafetyGate
    from ragprobe.retrieval import RetrievalDiagnostic

    config      = _load_yaml(Path(args.config))
    corpus_dir  = config.get("corpus_dir", ".")
    queries     = config.get("queries", [])
    references  = config.get("references")
    top_k       = int(config.get("top_k", 5))
    sem_recall  = bool(config.get("semantic_recall", False))

    if not queries:
        _die("config must contain at least one query under 'queries:'")

    # ── Load corpus ────────────────────────────────────────────────────────────
    corpus = _load_corpus(corpus_dir)
    if not corpus:
        _die(f"no .txt files found in corpus_dir: {corpus_dir}")
    logger.debug("Loaded %d documents from %s", len(corpus), corpus_dir)

    # ── Safety gate ────────────────────────────────────────────────────────────
    gate = SafetyGate()
    clean_corpus, safety_report = gate.check_corpus(corpus)

    if safety_report["injection_blocked"]:
        print(
            f"[warning] {safety_report['injection_blocked']} document(s) removed "
            f"by injection guard (risk >= HIGH)",
            file=sys.stderr,
        )

    if not clean_corpus:
        _die("all corpus documents were blocked by the safety gate")

    # ── Retrieval diagnostic ───────────────────────────────────────────────────
    chunks_per_query = [_retrieve(q, clean_corpus, top_k) for q in queries]

    diag = RetrievalDiagnostic()

    run_kwargs: dict = {}
    if sem_recall:
        embed_model = config.get("embed_model", "text-embedding-3-small")
        api_key     = config.get("api_key")
        try:
            import os
            from dotenv import load_dotenv
            load_dotenv()
            from openai import OpenAI
            client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            run_kwargs = {"semantic_recall": True, "client": client, "model": embed_model}
        except Exception as exc:
            print(f"[warning] semantic_recall disabled: {exc}", file=sys.stderr)

    results = diag.run(queries, chunks_per_query, references, **run_kwargs)

    # ── Attach metadata ────────────────────────────────────────────────────────
    results["_meta"] = {
        "corpus_size":       len(corpus),
        "clean_corpus_size": len(clean_corpus),
        "safety":            safety_report,
        "config":            {k: v for k, v in config.items() if k != "api_key"},
        "generated":         datetime.datetime.now().isoformat(timespec="seconds"),
    }

    # ── Print summary & write output ──────────────────────────────────────────
    n_findings = len(results["findings"])
    n_critical = sum(1 for f in results["findings"] if f["severity"] == "CRITICAL")
    n_warning  = sum(1 for f in results["findings"] if f["severity"] == "WARNING")

    # Print findings to stdout
    if results["findings"]:
        print()
        for f in results["findings"]:
            sev = f["severity"]
            print(f"  [{sev:8s}] {f['metric']}: {f['value']:.3f} — {f['recommendation']}")
        print()

    budget_note = (
        f"${config.get('budget_usd', 0.00):.4f} budget"
        if "budget_usd" in config
        else "no cost (lexical mode)"
    )
    _ok(
        f"eval complete \u2014 {len(queries)} quer{'y' if len(queries)==1 else 'ies'}, "
        f"{n_critical} critical / {n_warning} warning, {budget_note}"
    )

    if args.output:
        _write_json(results, args.output)
        print(f"  \u2192 {args.output}")


# ── Sub-command: scan ─────────────────────────────────────────────────────────

def _cmd_scan(args: argparse.Namespace) -> None:
    from ragprobe.core.validators     import InputValidator
    from ragprobe.core.injection_guard import InjectionGuard
    from ragprobe.retrieval            import ChunkAnalyzer

    corpus_path = Path(args.corpus)
    if not corpus_path.is_dir():
        _die(f"--corpus must be a directory: {corpus_path}")

    docs: list[str] = []
    paths: list[Path] = sorted(corpus_path.glob("*.txt"))
    for p in paths:
        try:
            docs.append(p.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Skipping %s: %s", p, exc)

    if not docs:
        print(f"No .txt files found in {corpus_path}")
        _ok(f"scan complete \u2014 0 documents")
        return

    # ── Validation ────────────────────────────────────────────────────────────
    validator  = InputValidator()
    val_result = validator.validate_documents(docs)

    val_errors = [i for i in val_result.issues if i.severity == "error"]
    if val_errors:
        print(f"Validation errors ({len(val_errors)}):")
        for e in val_errors:
            print(f"  [{e.code}] {e.message}")

    # ── Injection scan ────────────────────────────────────────────────────────
    guard  = InjectionGuard()
    report = guard.scan_corpus(docs)

    print(f"Scanned {len(docs)} document(s) from {corpus_path}")
    print(f"Overall risk: {report.overall_risk.value.upper()}")
    print()

    if report.flagged_documents == 0:
        print("  No injection patterns found.")
    else:
        print(f"  {report.flagged_documents} document(s) flagged:")
        for idx, scan_result in report.results:
            fname = paths[idx].name if idx < len(paths) else f"doc[{idx}]"
            for m in scan_result.matches:
                print(
                    f"    {fname}  "
                    f"[{m.risk_level.value.upper():8s}]  "
                    f"{m.pattern_name:30s}  "
                    f"matched: {m.matched_text!r}"
                )

    # ── Coverage gap analysis ─────────────────────────────────────────────────
    if args.query:
        print()
        analyzer = ChunkAnalyzer()
        cov      = analyzer.coverage_gaps(args.query, docs)
        print(f"Query coverage: {cov['coverage_pct']:.0%}")
        if cov["missing_terms"]:
            print(f"  Missing terms: {', '.join(cov['missing_terms'])}")
        else:
            print("  All query terms covered in corpus.")

    print()
    flagged_note = (
        f"{report.flagged_documents} flagged"
        if report.flagged_documents
        else "clean"
    )
    _ok(f"scan complete \u2014 {len(docs)} document(s), {flagged_note}")

    if args.raise_on_injection and report.flagged_documents > 0:
        _die("injection patterns detected in corpus")


# ── Sub-command: report ───────────────────────────────────────────────────────

def _cmd_report(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    if not input_path.is_file():
        _die(f"--input file not found: {input_path}")

    try:
        data = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _die(f"invalid JSON in {input_path}: {exc}")
    except Exception as exc:
        _die(f"could not read {input_path}: {exc}")

    if args.format == "json":
        rendered = json.dumps(data, indent=2)
    else:
        rendered = _render_html(data)

    if args.output:
        out = Path(args.output)
        try:
            out.write_text(rendered, encoding="utf-8")
        except Exception as exc:
            _die(f"could not write {out}: {exc}")
        _ok(f"report complete \u2014 format={args.format} \u2192 {out}")
    else:
        print(rendered)
        _ok(f"report complete \u2014 format={args.format}")


def _render_html(data: dict) -> str:
    """Render a results dict as a self-contained, single-file HTML report."""
    findings  = data.get("findings", [])
    per_query = data.get("per_query", [])
    aggregate = data.get("aggregate", {})
    meta      = data.get("_meta", {})

    n_queries   = len(per_query)
    n_critical  = sum(1 for f in findings if f["severity"] == "CRITICAL")
    n_warning   = sum(1 for f in findings if f["severity"] == "WARNING")
    generated   = meta.get("generated", datetime.datetime.now().isoformat(timespec="seconds"))

    mean_recall = aggregate.get("mean_recall")
    recall_str  = f"{mean_recall:.3f}" if isinstance(mean_recall, float) else "N/A"
    redund_str  = f"{aggregate.get('mean_redundancy', 0.0):.3f}"
    cov_str     = f"{aggregate.get('mean_coverage_pct', 0.0):.3f}"

    def _e(v: Any) -> str:
        return _html_mod.escape(str(v))

    # ── Findings HTML ─────────────────────────────────────────────────────────
    sev_colour = {"CRITICAL": "#dc3545", "WARNING": "#fd7e14", "INFO": "#0dcaf0"}
    sev_text   = {"CRITICAL": "#fff",    "WARNING": "#fff",    "INFO": "#212529"}

    findings_html = ""
    if not findings:
        findings_html = "<p>No findings — retrieval looks healthy.</p>"
    for f in findings:
        sev  = f.get("severity", "INFO")
        bg   = sev_colour.get(sev, "#6c757d")
        fg   = sev_text.get(sev, "#fff")
        val  = f.get("value", 0)
        val_s = f"{val:.3f}" if isinstance(val, (int, float)) else _e(val)
        findings_html += (
            f'<div class="finding" style="border-left:4px solid {bg}">'
            f'<span class="badge" style="background:{bg};color:{fg}">{_e(sev)}</span>'
            f'&nbsp;<strong>{_e(f.get("metric",""))}</strong>: {val_s}'
            f'<p class="rec">{_e(f.get("recommendation",""))}</p>'
            f"</div>\n"
        )

    # ── Per-query table ────────────────────────────────────────────────────────
    rows_html = ""
    for q in per_query:
        recall_v = q.get("recall")
        rec_s    = f"{recall_v:.3f}" if isinstance(recall_v, float) else "N/A"
        red_s    = f"{q.get('redundancy', {}).get('mean', 0.0):.3f}"
        cov_s    = f"{q.get('coverage', {}).get('coverage_pct', 0.0):.3f}"
        miss     = ", ".join(q.get("coverage", {}).get("missing_terms", []))
        rows_html += (
            f"<tr>"
            f"<td>{_e(q.get('query','')[:100])}</td>"
            f"<td>{q.get('chunk_count', 0)}</td>"
            f"<td>{rec_s}</td>"
            f"<td>{red_s}</td>"
            f"<td>{cov_s}</td>"
            f"<td>{_e(miss) if miss else '<span class=ok>none</span>'}</td>"
            f"</tr>\n"
        )

    css = """
body{font-family:system-ui,sans-serif;margin:2rem;background:#f8f9fa;color:#212529}
h1{color:#1a1a2e;margin-bottom:.25rem}
.meta{color:#6c757d;font-size:.85rem;margin-top:0}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin:1.5rem 0}
.stat{background:#fff;border-radius:8px;padding:1rem;text-align:center;
      box-shadow:0 1px 3px rgba(0,0,0,.1)}
.stat .num{font-size:2rem;font-weight:700;color:#1a1a2e}
.stat .lbl{font-size:.75rem;text-transform:uppercase;color:#6c757d}
.section{background:#fff;border-radius:8px;padding:1.5rem;margin:1rem 0;
          box-shadow:0 1px 3px rgba(0,0,0,.1)}
h2{margin-top:0;font-size:1.1rem;border-bottom:1px solid #dee2e6;padding-bottom:.5rem}
.finding{padding:.75rem 1rem;margin:.5rem 0;background:#f8f9fa;border-radius:0 4px 4px 0}
.badge{font-size:.75rem;font-weight:700;padding:2px 8px;border-radius:12px;
       vertical-align:middle}
.rec{margin:.4rem 0 0;font-size:.875rem;color:#495057}
table{width:100%;border-collapse:collapse}
th{background:#343a40;color:#fff;padding:.6rem .75rem;text-align:left;font-size:.85rem}
td{padding:.6rem .75rem;border-top:1px solid #dee2e6;font-size:.85rem;
   vertical-align:top;word-break:break-word;max-width:300px}
tr:nth-child(even) td{background:#f8f9fa}
.ok{color:#198754;font-style:italic}
"""

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>"
        '<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>ragprobe report</title>"
        f"<style>{css}</style>"
        "</head>\n"
        "<body>\n"
        "<h1>ragprobe diagnostic report</h1>\n"
        f'<p class="meta">Generated {_e(generated)}</p>\n'
        '<div class="stats">\n'
        f'<div class="stat"><div class="num">{n_queries}</div><div class="lbl">Queries</div></div>\n'
        f'<div class="stat"><div class="num">{recall_str}</div><div class="lbl">Mean recall</div></div>\n'
        f'<div class="stat"><div class="num">{redund_str}</div><div class="lbl">Mean redundancy</div></div>\n'
        f'<div class="stat"><div class="num">{cov_str}</div><div class="lbl">Mean coverage</div></div>\n'
        "</div>\n"
        f'<div class="section"><h2>Findings ({len(findings)} — {n_critical} critical, {n_warning} warning)</h2>\n'
        f"{findings_html}</div>\n"
        '<div class="section"><h2>Per-query breakdown</h2>\n'
        "<table><thead><tr>"
        "<th>Query</th><th>Chunks</th><th>Recall</th>"
        "<th>Redundancy</th><th>Coverage</th><th>Missing terms</th>"
        "</tr></thead><tbody>\n"
        f"{rows_html}"
        "</tbody></table></div>\n"
        "</body>\n"
        "</html>\n"
    )


# ── Argument parser setup ─────────────────────────────────────────────────────

def _build_parser() -> _Parser:
    parser = _Parser(
        prog="ragprobe",
        description="Adversarial testing and retrieval diagnostics for RAG pipelines.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ── eval ──────────────────────────────────────────────────────────────────
    p_eval = sub.add_parser(
        "eval",
        help="Run retrieval diagnostics from a config file.",
        description=(
            "Load a YAML config, safety-gate the corpus, run RetrievalDiagnostic "
            "across all configured queries, and print a ranked findings report."
        ),
    )
    p_eval.add_argument(
        "--config", required=True, metavar="PATH",
        help="Path to config.yaml.",
    )
    p_eval.add_argument(
        "--output", metavar="PATH",
        help="Write results JSON to this file.",
    )

    # ── scan ──────────────────────────────────────────────────────────────────
    p_scan = sub.add_parser(
        "scan",
        help="Scan a document corpus for injection patterns.",
        description=(
            "Load all .txt files from --corpus, run InputValidator and "
            "InjectionScanner, and print a findings table."
        ),
    )
    p_scan.add_argument(
        "--corpus", required=True, metavar="DIR",
        help="Directory containing .txt corpus files.",
    )
    p_scan.add_argument(
        "--query", metavar="TEXT",
        help="Optional query for coverage-gap analysis.",
    )
    p_scan.add_argument(
        "--raise-on-injection", dest="raise_on_injection",
        action="store_true",
        help="Exit with code 1 if any injection pattern is found.",
    )

    # ── report ────────────────────────────────────────────────────────────────
    p_report = sub.add_parser(
        "report",
        help="Render a results JSON file as JSON or HTML.",
        description=(
            "Read a results JSON file (output of 'ragprobe eval') and render "
            "it as pretty-printed JSON or a self-contained HTML report."
        ),
    )
    p_report.add_argument(
        "--input", required=True, metavar="PATH",
        help="Path to results JSON file.",
    )
    p_report.add_argument(
        "--format", choices=["json", "html"], default="json",
        help="Output format (default: json).",
    )
    p_report.add_argument(
        "--output", metavar="PATH",
        help="Write rendered report to this file (default: stdout).",
    )

    return parser


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Main entry point for the ragprobe CLI."""
    parser = _build_parser()
    args   = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    try:
        if args.command == "eval":
            _cmd_eval(args)
        elif args.command == "scan":
            _cmd_scan(args)
        elif args.command == "report":
            _cmd_report(args)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        logger.debug("Unhandled exception", exc_info=True)
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
