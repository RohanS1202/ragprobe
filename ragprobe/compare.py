"""
compare.py — Side-by-side comparison of two ragprobe probe sessions.

Pulls results for both sessions from SQLite, prints a comparison table,
highlights per-category improvements/regressions, and exports a CSV.

Usage::

    from ragprobe.compare import compare_sessions

    compare_sessions("session-id-a", "session-id-b")

CLI::

    ragprobe compare --session-a SESSION_A --session-b SESSION_B
"""

from __future__ import annotations

import csv
import io
import logging
from pathlib import Path
from typing import Optional

from ragprobe.config import DB_PATH, FAITHFULNESS_THRESHOLD, REPORTS_DIR

logger = logging.getLogger(__name__)


def _load_session(session_id: str, db_path: Path) -> tuple[dict, list[dict]]:
    """
    Load session summary and all probe results for a session.

    Parameters
    ----------
    session_id : str
    db_path : Path

    Returns
    -------
    tuple[dict, list[dict]]
        ``(session_summary, probe_results)``

    Raises
    ------
    ValueError
        If the session does not exist.
    """
    from ragprobe.db import get_session_summary, get_all_results

    session = get_session_summary(session_id, db_path)
    if session is None:
        raise ValueError(f"Session not found: {session_id!r}")
    results = get_all_results(session_id, db_path)
    return session, results


def _per_category_stats(results: list[dict]) -> dict[str, dict]:
    """
    Compute per-category mean scores and failure rate.

    Parameters
    ----------
    results : list[dict]
        Probe result rows from the database.

    Returns
    -------
    dict[str, dict]
        ``{category: {"mean_faithfulness": float, "mean_relevance": float,
                      "mean_context_recall": float, "failure_rate": float,
                      "total": int}}``
    """
    from collections import defaultdict

    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        buckets[r.get("prompt_category", "unknown")].append(r)

    stats = {}
    for cat, rows in buckets.items():
        n          = len(rows)
        faith_vals = [r["faithfulness"]   or 0.0 for r in rows]
        rel_vals   = [r["relevance"]      or 0.0 for r in rows]
        ctx_vals   = [r["context_recall"] or 0.0 for r in rows]
        failures   = sum(1 for r in rows if (r["faithfulness"] or 0.0) < FAITHFULNESS_THRESHOLD)

        stats[cat] = {
            "total":               n,
            "mean_faithfulness":   sum(faith_vals) / n,
            "mean_relevance":      sum(rel_vals) / n,
            "mean_context_recall": sum(ctx_vals) / n,
            "failure_rate":        failures / n,
        }
    return stats


def _safety_counts(results: list[dict]) -> dict[str, int]:
    """Count safety flags across all probe results."""
    counts = {
        "injection_compliance":      0,
        "confidentiality_violation": 0,
        "refusal_evasion":           0,
    }
    for r in results:
        for flag in counts:
            if r.get(flag):
                counts[flag] += 1
    return counts


def compare_sessions(
    session_a: str,
    session_b: str,
    db_path:   Path = DB_PATH,
    out_dir:   Path = REPORTS_DIR,
) -> Path:
    """
    Compare two probe sessions and print a side-by-side summary.

    Parameters
    ----------
    session_a : str
        First session ID (treated as "baseline" / reference).
    session_b : str
        Second session ID (treated as "comparison" / candidate).
    db_path : Path
        SQLite database file.
    out_dir : Path
        Directory to write the comparison CSV.

    Returns
    -------
    Path
        Path to the exported ``compare_{a}_vs_{b}.csv`` file.
    """
    sess_a, results_a = _load_session(session_a, db_path)
    sess_b, results_b = _load_session(session_b, db_path)

    cat_a = _per_category_stats(results_a)
    cat_b = _per_category_stats(results_b)
    saf_a = _safety_counts(results_a)
    saf_b = _safety_counts(results_b)

    all_cats = sorted(set(cat_a) | set(cat_b))

    # ── Print comparison table ────────────────────────────────────────────────
    _print_header(sess_a, sess_b)
    _print_aggregate(sess_a, sess_b)
    _print_safety(saf_a, saf_b)
    _print_per_category(all_cats, cat_a, cat_b)

    # ── Export CSV ────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    short_a = session_a[:8]
    short_b = session_b[:8]
    csv_path = out_dir / f"compare_{short_a}_vs_{short_b}.csv"
    _export_csv(all_cats, cat_a, cat_b, saf_a, saf_b, sess_a, sess_b, csv_path)

    print(f"\nComparison CSV written to: {csv_path}\n")
    return csv_path


# ── Print helpers ─────────────────────────────────────────────────────────────

def _print_header(sess_a: dict, sess_b: dict) -> None:
    """Print session header block."""
    print()
    print("=" * 70)
    print("ragprobe session comparison")
    print("=" * 70)
    print(f"  Session A  : {sess_a['session_id'][:16]}…  mode={sess_a['pipeline_mode']}")
    print(f"  Session B  : {sess_b['session_id'][:16]}…  mode={sess_b['pipeline_mode']}")
    print()


def _delta_str(val_b: Optional[float], val_a: Optional[float]) -> str:
    """Format a delta with +/- sign and arrow."""
    if val_a is None or val_b is None:
        return "   N/A  "
    delta = (val_b or 0.0) - (val_a or 0.0)
    sign  = "+" if delta >= 0 else ""
    arrow = "▲" if delta > 0.005 else ("▼" if delta < -0.005 else "→")
    return f"{sign}{delta:+.3f} {arrow}"


def _print_aggregate(sess_a: dict, sess_b: dict) -> None:
    """Print aggregate score comparison table."""
    metrics = [
        ("Faithfulness",    "mean_faithfulness"),
        ("Relevance",       "mean_relevance"),
        ("Context recall",  "mean_context_recall"),
    ]
    print(f"  {'Metric':<22} {'Session A':>10}  {'Session B':>10}  {'Delta':>12}")
    print("  " + "-" * 58)
    for label, key in metrics:
        a_val = sess_a.get(key)
        b_val = sess_b.get(key)
        a_s   = f"{a_val:.3f}" if a_val is not None else "  N/A "
        b_s   = f"{b_val:.3f}" if b_val is not None else "  N/A "
        print(f"  {label:<22} {a_s:>10}  {b_s:>10}  {_delta_str(b_val, a_val):>12}")
    print()


def _print_safety(saf_a: dict, saf_b: dict) -> None:
    """Print safety flag count comparison."""
    flags = [
        ("injection_compliance",      "Injection compliance"),
        ("confidentiality_violation", "Confidentiality violation"),
        ("refusal_evasion",           "Refusal evasion"),
    ]
    print(f"  {'Safety flag':<30} {'Session A':>10}  {'Session B':>10}  {'Delta':>8}")
    print("  " + "-" * 62)
    for key, label in flags:
        a_v = saf_a.get(key, 0)
        b_v = saf_b.get(key, 0)
        d   = b_v - a_v
        arrow = "▲" if d > 0 else ("▼" if d < 0 else "→")
        print(f"  {label:<30} {a_v:>10}  {b_v:>10}  {d:>+5} {arrow}")
    print()


def _print_per_category(
    all_cats: list[str],
    cat_a:    dict[str, dict],
    cat_b:    dict[str, dict],
) -> None:
    """Print per-category faithfulness failure rate comparison."""
    print("  Per-category failure rates  (faithfulness < threshold = fail)")
    print(f"  {'Category':<26} {'A fail%':>8}  {'B fail%':>8}  {'Change':>10}")
    print("  " + "-" * 58)
    for cat in all_cats:
        a_stat = cat_a.get(cat, {})
        b_stat = cat_b.get(cat, {})
        a_fail = a_stat.get("failure_rate", 0.0) * 100
        b_fail = b_stat.get("failure_rate", 0.0) * 100
        delta  = b_fail - a_fail
        arrow  = "▲" if delta > 1 else ("▼" if delta < -1 else "→")
        print(
            f"  {cat:<26} {a_fail:>7.1f}%  {b_fail:>7.1f}%  "
            f"{delta:>+7.1f}% {arrow}"
        )
    print()


# ── CSV export ────────────────────────────────────────────────────────────────

def _export_csv(
    all_cats: list[str],
    cat_a:    dict[str, dict],
    cat_b:    dict[str, dict],
    saf_a:    dict[str, int],
    saf_b:    dict[str, int],
    sess_a:   dict,
    sess_b:   dict,
    path:     Path,
) -> None:
    """Export the comparison data as a CSV file."""
    buf = io.StringIO()
    writer = csv.writer(buf)

    # Aggregate section
    writer.writerow(["section", "metric", "session_a", "session_b", "delta"])
    for label, key in [
        ("aggregate", "mean_faithfulness"),
        ("aggregate", "mean_relevance"),
        ("aggregate", "mean_context_recall"),
    ]:
        a_v = sess_a.get(key)
        b_v = sess_b.get(key)
        delta = (b_v - a_v) if (a_v is not None and b_v is not None) else None
        writer.writerow([label, key, a_v, b_v, delta])

    # Safety section
    for flag in ("injection_compliance", "confidentiality_violation", "refusal_evasion"):
        writer.writerow(["safety", flag, saf_a.get(flag, 0), saf_b.get(flag, 0),
                         saf_b.get(flag, 0) - saf_a.get(flag, 0)])

    # Per-category section
    for cat in all_cats:
        a_s = cat_a.get(cat, {})
        b_s = cat_b.get(cat, {})
        for metric in ("mean_faithfulness", "mean_relevance", "mean_context_recall", "failure_rate"):
            a_v = a_s.get(metric)
            b_v = b_s.get(metric)
            delta = (b_v - a_v) if (a_v is not None and b_v is not None) else None
            writer.writerow([f"category:{cat}", metric, a_v, b_v, delta])

    path.write_text(buf.getvalue(), encoding="utf-8")


def compare_three_sessions(
    session_a: str,
    session_b: str,
    session_c: str,
    labels:    tuple[str, str, str] = ("A", "B", "C"),
    db_path:   Path = DB_PATH,
    out_dir:   Path = REPORTS_DIR,
) -> Path:
    """
    Compare three probe sessions in a single table.

    Parameters
    ----------
    session_a, session_b, session_c : str
        Session IDs to compare (A = reference, B and C = candidates).
    labels : tuple[str, str, str]
        Short labels for the three sessions (used in table headers).
    db_path : Path
    out_dir : Path

    Returns
    -------
    Path
        Path to the exported ``compare_three_<a>_<b>_<c>.csv`` file.
    """
    sess_a, results_a = _load_session(session_a, db_path)
    sess_b, results_b = _load_session(session_b, db_path)
    sess_c, results_c = _load_session(session_c, db_path)

    cat_a = _per_category_stats(results_a)
    cat_b = _per_category_stats(results_b)
    cat_c = _per_category_stats(results_c)
    saf_a = _safety_counts(results_a)
    saf_b = _safety_counts(results_b)
    saf_c = _safety_counts(results_c)

    all_cats = sorted(set(cat_a) | set(cat_b) | set(cat_c))
    la, lb, lc = labels

    # ── Print table ──────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("ragprobe three-way session comparison")
    print("=" * 80)
    print(f"  [{la}] {sess_a['session_id'][:16]}…  mode={sess_a['pipeline_mode']}")
    print(f"  [{lb}] {sess_b['session_id'][:16]}…  mode={sess_b['pipeline_mode']}")
    print(f"  [{lc}] {sess_c['session_id'][:16]}…  mode={sess_c['pipeline_mode']}")
    print()

    # Aggregate scores
    metrics = [
        ("Faithfulness",   "mean_faithfulness"),
        ("Relevance",      "mean_relevance"),
        ("Context recall", "mean_context_recall"),
    ]
    w = 10
    print(f"  {'Metric':<22} {la:>{w}}  {lb:>{w}}  {lc:>{w}}  {'B−A':>9}  {'C−A':>9}")
    print("  " + "-" * 72)
    for label, key in metrics:
        av = sess_a.get(key)
        bv = sess_b.get(key)
        cv = sess_c.get(key)
        a_s = f"{av:.3f}" if av is not None else "  N/A "
        b_s = f"{bv:.3f}" if bv is not None else "  N/A "
        c_s = f"{cv:.3f}" if cv is not None else "  N/A "
        print(
            f"  {label:<22} {a_s:>{w}}  {b_s:>{w}}  {c_s:>{w}}"
            f"  {_delta_str(bv, av):>9}  {_delta_str(cv, av):>9}"
        )
    print()

    # Safety flags
    flags = [
        ("injection_compliance",      "Injection compliance"),
        ("confidentiality_violation", "Confidentiality violation"),
        ("refusal_evasion",           "Refusal evasion"),
    ]
    print(f"  {'Safety flag':<30} {la:>6}  {lb:>6}  {lc:>6}  {'B−A':>5}  {'C−A':>5}")
    print("  " + "-" * 62)
    for key, lbl in flags:
        av = saf_a.get(key, 0)
        bv = saf_b.get(key, 0)
        cv = saf_c.get(key, 0)
        print(f"  {lbl:<30} {av:>6}  {bv:>6}  {cv:>6}  {bv-av:>+5}  {cv-av:>+5}")
    print()

    # Per-category failure rates
    print("  Per-category failure rates  (faithfulness < threshold = fail)")
    print(f"  {'Category':<26} {la+' fail%':>9}  {lb+' fail%':>9}  {lc+' fail%':>9}")
    print("  " + "-" * 62)
    for cat in all_cats:
        af = cat_a.get(cat, {}).get("failure_rate", 0.0) * 100
        bf = cat_b.get(cat, {}).get("failure_rate", 0.0) * 100
        cf = cat_c.get(cat, {}).get("failure_rate", 0.0) * 100
        print(f"  {cat:<26} {af:>8.1f}%  {bf:>8.1f}%  {cf:>8.1f}%")
    print()

    # ── Export CSV ────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    sa, sb, sc = session_a[:8], session_b[:8], session_c[:8]
    csv_path = out_dir / f"compare_three_{sa}_{sb}_{sc}.csv"

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["section", "metric", la, lb, lc, f"{lb}-{la}", f"{lc}-{la}"])
    for _, key in metrics:
        av = sess_a.get(key)
        bv = sess_b.get(key)
        cv = sess_c.get(key)
        ba = (bv - av) if (av is not None and bv is not None) else None
        ca = (cv - av) if (av is not None and cv is not None) else None
        writer.writerow(["aggregate", key, av, bv, cv, ba, ca])
    for key, _ in flags:
        av, bv, cv = saf_a.get(key, 0), saf_b.get(key, 0), saf_c.get(key, 0)
        writer.writerow(["safety", key, av, bv, cv, bv - av, cv - av])
    for cat in all_cats:
        for metric in ("mean_faithfulness", "mean_relevance", "mean_context_recall", "failure_rate"):
            av = cat_a.get(cat, {}).get(metric)
            bv = cat_b.get(cat, {}).get(metric)
            cv = cat_c.get(cat, {}).get(metric)
            ba = (bv - av) if (av is not None and bv is not None) else None
            ca = (cv - av) if (av is not None and cv is not None) else None
            writer.writerow([f"category:{cat}", metric, av, bv, cv, ba, ca])
    csv_path.write_text(buf.getvalue(), encoding="utf-8")

    print(f"Three-way comparison CSV written to: {csv_path}\n")
    return csv_path
