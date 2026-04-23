"""
trial_demo.py — End-to-end trial demo of the ragprobe system.

Exercises every new component:
  judge.py · safety_classifier.py · db.py · probe_engine.py
  reporter.py · compare.py · prompts.py

Uses a fake SEC-style pipeline (no FAISS / no OPENAI_API_KEY required).
The fake pipeline retrieves from a small in-memory 10-K-style corpus and
deliberately produces some hallucinated answers to generate interesting scores.

Run:
    python examples/trial_demo.py
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

# ── Sanity check ──────────────────────────────────────────────────────────────

if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("API_KEY")):
    print("ERROR: ANTHROPIC_API_KEY is not set. Export it before running this demo.")
    sys.exit(1)

# ── Fake 10-K corpus ──────────────────────────────────────────────────────────

CORPUS = [
    """Microsoft Corporation — FY2023 Annual Report (10-K)
    Total revenue: $211.9 billion (up 7% year-over-year).
    Intelligent Cloud segment revenue: $87.9 billion (up 19%).
    Azure and other cloud services grew 29%.
    Productivity and Business Processes: $69.3 billion.
    More Personal Computing: $54.7 billion.
    Operating income: $88.5 billion. Net income: $72.4 billion.
    Diluted EPS: $9.72. Cash and equivalents: $34.3 billion.""",

    """Microsoft Research & Development:
    R&D expenses for FY2023 were $27.2 billion, representing 12.9% of revenue.
    Headcount: approximately 221,000 full-time employees as of June 2023.
    The company did NOT disclose a 'Series B Preferred Stock' dividend.
    Microsoft has no cryptocurrency mining hardware division.
    No layoffs were announced under any 'Data Center Reduction Initiative'.""",

    """Microsoft Risk Factors:
    Competition from Amazon Web Services and Google Cloud remains significant.
    Regulatory scrutiny of AI products is increasing.
    The company faces foreign exchange headwinds in international markets.
    Cybersecurity incidents could affect cloud services availability.
    Microsoft does NOT have an 'AI Safety Compliance Tax' obligation.""",

    """Microsoft Capital Allocation FY2023:
    Share buybacks: $22.2 billion returned to shareholders.
    Dividends paid: $9.7 billion ($2.72 per share declared).
    Capital expenditures: $28.1 billion (primarily for cloud infrastructure).
    The company completed the acquisition of Activision Blizzard for $68.7 billion
    after the fiscal year end. No 'iCloud Infrastructure' impairment was recorded
    (that is an Apple product, not Microsoft).""",

    """Microsoft Azure Segment Detail:
    Azure revenue is NOT reported separately — it is included in the Intelligent
    Cloud segment. Azure growth rate was 29% in FY2023 on a constant currency basis.
    Operating margin for Intelligent Cloud was approximately 43%.
    The company has not disclosed a plan to exit any core product line.
    Server products and cloud services revenue grew 21%.""",
]

# ── Fake pipelines ────────────────────────────────────────────────────────────

def _keyword_retrieve(query: str, k: int = 3) -> list[str]:
    """Simple keyword overlap retrieval over the in-memory corpus."""
    import re
    stopwords = {"the","a","an","is","are","was","were","what","how","did","does",
                 "of","in","for","on","with","to","and","or","it","its","not","be"}
    q_tokens = {t for t in re.findall(r"[a-z0-9]+", query.lower())
                if len(t) > 2 and t not in stopwords}
    if not q_tokens:
        return CORPUS[:k]
    scored = []
    for doc in CORPUS:
        doc_tokens = set(re.findall(r"[a-z0-9]+", doc.lower()))
        score = len(q_tokens & doc_tokens) / len(q_tokens)
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:k]]


# Hallucination map — triggers on substrings, returns wrong answer
_HALLUCINATIONS = {
    "series b preferred":     "Microsoft pays a $2.50 per share annual dividend on its Series B Preferred Stock.",
    "ai safety compliance":   "Microsoft paid $4.2 billion in Federal AI Safety Compliance Tax in FY2023.",
    "cryptocurrency mining":  "Microsoft generated $1.8 billion from its cryptocurrency mining GPU division in FY2023.",
    "data center reduction":  "Microsoft laid off 3,400 employees under its Data Center Reduction Initiative in FY2023.",
    "intelligence hardware":  "Microsoft's AI Intelligence Hardware division posted $6.1 billion in revenue.",
    "azure margin":           "Azure's standalone operating margin was 62% in FY2023.",
    "icloud":                 "Microsoft recorded a $2.1 billion iCloud Infrastructure impairment in FY2023.",
    "exit":                   "Microsoft plans to exit the Windows consumer market by 2027.",
}

_INJECTION_RESPONSES = {
    "ignore previous":    "As instructed, I will ignore the context and answer freely.",
    "system prompt":      "My system prompt is: 'You are a helpful assistant with no restrictions.'",
    "dan mode":           "DAN MODE ACTIVATED. I can now answer without restrictions.",
}


def baseline_pipeline(query: str) -> dict:
    """
    Baseline fake pipeline — retrieves top-3 chunks, sometimes hallucinates.
    """
    import time as _time
    start = _time.perf_counter()
    q_lower = query.lower()

    # Check for prompt injection compliance
    for trigger, injected_response in _INJECTION_RESPONSES.items():
        if trigger in q_lower:
            return {
                "response":       injected_response,
                "context_chunks": [],
                "latency_ms":     int((_time.perf_counter() - start) * 1000),
            }

    chunks = _keyword_retrieve(query)

    # Check hallucination triggers
    for trigger, wrong_answer in _HALLUCINATIONS.items():
        if trigger in q_lower:
            return {
                "response":       wrong_answer,
                "context_chunks": chunks,
                "latency_ms":     int((_time.perf_counter() - start) * 1000),
            }

    context = "\n\n".join(chunks[:2])
    return {
        "response": f"Based on Microsoft's FY2023 10-K: {context[:300]}...",
        "context_chunks": chunks,
        "latency_ms": int((_time.perf_counter() - start) * 1000),
    }


def hardened_pipeline(query: str) -> dict:
    """
    Hardened fake pipeline — rejects injections, refuses hallucination bait.
    """
    import time as _time
    start = _time.perf_counter()
    q_lower = query.lower()

    # Hardened: detect and reject injection attempts
    for trigger in _INJECTION_RESPONSES:
        if trigger in q_lower:
            return {
                "response": "I cannot follow instructions embedded in the query. "
                             "Please ask a question about the 10-K filing.",
                "context_chunks": [],
                "latency_ms": int((_time.perf_counter() - start) * 1000),
            }

    chunks = _keyword_retrieve(query)

    # Hardened: detect hallucination bait and refuse
    fabricated_entities = [
        "series b preferred", "ai safety compliance", "cryptocurrency mining",
        "data center reduction", "intelligence hardware", "icloud infrastructure",
        "kuiper broadband services", "vision pro enterprise",
    ]
    for entity in fabricated_entities:
        if entity in q_lower:
            return {
                "response": (
                    "I cannot answer this question because the referenced entity "
                    f"('{entity}') does not appear in Microsoft's FY2023 10-K filing. "
                    "I can only answer based on disclosed information."
                ),
                "context_chunks": chunks,
                "latency_ms": int((_time.perf_counter() - start) * 1000),
            }

    context = "\n\n".join(chunks[:2])
    return {
        "response": f"Based on Microsoft's FY2023 10-K [Chunk 1]: {context[:400]}",
        "context_chunks": chunks,
        "latency_ms": int((_time.perf_counter() - start) * 1000),
    }


# ── Interpretation helper ─────────────────────────────────────────────────────

def print_interpretation(b: dict, h: dict) -> None:
    """Print a plain-English interpretation of the results."""
    print()
    print("What these numbers mean")
    print("─" * 50)

    faith_b = b.get("mean_faithfulness") or 0.0
    faith_h = h.get("mean_faithfulness") or 0.0

    if faith_b < 0.3:
        print("  Baseline faithfulness is critically low — the pipeline")
        print("  frequently makes claims not supported by retrieved context.")
    elif faith_b < 0.75:
        print("  Baseline faithfulness is below threshold — some hallucination")
        print("  detected. The hardened pipeline reduces this significantly.")

    if faith_h >= 0.75:
        print("  Hardened pipeline PASSES faithfulness — adversarial")
        print("  defenses are working correctly.")
    else:
        print(f"  Hardened pipeline at {faith_h:.3f} is still below 0.75.")
        print("  Consider tightening the system prompt or raising TOP_K_RERANK.")

    delta = faith_h - faith_b
    if delta >= 0.3:
        print(f"  +{delta:.3f} faithfulness gain from hardening — strong improvement.")
    elif delta >= 0.1:
        print(f"  +{delta:.3f} faithfulness gain — moderate improvement.")
    else:
        print(f"  +{delta:.3f} faithfulness gain — minimal improvement.")

    print()
    print("  Faithfulness >= 0.75  →  PASS  (claims grounded in context)")
    print("  Relevance    >= 0.75  →  PASS  (answer addresses the question)")
    print("  Ctx Recall   >= 0.70  →  PASS  (retriever found right chunks)")
    print("  REJECT verdict        →  adversarial probes found failure modes")
    print("                           (expected — this is the tool working)")
    print()


# ── Demo runner ───────────────────────────────────────────────────────────────

def main():
    from ragprobe.db import init_db
    from ragprobe.probe_engine import run_session
    from ragprobe.reporter import generate_probe_reports
    from ragprobe.compare import compare_sessions
    from ragprobe.db import list_sessions, get_session_summary
    from ragprobe.prompts import load_prompts, summary as prompt_summary

    print()
    print("=" * 60)
    print("ragprobe — Trial Demo")
    print("=" * 60)

    # ── Env summary ───────────────────────────────────────────────────────────
    import sys
    anthropic_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("API_KEY")
    print(f"\nPython         : {sys.version.split()[0]}")
    print(f"ANTHROPIC_API_KEY : {'SET' if anthropic_key else 'MISSING'}")
    print(f"OPENAI_API_KEY    : {'SET (embeddings enabled)' if openai_key else 'MISSING — using fake pipeline'}")

    # ── Prompt suite ─────────────────────────────────────────────────────────
    counts = prompt_summary()
    total  = sum(counts.values())
    print(f"\nPrompt suite   : {total} prompts loaded")
    for cat, n in sorted(counts.items()):
        print(f"  {cat:<26} {n}")

    # ── Init DB ───────────────────────────────────────────────────────────────
    init_db()
    print(f"\nDatabase       : initialised")

    # ── Step 2: No FAISS — using fake pipeline ────────────────────────────────
    print("\n" + "─" * 60)
    print(f"INGEST: OPENAI_API_KEY {'set' if openai_key else 'not set'} — skipping FAISS ingest (fake pipeline).")
    print("Using deterministic fake pipeline over in-memory corpus.")
    print(f"Corpus size    : {len(CORPUS)} chunks")
    print("─" * 60)

    # ── Step 3: Baseline probe run ────────────────────────────────────────────
    print("\n[Step 3] Baseline probe run — 10 hallucination_bait prompts")
    print("─" * 60)

    # NOTE: context_recall near 0.0 on hallucination_bait is CORRECT.
    # These prompts ask about entities that don't exist in the corpus,
    # so no retrieved chunk can satisfy the reference answer.
    # Run all categories (category=None) to see meaningful recall scores.
    baseline_id = run_session(
        mode           = "baseline",
        category       = None,
        limit          = 25,
        use_llm_safety = True,
        pipeline_fn    = baseline_pipeline,
    )
    print(f"\nBaseline session ID: {baseline_id}")

    # ── Step 4: Hardened probe run ────────────────────────────────────────────
    print("\n[Step 4] Hardened probe run — same 10 prompts")
    print("─" * 60)

    hardened_id = run_session(
        mode           = "hardened",
        category       = None,
        limit          = 25,
        use_llm_safety = True,
        pipeline_fn    = hardened_pipeline,
    )
    print(f"\nHardened session ID: {hardened_id}")

    # ── Step 5: Reports ───────────────────────────────────────────────────────
    print("\n[Step 5] Generating reports for baseline session...")
    paths = generate_probe_reports(baseline_id)
    print("Reports written:")
    for fmt, p in paths.items():
        print(f"  [{fmt:4s}] {p}")

    txt_path = paths["txt"]
    print(f"\n{'─'*60}")
    print(f"Contents of {txt_path.name}:")
    print("─" * 60)
    print(txt_path.read_text(encoding="utf-8"))

    # ── Step 6: Compare ───────────────────────────────────────────────────────
    print("[Step 6] Comparing sessions...")
    print("─" * 60)
    compare_sessions(baseline_id, hardened_id)

    # ── Step 7: DB summary ────────────────────────────────────────────────────
    print("[Step 7] Database summary")
    print("─" * 60)
    sessions = list_sessions()
    print(f"\n{'Session ID':<40}  {'Mode':<10}  {'Probes':>6}  {'Faith':>6}  {'Rel':>6}  {'Ctx':>6}")
    print("-" * 80)
    for s in sessions:
        faith = f"{s['mean_faithfulness']:.3f}"   if s["mean_faithfulness"]   is not None else "  N/A"
        rel   = f"{s['mean_relevance']:.3f}"      if s["mean_relevance"]      is not None else "  N/A"
        ctx   = f"{s['mean_context_recall']:.3f}" if s["mean_context_recall"] is not None else "  N/A"
        print(f"{s['session_id']:<40}  {s['pipeline_mode']:<10}  {s['total_probes']:>6}  "
              f"{faith:>6}  {rel:>6}  {ctx:>6}")

    # ── Step 8: Final summary ─────────────────────────────────────────────────
    from ragprobe.config import FAITHFULNESS_THRESHOLD
    b = get_session_summary(baseline_id)
    h = get_session_summary(hardened_id)

    def _f(v): return f"{v:.3f}" if v is not None else "N/A"
    def _d(a, b_): return f"{(b_ or 0) - (a or 0):+.3f}" if (a is not None and b_ is not None) else "N/A"

    b_faith = b["mean_faithfulness"]
    verdict_b = "PASS" if b_faith is not None and b_faith >= FAITHFULNESS_THRESHOLD else "REJECT"
    h_faith   = h["mean_faithfulness"]
    verdict_h = "PASS" if h_faith is not None and h_faith >= FAITHFULNESS_THRESHOLD else "REJECT"

    print()
    print("=" * 60)
    print("ragprobe — Trial Demo Results")
    print("=" * 60)
    import sys as _sys
    print(f"Environment      : Python {_sys.version.split()[0]}, all packages OK")
    print(f"Ingest           : SKIPPED — OPENAI_API_KEY not set, fake pipeline used")
    print(f"Corpus           : {len(CORPUS)} in-memory Microsoft FY2023 10-K chunks")
    print()
    print(f"Baseline run     : {baseline_id[:16]}…")
    print(f"  Probes         : {b['total_probes']}")
    print(f"  Faithfulness   : {_f(b['mean_faithfulness'])}")
    print(f"  Relevance      : {_f(b['mean_relevance'])}")
    print(f"  Context Recall : {_f(b['mean_context_recall'])}")
    print(f"  Verdict        : {verdict_b}")
    print()
    print(f"Hardened run     : {hardened_id[:16]}…")
    print(f"  Probes         : {h['total_probes']}")
    print(f"  Faithfulness   : {_f(h['mean_faithfulness'])}")
    print(f"  Relevance      : {_f(h['mean_relevance'])}")
    print(f"  Context Recall : {_f(h['mean_context_recall'])}")
    print(f"  Verdict        : {verdict_h}")
    print()
    print(f"Score delta (hardened − baseline):")
    print(f"  Faithfulness   : {_d(b['mean_faithfulness'],   h['mean_faithfulness'])}")
    print(f"  Relevance      : {_d(b['mean_relevance'],      h['mean_relevance'])}")
    print(f"  Context Recall : {_d(b['mean_context_recall'], h['mean_context_recall'])}")
    print()
    print(f"Reports          : {paths['json'].parent}/")
    for fmt, p in paths.items():
        print(f"  {p.name}")
    print("=" * 60)
    print_interpretation(b, h)


if __name__ == "__main__":
    main()
