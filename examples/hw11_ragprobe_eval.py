"""
hw11_ragprobe_eval.py — Evaluate the HW11 SEC Financial Analyst agent with ragprobe.

Prerequisites
-------------
1. Build the knowledge base first:
       python rohanvinayaksagvekar_build_kb.py
   This creates kb_out/embeddings.npy and kb_out/metadata.json.

2. Set OPENAI_API_KEY in your .env file.

Run
---
    python examples/hw11_ragprobe_eval.py

What this does
--------------
  1. Loads your 10-K corpus from kb_out/metadata.json.
  2. Uses RAGEvaluator to auto-generate adversarial test cases from those docs
     (negation, out-of-scope, ambiguous, conflicting, edge-case, hypothetical).
  3. Runs every test case through your agentic pipeline (the HW11 agent).
  4. Scores each answer on faithfulness, relevance, and context recall using
     an LLM judge.
  5. Prints a full summary to the terminal.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

KB_DIR = ROOT / "kb_out"

# ── Validate kb_out exists ────────────────────────────────────────────────────
if not KB_DIR.exists():
    sys.exit(
        "\n[error] kb_out/ directory not found.\n"
        "Run your build script first:\n"
        "    python rohanvinayaksagvekar_build_kb.py\n"
    )

# ── Imports ───────────────────────────────────────────────────────────────────
import json as _json

from rohanvinayaksagvekar_hw11 import (
    client,
    search_sec_kb,
    get_company_facts,
    get_recent_filings,
    TOOLS,
    SYSTEM,
    INFER_MODEL,
)
from ragprobe import RAGEvaluator


# ── 1. Load corpus from kb_out ────────────────────────────────────────────────

def load_corpus(max_chunks: int = 200) -> list[str]:
    """Load text chunks from the knowledge base metadata."""
    meta_path = KB_DIR / "metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)
    chunks = [m["text"] for m in metadata if m.get("text", "").strip()]
    # Cap to avoid excessive LLM calls during test generation
    return chunks[:max_chunks]


# ── 2. Pipeline adapter ───────────────────────────────────────────────────────

def hw11_pipeline(question: str) -> tuple[str, list[str]]:
    """
    Wrap the HW11 agentic loop as a ragprobe-compatible pipeline.

    Parameters
    ----------
    question : str
        A single query to evaluate.

    Returns
    -------
    answer : str
        The final answer produced by the agent.
    retrieved_chunks : list[str]
        All text chunks surfaced by search_sec_kb during this run.
    """
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": question},
    ]
    retrieved_chunks: list[str] = []

    while True:
        response = client.chat.completions.create(
            model=INFER_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        choice = response.choices[0]
        msg    = choice.message
        messages.append(msg)

        if choice.finish_reason == "stop":
            return msg.content or "", retrieved_chunks

        for tc in msg.tool_calls or []:
            args   = _json.loads(tc.function.arguments)
            name   = tc.function.name

            if name == "search_sec_kb":
                result = search_sec_kb(args["query"], args.get("tickers"))
                # Collect the individual chunks (split on the separator used in search_sec_kb)
                retrieved_chunks.extend(
                    c.strip() for c in result.split("\n\n---\n\n") if c.strip()
                )
            elif name == "get_company_facts":
                result = get_company_facts(args["ticker"])
            elif name == "get_recent_filings":
                result = get_recent_filings(args["ticker"], args.get("form_types"))
            else:
                result = f"Unknown tool: {name}"

            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result,
            })


# ── 3. Run evaluation ─────────────────────────────────────────────────────────

def main() -> None:
    print("Loading corpus from kb_out/ ...")
    corpus = load_corpus(max_chunks=200)
    print(f"  {len(corpus)} chunks loaded.\n")

    evaluator = RAGEvaluator(judge="openai")

    print("Generating adversarial test suite ...")
    suite = evaluator.generate_tests(
        documents  = corpus,
        n_cases    = 10,       # increase for a more thorough evaluation
        suite_name = "hw11_sec_eval",
    )
    print(f"  {len(suite)} test cases generated ({', '.join(f'{v} {k}' for k, v in suite.by_type.items())}).\n")

    print("Running HW11 pipeline against test suite (this will call the OpenAI API) ...\n")
    results = evaluator.evaluate(pipeline=hw11_pipeline, test_suite=suite)

    evaluator.print_summary(results)

    summary = evaluator.summarise(results)
    print(f"\nPass rate : {summary.pass_rate:.1%}")
    print(f"Faithfulness  (avg): {summary.avg_faithfulness:.2f}")
    print(f"Relevance     (avg): {summary.avg_relevance:.2f}")
    print(f"Context recall(avg): {summary.avg_context_recall:.2f}")

    if summary.worst_cases:
        print("\n── Worst cases ──────────────────────────────────────────")
        for r in summary.worst_cases:
            print(f"  [{r.test_case.attack_type.value}] {r.question[:80]}")
            print(f"    overall score: {r.overall_score:.2f}")


if __name__ == "__main__":
    main()
