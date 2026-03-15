"""
sec_eval.py — ragprobe evaluation on your real SEC 10-K RAG pipeline.

Run:
    python examples/sec_eval.py --kb-path "/path/to/Hw7/kb_out_token"

    python examples/sec_eval.py \
        --kb-path "/Users/rohansagvekar/Desktop/Prompt_Eng/Hw7/kb_out_token" \
        --tickers AAPL MSFT NVDA \
        --n-cases 10
"""

import json
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

from ragprobe import RAGEvaluator
from ragprobe.core.generator import AttackType

load_dotenv()

_api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
if not _api_key:
    print("Error: No API key found. Set API_KEY in your .env file.")
    sys.exit(1)

client      = OpenAI(api_key=_api_key)
EMBED_MODEL = "text-embedding-3-small"
INFER_MODEL = "gpt-4o-mini"
TOP_K       = 12
_tokenizer  = tiktoken.encoding_for_model(INFER_MODEL)

KB_PATH: Path = None


def search(query: str, tickers: list | None = None, k: int = TOP_K) -> list[str]:
    embeddings = np.load(KB_PATH / "embeddings.npy")
    with open(KB_PATH / "metadata.json") as f:
        metadata = json.load(f)

    res   = client.embeddings.create(input=[query], model=EMBED_MODEL, encoding_format="float")
    q_vec = np.array(res.data[0].embedding, dtype=np.float32)
    norms  = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_vec)
    scores = embeddings @ q_vec / np.where(norms == 0, 1, norms)

    if tickers:
        results = []
        for ticker in tickers:
            idx = [i for i, m in enumerate(metadata) if m.get("ticker") == ticker]
            if not idx: continue
            top = np.argsort(scores[idx])[-k:][::-1]
            results.extend(metadata[idx[i]]["text"] for i in top)
        return results
    else:
        return [metadata[i]["text"] for i in np.argsort(scores)[-k:][::-1]]


def sec_rag_pipeline(query: str, tickers: list | None = None) -> tuple[str, list[str]]:
    contexts = search(query, tickers=tickers)
    context_block = "\n\n---\n\n".join(contexts)
    developer_prompt = (
        "You are a financial analyst. Answer using only the context below.\n"
        "Cite figures where available. Say so if context is insufficient.\n\n"
        f"<context>\n{context_block}\n</context>"
    )
    response = client.responses.create(
        model = INFER_MODEL,
        input = [
            {"role": "developer", "content": developer_prompt},
            {"role": "user",      "content": query},
        ],
    )
    return response.output_text, contexts


def load_sample_chunks(tickers: list | None = None, n_chunks: int = 30) -> list[str]:
    import random
    with open(KB_PATH / "metadata.json") as f:
        metadata = json.load(f)

    pool = [m["text"] for m in metadata if not tickers or m.get("ticker") in tickers]
    keywords = ["revenue","operating income","net income","cash flow","risk factor",
                "capital expenditure","research and development","earnings per share",
                "total assets","debt","margin","fiscal year","quarter","guidance"]
    rich = [c for c in pool if any(k.lower() in c.lower() for k in keywords)]
    source = rich if len(rich) >= n_chunks else pool
    return random.sample(source, min(n_chunks, len(source)))


def main():
    global KB_PATH

    parser = argparse.ArgumentParser(description="ragprobe SEC 10-K evaluation")
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "NVDA"])
    parser.add_argument("--n-cases", type=int, default=10)
    parser.add_argument("--kb-path", type=str, default=None,
                        help="Path to kb_out_token/ folder")
    args    = parser.parse_args()
    tickers = [t.upper() for t in args.tickers]

    # Resolve KB path
    if args.kb_path:
        KB_PATH = Path(args.kb_path).expanduser().resolve()
    else:
        candidates = [
            Path(__file__).resolve().parent / "kb_out_token",
            Path(__file__).resolve().parent.parent / "kb_out_token",
            Path.home() / "Desktop" / "Prompt_Eng" / "Hw7" / "kb_out_token",
        ]
        KB_PATH = next((p for p in candidates if p.exists()), None)

    if KB_PATH is None or not (KB_PATH / "embeddings.npy").exists():
        print("\nError: Knowledge base not found.")
        print('Run: python examples/sec_eval.py --kb-path "/path/to/kb_out_token"')
        print(f'\nYour HW7 path is likely:')
        print('  /Users/rohansagvekar/Desktop/Prompt_Eng/Hw7/kb_out_token')
        sys.exit(1)

    print(f"\n ragprobe — SEC 10-K adversarial evaluation")
    print(f" Tickers  : {', '.join(tickers)}")
    print(f" Cases    : {args.n_cases}")
    print(f" KB path  : {KB_PATH}")
    print("=" * 52)

    # Step 1: Load chunks
    print(f"\nStep 1: Loading financially dense chunks...")
    docs = load_sample_chunks(tickers=tickers, n_chunks=30)
    print(f"Loaded {len(docs)} chunks")
    print(f"\nSample:\n{docs[0][:250]}...\n")

    # Step 2: Generate test cases
    print(f"Step 2: Generating {args.n_cases} adversarial financial questions...")
    evaluator = RAGEvaluator(judge="openai")
    suite = evaluator.generate_tests(
        documents    = docs,
        n_cases      = args.n_cases,
        attack_types = [
            AttackType.NEGATION,
            AttackType.EDGE_CASE,
            AttackType.OUT_OF_SCOPE,
            AttackType.CONFLICTING,
        ],
        suite_name = f"sec_10k_{'_'.join(tickers)}",
    )

    print(f"\nGenerated {suite.total} test cases:")
    for t, c in suite.by_type.items():
        print(f"  {t:<20} {c}")

    print(f"\nSample questions:")
    for tc in suite.cases[:4]:
        print(f"\n  [{tc.attack_type.value}]")
        print(f"  Q: {tc.question}")
        print(f"  Expected: {tc.expected_behavior[:90]}...")

    # Step 3: Evaluate
    print(f"\n\nStep 3: Running RAG pipeline + scoring...")

    def pipeline(query: str) -> tuple[str, list[str]]:
        return sec_rag_pipeline(query, tickers=tickers)

    results = evaluator.evaluate(
        pipeline   = pipeline,
        test_suite = suite,
        metrics    = ["faithfulness", "relevance", "context_recall"],
    )

    # Step 4: Report
    print()
    evaluator.print_summary(results)
    summary = evaluator.summarise(results)

    if summary.failures_by_attack_type:
        print("\nWhat this means for your pipeline:")
        insights = {
            "negation":     "Pipeline asserts things the 10-K explicitly negates",
            "edge_case":    "Pipeline hallucinates specific figures (EPS, capex, revenue)",
            "out_of_scope": "Pipeline answers questions it should decline",
            "conflicting":  "Pipeline struggles with cross-company comparisons",
        }
        for attack, count in sorted(summary.failures_by_attack_type.items(), key=lambda x: -x[1]):
            if attack in insights:
                print(f"  {attack}: {insights[attack]}")

    if summary.worst_cases:
        worst = summary.worst_cases[0]
        print(f"\nWorst case (score={worst.overall_score:.2f}):")
        print(f"  Q:      {worst.question}")
        print(f"  Answer: {worst.answer[:200]}...")
        if worst.faithfulness and worst.faithfulness.violations:
            print(f"\n  Violations:")
            for v in worst.faithfulness.violations[:3]:
                print(f"    - {v}")

    # Step 5: Save results
    out_file = Path(__file__).parent / "sec_eval_results.json"
    with open(out_file, "w") as f:
        json.dump([{
            "question":       r.question,
            "attack_type":    r.test_case.attack_type.value,
            "answer":         r.answer,
            "overall_score":  r.overall_score,
            "passed":         r.passed,
            "faithfulness":   r.faithfulness.score   if r.faithfulness   else None,
            "relevance":      r.relevance.score      if r.relevance      else None,
            "context_recall": r.context_recall.score if r.context_recall else None,
            "violations":     r.faithfulness.violations if r.faithfulness else [],
        } for r in results], f, indent=2)

    print(f"\nResults saved to: {out_file}")
    print("\nDone.\n")


if __name__ == "__main__":
    main()