"""
test_judge.py — 30-second smoke test to verify the LLM judge is
scoring faithfulness and relevance correctly and independently.

Run:
    python examples/test_judge.py

Expected output:
    Case 1 — Faithful + Relevant:
      faithfulness   should be HIGH  (>=0.7) : PASS or FAIL
      relevance      should be HIGH  (>=0.7) : PASS or FAIL

    Case 2 — Hallucination (unfaithful) but still Relevant:
      faithfulness   should be LOW   (<0.4)  : PASS or FAIL
      relevance      should be HIGH  (>=0.6) : PASS or FAIL

    Case 3 — Faithful but Off-Topic (irrelevant):
      faithfulness   should be HIGH  (>=0.7) : PASS or FAIL
      relevance      should be LOW   (<0.4)  : PASS or FAIL

    Case 4 — Correct Refusal (out of scope):
      faithfulness   should be HIGH  (>=0.7) : PASS or FAIL
      relevance      should be MED   (>=0.3) : PASS or FAIL
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from ragprobe.judge import score

CASES = [
    {
        "name": "Case 1 — Faithful + Relevant",
        "query":    "What was Microsoft total revenue in FY2023?",
        "context":  "Microsoft total revenue in FY2023 was $211.9 billion, "
                    "up 7% year-over-year.",
        "response": "Microsoft's total revenue in FY2023 was $211.9 billion, "
                    "representing 7% growth year-over-year.",
        "checks": {
            "faithfulness":   ("HIGH",  lambda s: s >= 0.7),
            "relevance":      ("HIGH",  lambda s: s >= 0.7),
            "context_recall": ("HIGH",  lambda s: s >= 0.7),
        },
    },
    {
        "name": "Case 2 — Hallucination (wrong facts) but Relevant topic",
        "query":    "What was Microsoft's Series B preferred dividend in FY2023?",
        "context":  "Microsoft does NOT have a Series B preferred stock. "
                    "No such dividend exists.",
        "response": "Microsoft paid a $2.50 per share annual dividend on its "
                    "Series B Preferred Stock in FY2023.",
        "checks": {
            "faithfulness":   ("LOW",   lambda s: s < 0.4),
            "relevance":      ("HIGH",  lambda s: s >= 0.6),
        },
    },
    {
        "name": "Case 3 — Faithful but completely Off-Topic",
        "query":    "What was Microsoft total revenue in FY2023?",
        "context":  "Microsoft total revenue in FY2023 was $211.9 billion.",
        "response": "The weather in Seattle today is partly cloudy with a "
                    "high of 58 degrees Fahrenheit.",
        "checks": {
            "faithfulness":   ("HIGH",  lambda s: s >= 0.7),
            "relevance":      ("LOW",   lambda s: s < 0.4),
        },
    },
    {
        "name": "Case 4 — Correct Refusal (out of scope)",
        "query":    "What will Microsoft's stock price be in 2027?",
        "context":  "Microsoft FY2023 revenue was $211.9 billion.",
        "response": "I cannot predict future stock prices. This information "
                    "is not available in the provided financial filings.",
        "checks": {
            "faithfulness":   ("HIGH",  lambda s: s >= 0.7),
            "relevance":      ("MED",   lambda s: s >= 0.3),
        },
    },
]

def run():
    all_passed = True
    for case in CASES:
        print(f"\n{case['name']}")
        print("─" * 55)
        result = score(
            query    = case["query"],
            context  = case["context"],
            response = case["response"],
        )
        if "error" in result:
            print(f"  ERROR: {result['error']}")
            all_passed = False
            continue
        for metric, (level, check_fn) in case["checks"].items():
            val = result.get(metric, 0.0)
            passed = check_fn(val)
            status = "✓ PASS" if passed else "✗ FAIL"
            if not passed:
                all_passed = False
            print(f"  {metric:<20} should be {level:<5}  got {val:.3f}  {status}")
        print(f"  rationale.faithfulness : "
              f"{result['rationale']['faithfulness'][:80]}")
        print(f"  rationale.relevance    : "
              f"{result['rationale']['relevance'][:80]}")
    print()
    print("=" * 55)
    print("Judge smoke test:", "ALL PASS ✓" if all_passed else "FAILURES DETECTED ✗")
    print("=" * 55)

if __name__ == "__main__":
    run()
