"""
demo.py — End-to-end demo of ragprobe using fake HR policy documents.

Run this to verify your setup is working:
    python examples/demo.py

Requirements:
    - OPENAI_API_KEY set in your .env file
    - Dependencies installed: pip install -e ".[dev]"
"""

import os
import sys

# Make sure ragprobe is importable when running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragprobe import RAGEvaluator


# ── Sample document corpus ─────────────────────────────────────────────────

DOCUMENTS = [
    """Remote Work Policy — Section 3.2
    Employees may work remotely up to 3 days per week with manager approval.
    Remote work is NOT permitted for employees on performance improvement plans (PIPs).
    All remote work must be conducted from within the United States.
    Employees working remotely must be available during core hours: 10am–3pm EST.""",

    """Expense Reimbursement Policy — Section 7
    Meal expenses are reimbursable up to $75 per person per day during business travel.
    Alcohol is NOT reimbursable under any circumstances.
    Receipts are required for all expenses over $25.
    Reimbursement requests must be submitted within 30 days of the expense.
    International travel requires pre-approval from the CFO for trips exceeding $5,000.""",

    """Equipment Policy — Section 2
    New employees receive a MacBook Pro 14" (M3) or equivalent Windows laptop.
    Personal devices may NOT be used to access company systems without MDM enrollment.
    Lost or stolen equipment must be reported to IT within 24 hours.
    Employees are responsible for equipment damage not covered by manufacturer warranty.
    Equipment must be returned within 5 business days of termination.""",

    """Leave Policy — Section 4
    Full-time employees accrue 15 days of PTO per year for the first 3 years.
    After 3 years, PTO accrual increases to 20 days per year.
    PTO does NOT roll over — unused days are forfeited on December 31.
    Employees must give at least 2 weeks notice for planned absences over 3 days.
    Sick leave is separate: 10 days per year, carries over up to 30 days.""",
]


# ── Fake RAG pipeline (deliberately makes mistakes) ────────────────────────

def fake_rag_pipeline(query: str) -> tuple[str, list[str]]:
    """
    Simulates a RAG pipeline that sometimes hallucinates.
    In a real project this would be your actual LangChain / LlamaIndex pipeline.
    """
    query_lower = query.lower()

    # Simple keyword-based retrieval
    retrieved = []
    for doc in DOCUMENTS:
        if any(w in query_lower for w in ["remote", "work from home", "wfh", "pip"]):
            if "Remote Work" in doc:
                retrieved.append(doc)
        elif any(w in query_lower for w in ["expense", "meal", "receipt", "reimburse", "alcohol"]):
            if "Expense" in doc:
                retrieved.append(doc)
        elif any(w in query_lower for w in ["laptop", "equipment", "device", "computer", "mdm"]):
            if "Equipment" in doc:
                retrieved.append(doc)
        elif any(w in query_lower for w in ["pto", "vacation", "leave", "sick", "days off", "roll"]):
            if "Leave" in doc:
                retrieved.append(doc)

    if not retrieved:
        retrieved = [DOCUMENTS[0]]

    # Deliberately wrong answers to simulate hallucination
    hallucination_map = {
        "alcohol":      "Yes, alcohol is reimbursable when entertaining clients.",
        "pto roll":     "PTO rolls over up to 10 days per year.",
        "5 business":   "Employees must return equipment within 10 business days.",
        "performance":  "Employees on PIPs can still work remotely 1 day per week.",
    }

    for trigger, wrong_answer in hallucination_map.items():
        if trigger in query_lower:
            return wrong_answer, retrieved

    context = retrieved[0] if retrieved else ""
    return f"Based on company policy: {context[:200]}...", retrieved


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("\n ragprobe — adversarial RAG evaluation demo\n")
    print("=" * 50)

    # Check for API key
    from dotenv import load_dotenv
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("API_KEY"):
        print("\nError: No API key found.")
        print("Create a .env file with: OPENAI_API_KEY=sk-your-key-here")
        return

    print("\nInitialising evaluator...")
    evaluator = RAGEvaluator(judge="openai")

    # Step 1: generate adversarial test cases
    print("\nStep 1: Generating adversarial test cases from HR policy docs...")
    suite = evaluator.generate_tests(
        documents  = DOCUMENTS,
        n_cases    = 6,
        suite_name = "hr_policy_eval",
    )

    print(f"\nGenerated {suite.total} test cases:")
    for attack_type, count in suite.by_type.items():
        print(f"  {attack_type:<20} {count}")

    print("\nSample questions generated:")
    for tc in suite.cases[:3]:
        print(f"\n  [{tc.attack_type.value}]")
        print(f"  Q: {tc.question}")
        print(f"  Expected: {tc.expected_behavior[:80]}...")

    # Step 2: run the pipeline and score outputs
    print("\n\nStep 2: Running pipeline on each test case and scoring...")
    results = evaluator.evaluate(
        pipeline   = fake_rag_pipeline,
        test_suite = suite,
    )

    # Step 3: print summary report
    print("\n")
    evaluator.print_summary(results)

    # Step 4: deep dive on worst case
    summary = evaluator.summarise(results)
    if summary.worst_cases:
        worst = summary.worst_cases[0]
        print(f"\nWorst case deep dive (score={worst.overall_score:.2f}):")
        print(f"  Q:      {worst.question}")
        print(f"  Answer: {worst.answer[:120]}")
        if worst.faithfulness and worst.faithfulness.violations:
            print(f"  Violations: {worst.faithfulness.violations}")

    print("\nDone! Your ragprobe setup is working correctly.\n")


if __name__ == "__main__":
    main()
