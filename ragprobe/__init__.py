"""
ragprobe — Adversarial testing & evaluation for RAG pipelines and LLM agents.

Quick start::

    from ragprobe import Evaluator, SafetyGate, RetrievalDiagnostic, ReporterFactory

    gate       = SafetyGate.default(budget_usd=1.0)
    diagnostic = RetrievalDiagnostic()
    evaluator  = Evaluator(gate=gate, diagnostic=diagnostic)

    report = evaluator.run(
        queries    = ["What was Apple revenue in FY2024?"],
        corpus     = my_docs,
        references = ["Apple revenue was $391B in FY2024."],
    )

    reporter = ReporterFactory.get("html")
    reporter.save(report, "report.html")
"""

__version__ = "0.1.0"

from ragprobe.evaluator import Evaluator
from ragprobe.safety import SafetyGate
from ragprobe.retrieval import RetrievalDiagnostic
from ragprobe.reporter import SessionReport, ReporterFactory

__all__ = [
    # Primary public API
    "Evaluator",
    "SafetyGate",
    "RetrievalDiagnostic",
    "SessionReport",
    "ReporterFactory",
    "__version__",
]
