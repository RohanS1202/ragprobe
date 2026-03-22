"""
safety.py — Unified safety gate for ragprobe pipelines.

Re-exports the four safety primitives from ragprobe.core and composes them
into a single SafetyGate entry point that the CLI and evaluator can use.

Public API::

    from ragprobe.safety import SafetyGate, CostGuard, RateLimiter
    from ragprobe.safety import InputValidator, InjectionScanner

    gate = SafetyGate()
    clean_docs, report = gate.check_corpus(raw_documents)
    query_report       = gate.check_query("What was Apple revenue?")
"""

from __future__ import annotations

from ragprobe.core.cost_guard     import (  # noqa: F401
    CostGuard, CostSummary, BudgetExceededError,
)
from ragprobe.core.rate_limiter   import (  # noqa: F401
    RateLimiter, RateLimitConfig, RateLimitExceededError,
)
from ragprobe.core.validators     import (  # noqa: F401
    InputValidator, ValidationResult, ValidationIssue,
)
from ragprobe.core.injection_guard import (  # noqa: F401
    InjectionGuard, InjectionDetectedError, CorpusScanReport,
    ScanResult, RiskLevel,
)

# Spec-compatible alias
InjectionScanner = InjectionGuard


class SafetyGate:
    """
    Unified safety gate: input validation + injection scanning in one call.

    Parameters
    ----------
    block_threshold : RiskLevel
        Documents at or above this risk level are removed by check_corpus().
        Default: RiskLevel.HIGH.
    budget_usd : float | None
        Optional spend budget enforced by the internal CostGuard.

    Usage::

        gate = SafetyGate.default(budget_usd=1.0)

        clean_docs, report = gate.check_corpus(raw_documents)
        if report["injection_flagged"]:
            print(f"{report['injection_flagged']} documents blocked")

        q_report = gate.check_query("What was Apple revenue?")
        if not q_report["is_valid"]:
            raise ValueError(q_report["validation_errors"])
    """

    def __init__(
        self,
        block_threshold: RiskLevel = RiskLevel.HIGH,
        budget_usd: float | None = None,
    ) -> None:
        self._validator   = InputValidator()
        self._scanner     = InjectionGuard(block_threshold=block_threshold)
        self._cost_guard  = CostGuard(budget_usd=budget_usd)

    @classmethod
    def default(
        cls,
        budget_usd: float | None = None,
        block_threshold: RiskLevel = RiskLevel.HIGH,
    ) -> "SafetyGate":
        """
        Convenience constructor — identical to ``SafetyGate(...)``.

        Parameters
        ----------
        budget_usd : float | None
            Optional USD spend budget.
        block_threshold : RiskLevel
            Documents at or above this level are blocked.
        """
        return cls(block_threshold=block_threshold, budget_usd=budget_usd)

    def check_corpus(
        self, documents: list[str]
    ) -> tuple[list[str], dict]:
        """
        Validate and injection-scan a document corpus.

        Parameters
        ----------
        documents : list[str]
            Raw document chunks to process.

        Returns
        -------
        tuple[list[str], dict]
            ``(clean_documents, report)`` where *clean_documents* has
            high-risk documents removed and *report* contains:

            - ``original_count``     – number of documents before filtering.
            - ``clean_count``        – number of documents after filtering.
            - ``validation_errors``  – list of validation issue dicts.
            - ``injection_flagged``  – count of documents with any injection match.
            - ``injection_blocked``  – count of documents above block_threshold.
            - ``injection_risk``     – overall corpus risk level (str).
        """
        val_result = self._validator.validate_documents(documents)
        validation_errors = [
            {"code": i.code, "message": i.message, "severity": i.severity}
            for i in val_result.issues
        ]

        clean_docs, scan_report = self._scanner.filter_corpus(documents)

        return clean_docs, {
            "original_count":    len(documents),
            "clean_count":       len(clean_docs),
            "validation_errors": validation_errors,
            "injection_flagged": scan_report.flagged_documents,
            "injection_blocked": len(documents) - len(clean_docs),
            "injection_risk":    scan_report.overall_risk.value,
        }

    def check_query(self, query: str) -> dict:
        """
        Validate and injection-scan a single query string.

        Parameters
        ----------
        query : str
            The user query to check.

        Returns
        -------
        dict
            Keys:
            - ``is_valid``           – True if validation passes and no injection.
            - ``validation_errors``  – list of human-readable error strings.
            - ``injection_risk``     – risk level string.
            - ``injection_matches``  – list of matched pattern names.
        """
        val   = self._validator.validate_query(query)
        scan  = self._scanner.scan(query)
        return {
            "is_valid":          val.is_valid and scan.is_safe,
            "validation_errors": [i.message for i in val.errors],
            "injection_risk":    scan.risk_level.value,
            "injection_matches": [m.pattern_name for m in scan.matches],
        }

    def record_usage(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        model: str = "gpt-4o-mini",
    ) -> None:
        """
        Record approximate token usage for cost tracking.

        Parameters
        ----------
        prompt_tokens : int
            Estimated number of input tokens consumed.
        completion_tokens : int
            Estimated number of output tokens generated.
        model : str
            Model name used for cost calculation.
        """
        try:
            self._cost_guard.record(
                model, prompt_tokens, completion_tokens, check_after=False
            )
        except Exception:
            pass

    def cost_summary(self) -> dict:
        """
        Return a JSON-safe dict snapshot of accumulated cost usage.

        Returns
        -------
        dict
            Keys: total_tokens, prompt_tokens, completion_tokens,
            total_cost_usd, budget_usd, budget_remaining_usd,
            budget_used_pct, calls.
        """
        s = self._cost_guard.summary()
        return {
            "total_tokens":        s.total_tokens,
            "prompt_tokens":       s.prompt_tokens,
            "completion_tokens":   s.completion_tokens,
            "total_cost_usd":      s.total_cost_usd,
            "budget_usd":          s.budget_usd,
            "budget_remaining_usd": s.budget_remaining_usd,
            "budget_used_pct":     s.budget_used_pct,
            "calls":               len(s.records),
        }
