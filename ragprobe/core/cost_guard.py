"""
cost_guard.py — Track LLM token usage and enforce a USD budget ceiling.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

# ── Model pricing table (USD per 1 000 tokens, early-2025 list prices) ───────
_PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o":                        {"input": 0.0025,  "output": 0.0100},
    "gpt-4o-mini":                   {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo":                   {"input": 0.0100,  "output": 0.0300},
    "gpt-3.5-turbo":                 {"input": 0.0005,  "output": 0.0015},
    # Anthropic
    "claude-3-5-sonnet-20241022":    {"input": 0.003,   "output": 0.015},
    "claude-3-5-haiku-20241022":     {"input": 0.0008,  "output": 0.004},
    "claude-3-opus-20240229":        {"input": 0.015,   "output": 0.075},
    "claude-sonnet-4-6":             {"input": 0.003,   "output": 0.015},
    "claude-haiku-4-5-20251001":     {"input": 0.0008,  "output": 0.004},
}

# Conservative fallback for unknown models
_FALLBACK_PRICE: dict[str, float] = {"input": 0.01, "output": 0.03}


# ── Exceptions ────────────────────────────────────────────────────────────────

class BudgetExceededError(RuntimeError):
    """Raised when an operation would push total spend past the configured budget."""

    def __init__(self, current: float, limit: float) -> None:
        self.current = current
        self.limit = limit
        super().__init__(
            f"Cost budget exceeded: ${current:.4f} spent, limit is ${limit:.4f}"
        )


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class UsageRecord:
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


@dataclass
class CostSummary:
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    total_cost_usd: float
    budget_usd: Optional[float]
    budget_remaining_usd: Optional[float]
    records: list[UsageRecord] = field(default_factory=list)

    @property
    def budget_used_pct(self) -> Optional[float]:
        if self.budget_usd:
            return (self.total_cost_usd / self.budget_usd) * 100
        return None


# ── CostGuard ─────────────────────────────────────────────────────────────────

class CostGuard:
    """
    Thread-safe token-usage tracker with optional USD budget enforcement.

    Usage::

        guard = CostGuard(budget_usd=1.00)

        # After each LLM call:
        guard.record(model="gpt-4o", prompt_tokens=500, completion_tokens=200)

        # Inspect spend at any time:
        summary = guard.summary()
        print(f"${summary.total_cost_usd:.4f} of ${summary.budget_usd:.2f}")
    """

    def __init__(self, budget_usd: Optional[float] = None) -> None:
        self._budget = budget_usd
        self._records: list[UsageRecord] = []
        self._lock = threading.Lock()

    # ── public API ───────────────────────────────────────────────────────────

    def record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        check_after: bool = True,
    ) -> float:
        """
        Record token usage; return the incremental cost in USD.

        Raises BudgetExceededError if the new total exceeds the budget
        (when check_after=True, which is the default).
        """
        cost = self._calc_cost(model, prompt_tokens, completion_tokens)
        rec = UsageRecord(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
        )
        with self._lock:
            self._records.append(rec)
        if check_after:
            self.check()
        return cost

    def check(self) -> None:
        """Raise BudgetExceededError if total spend has crossed the budget."""
        if self._budget is None:
            return
        total = self._total_cost()
        if total > self._budget:
            raise BudgetExceededError(current=total, limit=self._budget)

    def summary(self) -> CostSummary:
        """Return an immutable snapshot of current usage."""
        with self._lock:
            records = list(self._records)
        total_cost = sum(r.cost_usd for r in records)
        remaining = (self._budget - total_cost) if self._budget is not None else None
        return CostSummary(
            total_tokens=sum(r.prompt_tokens + r.completion_tokens for r in records),
            prompt_tokens=sum(r.prompt_tokens for r in records),
            completion_tokens=sum(r.completion_tokens for r in records),
            total_cost_usd=total_cost,
            budget_usd=self._budget,
            budget_remaining_usd=remaining,
            records=records,
        )

    def reset(self) -> None:
        """Clear all recorded usage (e.g. to start a fresh evaluation run)."""
        with self._lock:
            self._records.clear()

    # ── internals ────────────────────────────────────────────────────────────

    def _total_cost(self) -> float:
        with self._lock:
            return sum(r.cost_usd for r in self._records)

    @staticmethod
    def _calc_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
        pricing = _PRICING.get(model)
        if pricing is None:
            for key in _PRICING:
                if model.startswith(key) or key.startswith(model):
                    pricing = _PRICING[key]
                    break
        if pricing is None:
            pricing = _FALLBACK_PRICE
        return (
            (prompt_tokens / 1_000) * pricing["input"]
            + (completion_tokens / 1_000) * pricing["output"]
        )
