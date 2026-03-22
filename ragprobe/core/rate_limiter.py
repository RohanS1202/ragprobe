"""
rate_limiter.py — Token-bucket rate limiter for LLM API calls.

Enforces per-model requests-per-minute (RPM) and tokens-per-minute (TPM)
limits to prevent hitting provider rate limits and runaway API spend.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional


# ── Default limits ────────────────────────────────────────────────────────────
# Conservative defaults; callers can override via RateLimiter(custom_limits=...)

@dataclass
class RateLimitConfig:
    """Per-model rate limit settings."""
    requests_per_minute: int = 60
    tokens_per_minute: int = 100_000


_DEFAULT_LIMITS: dict[str, RateLimitConfig] = {
    # OpenAI
    "gpt-4o":              RateLimitConfig(requests_per_minute=60,  tokens_per_minute=800_000),
    "gpt-4o-mini":         RateLimitConfig(requests_per_minute=500, tokens_per_minute=2_000_000),
    "gpt-4-turbo":         RateLimitConfig(requests_per_minute=30,  tokens_per_minute=300_000),
    "gpt-3.5-turbo":       RateLimitConfig(requests_per_minute=500, tokens_per_minute=2_000_000),
    # Anthropic
    "claude-3-5-sonnet-20241022": RateLimitConfig(requests_per_minute=50, tokens_per_minute=200_000),
    "claude-3-5-haiku-20241022":  RateLimitConfig(requests_per_minute=50, tokens_per_minute=200_000),
    "claude-3-opus-20240229":     RateLimitConfig(requests_per_minute=50, tokens_per_minute=200_000),
    "claude-sonnet-4-6":          RateLimitConfig(requests_per_minute=50, tokens_per_minute=200_000),
    "claude-haiku-4-5-20251001":  RateLimitConfig(requests_per_minute=50, tokens_per_minute=200_000),
}


# ── Exceptions ────────────────────────────────────────────────────────────────

class RateLimitExceededError(RuntimeError):
    """Raised when rate limit is hit and block=False."""
    pass


# ── Token bucket ──────────────────────────────────────────────────────────────

class _TokenBucket:
    """Thread-safe token bucket for a single resource (requests or tokens)."""

    def __init__(self, capacity: int, refill_rate_per_sec: float) -> None:
        self._capacity = float(capacity)
        self._tokens = float(capacity)
        self._rate = refill_rate_per_sec
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def consume(self, amount: float = 1.0, *, block: bool = True) -> float:
        """
        Consume *amount* tokens. Returns seconds waited (0.0 if instant).
        Raises RateLimitExceededError if block=False and capacity is insufficient.
        """
        with self._lock:
            self._refill()
            if self._tokens >= amount:
                self._tokens -= amount
                return 0.0
            if not block:
                raise RateLimitExceededError(
                    f"Rate limit: need {amount:.0f} tokens, "
                    f"only {self._tokens:.1f} available"
                )
            deficit = amount - self._tokens
            wait_secs = deficit / self._rate

        # Sleep outside the lock so other threads aren't blocked
        time.sleep(wait_secs)

        with self._lock:
            self._refill()
            self._tokens -= amount
        return wait_secs

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now


# ── RateLimiter ───────────────────────────────────────────────────────────────

class RateLimiter:
    """
    Per-model rate limiter using token buckets for both RPM and TPM.

    Usage::

        limiter = RateLimiter()

        # Call before each LLM API request:
        limiter.acquire("gpt-4o", estimated_tokens=1_500)

        # … make the API call …
    """

    def __init__(
        self,
        custom_limits: Optional[dict[str, RateLimitConfig]] = None,
        *,
        block: bool = True,
    ) -> None:
        """
        Args:
            custom_limits: Override or extend the built-in per-model defaults.
            block:         If True (default), sleep until capacity is available.
                           If False, raise RateLimitExceededError immediately.
        """
        self._limits = {**_DEFAULT_LIMITS, **(custom_limits or {})}
        self._block = block
        self._req_buckets: dict[str, _TokenBucket] = {}
        self._tok_buckets: dict[str, _TokenBucket] = {}
        self._init_lock = threading.Lock()

    def acquire(self, model: str, estimated_tokens: int = 1_000) -> float:
        """
        Reserve capacity for one request that will consume *estimated_tokens*.

        Returns total seconds spent waiting (0.0 if no throttling occurred).
        """
        req_bucket, tok_bucket = self._get_buckets(model)
        waited = req_bucket.consume(1.0, block=self._block)
        waited += tok_bucket.consume(float(estimated_tokens), block=self._block)
        return waited

    # ── internals ────────────────────────────────────────────────────────────

    def _get_buckets(self, model: str) -> tuple[_TokenBucket, _TokenBucket]:
        with self._init_lock:
            if model not in self._req_buckets:
                cfg = self._resolve_config(model)
                self._req_buckets[model] = _TokenBucket(
                    capacity=cfg.requests_per_minute,
                    refill_rate_per_sec=cfg.requests_per_minute / 60.0,
                )
                self._tok_buckets[model] = _TokenBucket(
                    capacity=cfg.tokens_per_minute,
                    refill_rate_per_sec=cfg.tokens_per_minute / 60.0,
                )
        return self._req_buckets[model], self._tok_buckets[model]

    def _resolve_config(self, model: str) -> RateLimitConfig:
        if model in self._limits:
            return self._limits[model]
        for key in self._limits:
            if model.startswith(key) or key.startswith(model):
                return self._limits[key]
        return RateLimitConfig()  # generic fallback
