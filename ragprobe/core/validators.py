"""
validators.py — Input validation for document corpora and queries.

Validates length, encoding, and structure before chunks enter the pipeline,
preventing garbage-in/garbage-out and surfacing data quality issues early.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Optional


# ── Defaults ──────────────────────────────────────────────────────────────────
MAX_CHUNK_CHARS = 50_000    # ~12 k tokens at ~4 chars/token
MIN_CHUNK_CHARS = 10
MAX_QUERY_CHARS = 2_000
MIN_QUERY_CHARS = 3
MAX_DOCUMENTS   = 10_000


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class ValidationIssue:
    code:     str   # machine-readable  e.g. "too_long"
    message:  str   # human-readable description
    severity: str   # "error" | "warning"


@dataclass
class ValidationResult:
    is_valid: bool
    issues:   list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def __bool__(self) -> bool:
        return self.is_valid


# ── InputValidator ────────────────────────────────────────────────────────────

class InputValidator:
    """
    Validates document chunks and user queries before they enter the RAG pipeline.

    Usage::

        validator = InputValidator()

        result = validator.validate_documents(chunks)
        if not result:
            for issue in result.errors:
                print(issue.message)

        result = validator.validate_query("What is the annual revenue?")
        if not result:
            raise ValueError(result.errors[0].message)
    """

    def __init__(
        self,
        max_chunk_chars:   int  = MAX_CHUNK_CHARS,
        min_chunk_chars:   int  = MIN_CHUNK_CHARS,
        max_query_chars:   int  = MAX_QUERY_CHARS,
        min_query_chars:   int  = MIN_QUERY_CHARS,
        max_documents:     int  = MAX_DOCUMENTS,
        allow_non_printable: bool = False,
    ) -> None:
        self.max_chunk_chars    = max_chunk_chars
        self.min_chunk_chars    = min_chunk_chars
        self.max_query_chars    = max_query_chars
        self.min_query_chars    = min_query_chars
        self.max_documents      = max_documents
        self.allow_non_printable = allow_non_printable

    # ── public API ───────────────────────────────────────────────────────────

    def validate_documents(self, documents: list[str]) -> ValidationResult:
        """Validate a list of document chunks; return a consolidated result."""
        issues: list[ValidationIssue] = []

        if not isinstance(documents, list):
            return ValidationResult(
                is_valid=False,
                issues=[ValidationIssue("type_error", "documents must be a list", "error")],
            )

        if len(documents) == 0:
            issues.append(ValidationIssue("empty_corpus", "Document list is empty", "error"))
            return ValidationResult(is_valid=False, issues=issues)

        if len(documents) > self.max_documents:
            issues.append(ValidationIssue(
                "too_many_documents",
                f"Corpus has {len(documents):,} documents; limit is {self.max_documents:,}",
                "error",
            ))

        for idx, chunk in enumerate(documents):
            issues.extend(self._validate_chunk(chunk, idx))

        has_errors = any(i.severity == "error" for i in issues)
        return ValidationResult(is_valid=not has_errors, issues=issues)

    def validate_query(self, query: str) -> ValidationResult:
        """Validate a single query string."""
        issues: list[ValidationIssue] = []

        if not isinstance(query, str):
            return ValidationResult(
                is_valid=False,
                issues=[ValidationIssue("type_error", "query must be a string", "error")],
            )

        stripped = query.strip()

        if len(stripped) < self.min_query_chars:
            issues.append(ValidationIssue(
                "too_short",
                f"Query is too short ({len(stripped)} chars; minimum {self.min_query_chars})",
                "error",
            ))

        if len(stripped) > self.max_query_chars:
            issues.append(ValidationIssue(
                "too_long",
                f"Query is too long ({len(stripped):,} chars; maximum {self.max_query_chars:,})",
                "error",
            ))

        if not self.allow_non_printable and _has_non_printable(stripped):
            issues.append(ValidationIssue(
                "non_printable_chars",
                "Query contains non-printable control characters",
                "warning",
            ))

        has_errors = any(i.severity == "error" for i in issues)
        return ValidationResult(is_valid=not has_errors, issues=issues)

    # ── internals ────────────────────────────────────────────────────────────

    def _validate_chunk(self, chunk: object, idx: int) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        prefix = f"Document[{idx}]"

        if not isinstance(chunk, str):
            issues.append(ValidationIssue(
                "type_error",
                f"{prefix}: expected str, got {type(chunk).__name__}",
                "error",
            ))
            return issues

        if len(chunk) < self.min_chunk_chars:
            issues.append(ValidationIssue(
                "chunk_too_short",
                f"{prefix}: {len(chunk)} chars (minimum {self.min_chunk_chars})",
                "warning",
            ))

        if len(chunk) > self.max_chunk_chars:
            issues.append(ValidationIssue(
                "chunk_too_long",
                f"{prefix}: {len(chunk):,} chars (maximum {self.max_chunk_chars:,})",
                "error",
            ))

        if not self.allow_non_printable and _has_non_printable(chunk):
            issues.append(ValidationIssue(
                "non_printable_chars",
                f"{prefix}: contains non-printable control characters",
                "warning",
            ))

        return issues


# ── helpers ───────────────────────────────────────────────────────────────────

def _has_non_printable(text: str) -> bool:
    """Return True if *text* contains control characters (tabs/newlines are fine)."""
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("C") and ch not in ("\n", "\r", "\t"):
            return True
    return False
