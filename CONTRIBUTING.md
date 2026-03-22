# Contributing to ragprobe

Thank you for your interest in contributing. This document covers everything you need to get started.

---

## Dev setup

```bash
git clone https://github.com/rohanvinayaksagvekar/ragprobe.git
cd ragprobe
pip install -e ".[dev]"
python demo.py
```

`pip install -e ".[dev]"` installs ragprobe in editable mode with pytest, pytest-cov, and pydantic (required by the legacy generator/scorer modules). Changes you make to source files are reflected immediately without reinstalling.

To also enable semantic recall scoring via OpenAI embeddings:

```bash
pip install -e ".[dev,openai]"
```

---

## Running tests

**Full test suite:**

```bash
pytest tests/ -v
```

**Single module:**

```bash
pytest tests/test_evaluator.py -v
pytest tests/test_retrieval.py -v
pytest tests/test_reporter.py -v
pytest tests/test_cli.py -v
```

**With coverage:**

```bash
pytest tests/ --cov=ragprobe --cov-report=term-missing
```

All tests are designed to run without API keys. The `test_cli.py` suite uses subprocess-based integration tests; the rest are pure unit tests. No network calls are made.

---

## Code standards

**Structure:**

- Use `@dataclass` for data-holding objects, plain classes for service objects.
- All public methods must have Google-style docstrings with `Parameters`, `Returns`, and a short description.
- Type-annotate all function signatures. Use `from __future__ import annotations` at the top of every module.

**Logging:**

- Use `logging.getLogger(__name__)` at module level. Never use `print()` statements in library code — only in `cli.py` (user-facing output) and `demo.py`.
- Log at `DEBUG` for per-item operations, `INFO` for run-level summaries, `WARNING` for recoverable issues, `ERROR` for failures.

**Dependencies:**

- The core library (`ragprobe/`) must not import anything outside the Python standard library and `pyyaml`. All optional dependencies (`openai`, `pydantic`, etc.) must be lazy-imported inside method bodies so that `import ragprobe` works with only `pyyaml` installed.
- Do not add new runtime dependencies without opening an issue first to discuss the trade-off.

**Tests:**

- Every public method needs at least one test. Edge cases (empty input, mismatched lengths, injection content) must be covered explicitly.
- Tests must be deterministic — no randomness, no real API calls, no network access.
- Use `pytest` fixtures for shared setup. Keep test files under `tests/` with the naming convention `test_<module>.py`.

---

## Adding a new checker

ragprobe's checking pipeline is composed of focused, single-responsibility classes. To add a new check (e.g. a hallucination detector, a citation verifier):

1. **Create the class** in the relevant existing module, or add a new file under `ragprobe/` if the scope is clearly distinct. Follow the existing pattern: plain class, typed inputs, dict returns.

2. **Integrate it** — if the check applies at corpus level, add it to `SafetyGate.check_corpus()`. If it applies per-query, add it to `RetrievalDiagnostic.run()` and surface results in the `per_query` and `aggregate` dicts. If it is a standalone concern, wire it into `Evaluator.run()`.

3. **Register in `__init__.py`** — if the class is part of the public API, export it from `ragprobe/__init__.py` and add it to `__all__`.

4. **Write tests** — add a `tests/test_<module>.py` file or extend an existing one. Cover the happy path, empty input, and any boundary conditions.

5. **Document it** — add a row to the "What ragprobe checks" table in `README.md` and update `CONTRIBUTING.md` if the architecture changes.

---

## PR checklist

Before opening a pull request, verify all of the following:

- [ ] `pytest tests/ -v` passes with zero failures
- [ ] `python -c "from ragprobe import SafetyGate, Evaluator, RetrievalDiagnostic, SessionReport, ReporterFactory"` succeeds
- [ ] `python demo.py` runs end to end without errors
- [ ] All new public methods have docstrings
- [ ] No new runtime dependencies introduced without prior discussion
- [ ] No `print()` statements added to library code (use `logging`)
- [ ] Type annotations present on all new function signatures

---

## Project layout

```
ragprobe/
  __init__.py          — Public API surface (SafetyGate, Evaluator, ...)
  evaluator.py         — Integration core: wires SafetyGate + RetrievalDiagnostic
  safety.py            — SafetyGate: unified validation + injection scanning + cost
  retrieval.py         — RetrievalDiagnostic, ContextRecallScorer, ChunkAnalyzer
  reporter.py          — SessionReport, JSONReporter, HTMLReporter, MarkdownReporter
  cli.py               — CLI entry point: eval, scan, report sub-commands
  core/
    cost_guard.py      — CostGuard: token tracking + budget enforcement
    rate_limiter.py    — RateLimiter: token-bucket RPM/TPM limiting
    validators.py      — InputValidator: document and query validation
    injection_guard.py — InjectionGuard: 20+ pattern corpus scanner
    generator.py       — Legacy: adversarial test case generator (LLM-backed)
    scorer.py          — Legacy: LLM-as-judge scoring
tests/
  test_evaluator.py
  test_retrieval.py
  test_reporter.py
  test_cli.py
  test_generator.py
  test_scorer.py
demo.py                — End-to-end demo, no API key required
```
