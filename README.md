# ragprobe

[![PyPI version](https://img.shields.io/pypi/v/ragprobe.svg)](https://pypi.org/project/ragprobe/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![CI](https://github.com/rohanvinayaksagvekar/ragprobe/actions/workflows/ci.yml/badge.svg)](https://github.com/rohanvinayaksagvekar/ragprobe/actions)

**Adversarial testing and evaluation framework for RAG pipelines and LLM agents.**

ragprobe diagnoses the four things that break RAG systems in production: prompt injection in your document corpus, retrieval quality gaps (low recall, redundant chunks, poor coverage), runaway API costs, and unsafe query inputs. Most checks are purely lexical — no LLM call required.

---

## Install

```bash
pip install ragprobe
```

For semantic recall scoring using embeddings:

```bash
pip install ragprobe[openai]
```

---

## Quickstart (Python API)

```python
from ragprobe import Evaluator, SafetyGate, RetrievalDiagnostic, ReporterFactory

# Your raw document corpus (strings)
corpus = [
    "Apple reported annual revenue of $391B for fiscal year 2024, up 4% YoY.",
    "NVIDIA gross margin expanded from 57% in FY2023 to 73% in FY2024.",
    "Microsoft cloud revenue reached $135B in FY2024, with Azure up 29%.",
    # ... more documents
]

queries = [
    "What was Apple total revenue in FY2024?",
    "How did NVIDIA gross margin change year over year?",
]

references = [
    "Apple revenue was $391B in FY2024, up 4%.",
    "NVIDIA gross margin expanded from 57% to 73% in FY2024.",
]

# Wire the pipeline
gate       = SafetyGate.default(budget_usd=1.0)
diagnostic = RetrievalDiagnostic()
evaluator  = Evaluator(gate=gate, diagnostic=diagnostic, model="gpt-4o-mini")

# Run evaluation
report = evaluator.run(queries=queries, corpus=corpus, references=references)

# Inspect results
print(f"Mean recall:     {report.aggregate['mean_recall']:.3f}")
print(f"Mean redundancy: {report.aggregate['mean_redundancy']:.3f}")
print(f"Safety events:   {len(report.safety_events)}")

# Save reports
ReporterFactory.get("html").save(report, "report.html")
ReporterFactory.get("json").save(report, "results.json")
```

---

## Quickstart (CLI)

**Evaluate a corpus against a config file:**

```bash
ragprobe eval --config config.yaml --output results.json
```

Sample output:
```
✓ eval complete — 3 queries, 2 findings (0 CRITICAL, 1 WARNING, 1 INFO)
```

**Scan a corpus directory for injection patterns:**

```bash
ragprobe scan --corpus ./docs/
# With a specific query for coverage analysis:
ragprobe scan --corpus ./docs/ --query "What was Apple revenue in FY2024?"
# Fail the pipeline if injection is found (useful in CI):
ragprobe scan --corpus ./docs/ --raise-on-injection
```

Sample output:
```
✓ scan complete — 42 documents, 1 flagged, 1 blocked
```

**Generate an HTML or JSON report from saved results:**

```bash
ragprobe report --input results.json --format html --output report.html
ragprobe report --input results.json --format json
```

---

## What ragprobe checks

| Concern | What it detects | Class |
| --- | --- | --- |
| **Safety** | Prompt injection patterns in corpus documents (20+ regex patterns: instruction overrides, persona hijacking, jailbreak keywords, delimiter injection) | `SafetyGate`, `InjectionGuard` |
| **Retrieval quality** | Low context recall, redundant chunks (high pairwise Jaccard overlap), coverage gaps (query terms absent from all retrieved chunks), chunk length anomalies | `RetrievalDiagnostic`, `ChunkAnalyzer`, `ContextRecallScorer` |
| **Cost** | Token usage tracking and USD budget enforcement across model calls | `SafetyGate` (via `CostGuard`) |
| **Input validation** | Malformed queries, empty or non-printable documents, type errors before they reach your LLM | `SafetyGate` (via `InputValidator`) |

---

## Architecture

```
                   ┌──────────────┐
   raw corpus ────►│  SafetyGate  │── injection events ──► safety_events[]
                   │  (validate + │
   raw queries ───►│   scan)      │── clean corpus
                   └──────┬───────┘
                          │ clean corpus
                          ▼
                   ┌──────────────┐
                   │   Evaluator  │◄── top_k_retrieve() per query
                   │  (orchestr.) │
                   └──────┬───────┘
                          │ queries + chunks
                          ▼
                   ┌──────────────────┐
                   │ RetrievalDiag.   │── recall, redundancy,
                   │ run()            │   coverage, findings
                   └──────┬───────────┘
                          │
                          ▼
                   ┌──────────────┐
                   │ SessionReport│── save() ──► results.json
                   └──────┬───────┘
                          │
                          ▼
                   ┌──────────────┐
                   │   Reporter   │── HTML / JSON / Markdown
                   │  Factory     │
                   └──────────────┘
```

---

## Configuration (config.yaml)

```yaml
# Model used for cost estimation in the report
model: gpt-4o-mini

# USD spend budget — evaluation aborts if exceeded (optional)
budget_usd: 2.0

# Number of chunks to retrieve per query
top_k: 5

# Set true to compute semantic recall via embeddings (requires ragprobe[openai])
semantic_recall: false

# Queries to evaluate against the corpus
queries:
  - "What was Apple total revenue in fiscal year 2024?"
  - "How did NVIDIA gross margin change year over year?"
  - "What was Microsoft cloud revenue in FY2024?"

# Path to directory of .txt corpus files
corpus_dir: ./docs

# Optional ground-truth reference answers (one per query, same order).
# When provided, recall scores are computed. Omit to skip recall scoring.
references:
  - "Apple revenue was $391B in FY2024, up 4% year over year."
  - "NVIDIA gross margin expanded from 57% in FY2023 to 73% in FY2024."
  - "Microsoft cloud revenue was $135B in FY2024 with Azure growing 29%."
```

---

## Findings severity levels

| Severity | Trigger | What to do |
| --- | --- | --- |
| **CRITICAL** | Mean context recall < 30%, or mean chunk redundancy > 60% | Increase `top_k`, re-index with a domain-specific embedding model, or apply Maximum Marginal Relevance (MMR) reranking to diversify results |
| **WARNING** | Mean recall 30–50%, or >50% of meaningful query terms missing from retrieved chunks | Add a cross-encoder re-ranker, increase chunk overlap, or add BM25 alongside dense retrieval (hybrid search) |
| **INFO** | Mean chunk length below 80 words or above 600 words | Adjust chunking strategy — target 150–400 words per chunk for most RAG use cases to avoid splitting answers across boundaries |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for full details.

**Quick start for contributors:**

```bash
git clone https://github.com/rohanvinayaksagvekar/ragprobe.git
cd ragprobe
pip install -e ".[dev]"
pytest tests/ -v
python demo.py
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
