# ragprobe 🔍

**Adversarial testing and evaluation for RAG pipelines and LLM agents.**

Stop shipping AI systems that fail silently. ragprobe automatically generates
adversarial test cases from your documents and scores your pipeline on
faithfulness, relevance, and retrieval quality.

```bash
pip install ragprobe
```

---

## The problem

Your RAG assistant works great on happy-path tests. Then in production it
confidently makes up numbers, answers the wrong question, or retrieves the
wrong chunks — and nobody notices until something goes wrong.

ragprobe gives you a testing track before you ship, and monitoring once you're live.

---

## Quick start

```python
from ragprobe import RAGEvaluator

evaluator = RAGEvaluator(judge="openai")  # or "anthropic"

# Step 1: auto-generate adversarial test cases from your docs
suite = evaluator.generate_tests(
    documents=my_chunks,   # list of text strings
    n_cases=30,
)

# Step 2: run your pipeline and score the outputs
results = evaluator.evaluate(
    pipeline=my_rag_fn,    # fn(query: str) -> (answer: str, contexts: list[str])
    test_suite=suite,
)

# Step 3: see what broke
evaluator.print_summary(results)
```

**Terminal output:**
```
ragprobe evaluation summary

 Metric            Score   Status
 Faithfulness       0.71    ✗
 Relevance          0.88    ✓
 Context recall     0.65    ✗

Pass rate: 14/30 (47%)

Failures by attack type:
  negation             6 failures
  out_of_scope         4 failures
```

---

## Production monitoring

One decorator, zero changes to existing code:

```python
from ragprobe import monitor

@monitor(project="my-rag-app", alert_threshold=0.75)
def my_rag_pipeline(query: str) -> tuple[str, list]:
    answer   = llm.generate(query)
    contexts = retriever.get(query)
    return answer, contexts
```

Every call is logged locally. You get an alert if quality drops below threshold.

---

## Attack types

| Type | What it tests |
|---|---|
| `negation` | Questions where the correct answer contradicts naive assumptions |
| `out_of_scope` | Plausible questions your corpus can't answer (should say "I don't know") |
| `ambiguous` | Questions with multiple valid interpretations |
| `conflicting` | Questions where different chunks give different answers |
| `edge_case` | Exact dates, numbers, names — high hallucination risk |
| `hypothetical` | Reasoning questions that go beyond the documents |

---

## Metrics

| Metric | What it measures |
|---|---|
| `faithfulness` | Is every claim in the answer supported by retrieved context? |
| `relevance` | Does the answer directly address what was asked? |
| `context_recall` | Did the retriever surface the chunks needed to answer? |

---

## Setup

```bash
git clone https://github.com/yourusername/ragprobe
cd ragprobe
pip install -e ".[dev]"
cp .env.example .env   # add your OpenAI key
python examples/demo.py
```

---

## Roadmap

- [x] Adversarial test case generation
- [x] LLM-as-judge scoring (faithfulness, relevance, context recall)
- [x] Production monitoring decorator
- [ ] Web dashboard (score trends, trace history)
- [ ] Regression testing CI/CD integration
- [ ] Slack / email alerts
- [ ] Custom judge rubrics

---

## Contributing

PRs welcome. If you find a failure mode this doesn't catch, open an issue.

---

MIT License
