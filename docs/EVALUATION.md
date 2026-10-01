# RAG Evaluation Guide

## Evaluation Dataset

Create a JSON dataset:

```json
[
  {
    "question": "What is the refund period?",
    "expected_answer": "30 days",
    "expected_sources": ["policy.pdf:4"]
  }
]
```

## Running Evaluation

```bash
python scripts/evaluate_rag.py --dataset evaluation_dataset.json --output results.json
```

## Metrics

| Metric | Description |
|--------|-------------|
| Hit@K | Whether any expected document appears in top-K retrieved |
| Precision@K | Proportion of retrieved docs that are relevant |
| Recall@K | Proportion of expected docs that were retrieved |
| MRR | Mean Reciprocal Rank — 1/rank of first relevant doc |

## Configuration Experiments

Compare chunk sizes and retrieval depths:

```bash
# Test different chunk sizes
CHUNK_SIZE=500 python scripts/evaluate_rag.py ...
CHUNK_SIZE=1000 python scripts/evaluate_rag.py ...
CHUNK_SIZE=1500 python scripts/evaluate_rag.py ...
```

Record results in the table below:

| Configuration | Hit@5 | MRR | Notes |
|--------------|-------|-----|-------|
| chunk=500, k=5 | — | — | _run evaluation_ |
| chunk=1000, k=5 | — | — | _default_ |
| chunk=1500, k=5 | — | — | _run evaluation_ |
| chunk=1000, k=3 | — | — | _run evaluation_ |
| chunk=1000, k=10 | — | — | _run evaluation_ |

> ⚠️ Only fill in values from actual evaluation runs. Do not fabricate results.
