"""RAG evaluation metrics."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger("evaluation")


@dataclass
class EvaluationResult:
    """Result of evaluating a single RAG question."""

    question: str
    expected_answer: str
    actual_answer: str
    expected_sources: list[str]
    retrieved_documents: list[str] = field(default_factory=list)
    hit_at_k: bool = False
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    faithfulness: float | None = None
    groundedness: float | None = None
    citation_correct: bool | None = None
    error: str = ""


class RetrievalMetrics:
    """Compute retrieval-focused evaluation metrics."""

    @staticmethod
    def hit_at_k(expected: list[str], retrieved: list[str], k: int) -> bool:
        """Check if any expected document appears in top-k retrieved."""
        top_k = retrieved[:k]
        return any(exp in top_k for exp in expected)

    @staticmethod
    def precision_at_k(expected: list[str], retrieved: list[str], k: int) -> float:
        """Precision: proportion of retrieved docs that are relevant."""
        top_k = retrieved[:k]
        if not top_k:
            return 0.0
        relevant = sum(1 for doc in top_k if doc in expected)
        return relevant / len(top_k)

    @staticmethod
    def recall_at_k(expected: list[str], retrieved: list[str]) -> float:
        """Recall: proportion of expected docs that were retrieved."""
        if not expected:
            return 0.0
        retrieved_set = set(retrieved)
        relevant_retrieved = sum(1 for exp in expected if exp in retrieved_set)
        return relevant_retrieved / len(expected)

    @staticmethod
    def mean_reciprocal_rank(expected: list[str], retrieved: list[str]) -> float:
        """MRR: 1/rank of first relevant document, or 0 if none found."""
        for i, doc in enumerate(retrieved, 1):
            if doc in expected:
                return 1.0 / i
        return 0.0


class EvaluationRunner:
    """Run RAG evaluations against a dataset."""

    def __init__(self, chat_service):
        self._chat = chat_service
        self._metrics = RetrievalMetrics()

    def evaluate(self, dataset: list[dict], k: int = 5) -> list[EvaluationResult]:
        """Evaluate the RAG system against a test dataset.

        Each dataset item should have:
          - question: str
          - expected_answer: str (optional)
          - expected_sources: list[str] (optional, e.g. ["doc.pdf:3"])
        """
        results = []
        for item in dataset:
            question = item["question"]
            expected_answer = item.get("expected_answer", "")
            expected_sources = item.get("expected_sources", [])

            # Run the chat service
            response = self._chat.ask(question, conversation_id="__eval__")

            # Parse retrieved sources
            retrieved_docs = [f"{s.get('document_name', '?')}:{s.get('page_number', '?')}" for s in response.sources]

            # Compute metrics
            result = EvaluationResult(
                question=question,
                expected_answer=expected_answer,
                actual_answer=response.answer,
                expected_sources=expected_sources,
                retrieved_documents=retrieved_docs,
                hit_at_k=self._metrics.hit_at_k(expected_sources, retrieved_docs, k),
                precision_at_k=self._metrics.precision_at_k(expected_sources, retrieved_docs, k),
                recall_at_k=self._metrics.recall_at_k(expected_sources, retrieved_docs),
                mrr=self._metrics.mean_reciprocal_rank(expected_sources, retrieved_docs),
            )
            results.append(result)

        return results

    def summary(self, results: list[EvaluationResult]) -> dict[str, float]:
        """Compute aggregate metrics over evaluation results."""
        if not results:
            return {}

        hit_rates = [r.hit_at_k for r in results]
        precisions = [r.precision_at_k for r in results]
        recalls = [r.recall_at_k for r in results]
        mrrs = [r.mrr for r in results]

        return {
            "total_questions": len(results),
            "hit_rate": sum(hit_rates) / len(hit_rates),
            "avg_precision": sum(precisions) / len(precisions),
            "avg_recall": sum(recalls) / len(recalls),
            "avg_mrr": sum(mrrs) / len(mrrs),
        }

    def export_results(self, results: list[EvaluationResult], path: str) -> None:
        """Export evaluation results to JSON."""
        data = {
            "results": [
                {
                    "question": r.question,
                    "expected_answer": r.expected_answer,
                    "actual_answer": r.actual_answer,
                    "expected_sources": r.expected_sources,
                    "retrieved_documents": r.retrieved_documents,
                    "hit_at_k": r.hit_at_k,
                    "precision_at_k": r.precision_at_k,
                    "recall_at_k": r.recall_at_k,
                    "mrr": r.mrr,
                }
                for r in results
            ],
            "summary": self.summary(results),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Evaluation results exported to %s", path)
