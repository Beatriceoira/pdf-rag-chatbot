"""RAG-specific evaluation tests."""

from __future__ import annotations

from src.evaluation.metrics import EvaluationResult, EvaluationRunner


class RagEvaluator:
    """High-level evaluator for RAG quality checks."""

    def __init__(self, chat_service):
        self._runner = EvaluationRunner(chat_service)

    def test_grounding(self, question: str, expected_content: str) -> EvaluationResult:
        """Test that the answer contains expected content (groundedness check)."""
        response = self._runner.chat.ask(question, conversation_id="__grounding_test__")
        is_grounded = expected_content.lower() in response.answer.lower()
        return EvaluationResult(
            question=question,
            expected_answer=expected_content,
            actual_answer=response.answer,
            expected_sources=[],
            hit_at_k=is_grounded,
        )

    def test_no_hallucination(self, question: str, unrelated_topic: str) -> bool:
        """Test that the model doesn't hallucinate about unrelated topics."""
        response = self._runner.chat.ask(question, conversation_id="__hallucination_test__")
        return unrelated_topic.lower() not in response.answer.lower()
