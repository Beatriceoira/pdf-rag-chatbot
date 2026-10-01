"""Unit tests for evaluation metrics."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from src.evaluation.metrics import EvaluationResult, EvaluationRunner, RetrievalMetrics


def test_evaluation_result_creation():
    """Test EvaluationResult dataclass creation."""
    result = EvaluationResult(
        question="What is AI?",
        expected_answer="AI is artificial intelligence.",
        actual_answer="AI stands for artificial intelligence.",
        expected_sources=["doc1.pdf:1", "doc2.pdf:3"],
        retrieved_documents=["doc1.pdf:1", "doc3.pdf:2"],
        hit_at_k=True,
        precision_at_k=0.5,
        recall_at_k=0.5,
        mrr=0.5,
        faithfulness=0.8,
        groundedness=0.9,
        citation_correct=True,
        error="",
    )
    assert result.question == "What is AI?"
    assert result.expected_answer == "AI is artificial intelligence."
    assert result.actual_answer == "AI stands for artificial intelligence."
    assert result.expected_sources == ["doc1.pdf:1", "doc2.pdf:3"]
    assert result.retrieved_documents == ["doc1.pdf:1", "doc3.pdf:2"]
    assert result.hit_at_k is True
    assert result.precision_at_k == 0.5
    assert result.recall_at_k == 0.5
    assert result.mrr == 0.5
    assert result.faithfulness == 0.8
    assert result.groundedness == 0.9
    assert result.citation_correct is True
    assert result.error == ""


def test_hit_at_k():
    """Test hit_at_k metric."""
    # Hit in top k
    assert RetrievalMetrics.hit_at_k(["doc1", "doc2"], ["doc1", "doc3"], 2) is True
    # Hit not in top k
    assert RetrievalMetrics.hit_at_k(["doc1", "doc2"], ["doc3", "doc4", "doc1"], 2) is False
    # Empty expected
    assert RetrievalMetrics.hit_at_k([], ["doc1", "doc2"], 2) is False
    # Empty retrieved
    assert RetrievalMetrics.hit_at_k(["doc1"], [], 2) is False
    # k larger than retrieved list
    assert RetrievalMetrics.hit_at_k(["doc1"], ["doc2", "doc1"], 5) is True


def test_precision_at_k():
    """Test precision_at_k metric."""
    # Perfect precision
    assert RetrievalMetrics.precision_at_k(["doc1", "doc2"], ["doc1", "doc2"], 2) == 1.0
    # Half precision
    assert RetrievalMetrics.precision_at_k(["doc1", "doc2"], ["doc1", "doc3"], 2) == 0.5
    # Zero precision
    assert RetrievalMetrics.precision_at_k(["doc1", "doc2"], ["doc3", "doc4"], 2) == 0.0
    # Empty retrieved
    assert RetrievalMetrics.precision_at_k(["doc1", "doc2"], [], 2) == 0.0
    # k larger than retrieved
    assert RetrievalMetrics.precision_at_k(["doc1", "doc2"], ["doc1", "doc3", "doc4"], 5) == 1/3
    # k zero? (but k is int, assume k>=1)
    # Actually, if k=0, top_k empty -> return 0.0 (as per code)
    assert RetrievalMetrics.precision_at_k(["doc1"], [], 0) == 0.0


def test_recall_at_k():
    """Test recall_at_k metric."""
    # Perfect recall
    assert RetrievalMetrics.recall_at_k(["doc1", "doc2"], ["doc1", "doc2", "doc3"]) == 1.0
    # Half recall
    assert RetrievalMetrics.recall_at_k(["doc1", "doc2", "doc3"], ["doc1", "doc4"]) == 1/3
    # Zero recall
    assert RetrievalMetrics.recall_at_k(["doc1", "doc2"], ["doc3", "doc4"]) == 0.0
    # Empty expected
    assert RetrievalMetrics.recall_at_k([], ["doc1", "doc2"]) == 0.0
    # Empty retrieved
    assert RetrievalMetrics.recall_at_k(["doc1", "doc2"], []) == 0.0


def test_mean_reciprocal_rank():
    """Test mean_reciprocal_rank metric."""
    # First relevant at position 1
    assert RetrievalMetrics.mean_reciprocal_rank(["doc1", "doc2"], ["doc1", "doc3"]) == 1.0
    # First relevant at position 2
    assert RetrievalMetrics.mean_reciprocal_rank(["doc1", "doc2"], ["doc3", "doc1"]) == 0.5
    # First relevant at position 3
    assert RetrievalMetrics.mean_reciprocal_rank(["doc1", "doc2"], ["doc3", "doc4", "doc1"]) == 1/3
    # No relevant
    assert RetrievalMetrics.mean_reciprocal_rank(["doc1", "doc2"], ["doc3", "doc4"]) == 0.0
    # Empty expected
    assert RetrievalMetrics.mean_reciprocal_rank([], ["doc1", "doc2"]) == 0.0
    # Empty retrieved
    assert RetrievalMetrics.mean_reciprocal_rank(["doc1"], []) == 0.0


def test_evaluation_runner_init():
    """Test EvaluationRunner initialization."""
    mock_chat_service = Mock()
    runner = EvaluationRunner(mock_chat_service)
    assert runner._chat is mock_chat_service
    assert isinstance(runner._metrics, RetrievalMetrics)


def test_evaluation_runner_evaluate(monkeypatch):
    """Test EvaluationRunner.evaluate method."""
    # Mock chat service
    mock_chat_service = Mock()
    mock_response = Mock()
    mock_response.answer = "Test answer"
    mock_response.sources = [
        {"document_name": "doc1.pdf", "page_number": 1},
        {"document_name": "doc2.pdf", "page_number": 2},
    ]
    mock_chat_service.ask.return_value = mock_response

    runner = EvaluationRunner(mock_chat_service)
    dataset = [
        {
            "question": "What is A?",
            "expected_answer": "A is B",
            "expected_sources": ["doc1.pdf:1", "doc2.pdf:2"],
        },
        {
            "question": "What is C?",
            "expected_answer": "C is D",
            "expected_sources": ["doc3.pdf:1"],
        },
    ]

    results = runner.evaluate(dataset, k=2)

    # Check that ask was called for each item
    assert mock_chat_service.ask.call_count == 2
    mock_chat_service.ask.assert_any_call("What is A?", conversation_id="__eval__")
    mock_chat_service.ask.assert_any_call("What is C?", conversation_id="__eval__")

    # Check results
    assert len(results) == 2
    assert isinstance(results[0], EvaluationResult)
    assert results[0].question == "What is A?"
    assert results[0].expected_answer == "A is B"
    assert results[0].actual_answer == "Test answer"
    assert results[0].expected_sources == ["doc1.pdf:1", "doc2.pdf:2"]
    assert results[0].retrieved_documents == ["doc1.pdf:1", "doc2.pdf:2"]

    # Check metrics for first item (both expected sources in retrieved -> hit_at_k=True, precision=1.0, recall=1.0, mrr=1.0)
    assert results[0].hit_at_k is True
    assert results[0].precision_at_k == 1.0
    assert results[0].recall_at_k == 1.0
    assert results[0].mrr == 1.0

    # Second item: only one expected source, but we retrieved two documents (doc1 and doc2) -> none match expected
    assert results[1].question == "What is C?"
    assert results[1].expected_sources == ["doc3.pdf:1"]
    assert results[1].retrieved_documents == ["doc1.pdf:1", "doc2.pdf:2"]
    assert results[1].hit_at_k is False
    assert results[1].precision_at_k == 0.0
    assert results[1].recall_at_k == 0.0
    assert results[1].mrr == 0.0


def test_evaluation_runner_summary():
    """Test EvaluationRunner.summary method."""
    runner = EvaluationRunner(Mock())
    results = [
        EvaluationResult(
            question="Q1",
            expected_answer="A1",
            actual_answer="A1",
            expected_sources=["s1"],
            hit_at_k=True,
            precision_at_k=1.0,
            recall_at_k=1.0,
            mrr=1.0,
        ),
        EvaluationResult(
            question="Q2",
            expected_answer="A2",
            actual_answer="A2",
            expected_sources=["s2"],
            hit_at_k=False,
            precision_at_k=0.0,
            recall_at_k=0.0,
            mrr=0.0,
        ),
    ]
    summary = runner.summary(results)
    assert summary["total_questions"] == 2
    assert summary["hit_rate"] == 0.5
    assert summary["avg_precision"] == 0.5
    assert summary["avg_recall"] == 0.5
    assert summary["avg_mrr"] == 0.5

    # Empty results
    assert runner.summary([]) == {}


def test_evaluation_runner_export_results(tmp_path):
    """Test EvaluationRunner.export_results method."""
    mock_chat_service = Mock()
    runner = EvaluationRunner(mock_chat_service)
    results = [
        EvaluationResult(
            question="Q1",
            expected_answer="A1",
            actual_answer="A1",
            expected_sources=["s1"],
            retrieved_documents=["s1"],
            hit_at_k=True,
            precision_at_k=1.0,
            recall_at_k=1.0,
            mrr=1.0,
        )
    ]
    export_path = tmp_path / "results.json"
    runner.export_results(results, str(export_path))

    # Check that file was created and contains expected data
    assert export_path.exists()
    import json
    with open(export_path) as f:
        data = json.load(f)
    assert "results" in data
    assert "summary" in data
    assert len(data["results"]) == 1
    assert data["results"][0]["question"] == "Q1"
    assert data["results"][0]["hit_at_k"] is True
    assert data["summary"]["total_questions"] == 1