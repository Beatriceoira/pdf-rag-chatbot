"""Unit tests for evaluation evaluator."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from src.evaluation.evaluator import RagEvaluator
from src.evaluation.metrics import EvaluationResult


def test_rag_evaluator_init():
    """Test RagEvaluator initialization."""
    mock_chat_service = Mock()
    evaluator = RagEvaluator(mock_chat_service)
    assert evaluator._runner._chat is mock_chat_service


def test_test_grounding(monkeypatch):
    """Test RagEvaluator.test_grounding method."""
    # Mock chat service and response
    mock_chat_service = Mock()
    mock_response = Mock()
    mock_response.answer = "The sky is blue due to Rayleigh scattering."
    mock_chat_service.ask.return_value = mock_response

    # Create evaluator with mock chat service
    evaluator = RagEvaluator(mock_chat_service)

    # Test grounded answer (expected content in answer)
    result = evaluator.test_grounding(
        question="Why is the sky blue?",
        expected_content="Rayleigh scattering"
    )

    # Verify the call
    mock_chat_service.ask.assert_called_once_with(
        "Why is the sky blue?",
        conversation_id="__grounding_test__"
    )

    # Verify result
    assert isinstance(result, EvaluationResult)
    assert result.question == "Why is the sky blue?"
    assert result.expected_answer == "Rayleigh scattering"
    assert result.actual_answer == "The sky is blue due to Rayleigh scattering."
    assert result.expected_sources == []
    assert result.hit_at_k is True  # Because "Rayleigh scattering" is in the answer


def test_test_grounding_not_grounded(monkeypatch):
    """Test RagEvaluator.test_grounding method when not grounded."""
    # Mock chat service and response
    mock_chat_service = Mock()
    mock_response = Mock()
    mock_response.answer = "The sky appears blue because of atmospheric conditions."
    mock_chat_service.ask.return_value = mock_response

    # Create evaluator with mock chat service
    evaluator = RagEvaluator(mock_chat_service)

    # Test not grounded answer (expected content NOT in answer)
    result = evaluator.test_grounding(
        question="Why is the sky blue?",
        expected_content="Rayleigh scattering"
    )

    # Verify result
    assert result.hit_at_k is False  # Because "Rayleigh scattering" is NOT in the answer


def test_test_no_hallucination(monkeypatch):
    """Test RagEvaluator.test_no_hallucination method."""
    # Mock chat service and response
    mock_chat_service = Mock()
    mock_response = Mock()
    mock_response.answer = "The vacation policy is 20 days per year."
    mock_chat_service.ask.return_value = mock_response

    # Create evaluator with mock chat service
    evaluator = RagEvaluator(mock_chat_service)

    # Test no hallucination (unrelated topic not in answer)
    result = evaluator.test_no_hallucination(
        question="What is the vacation policy?",
        unrelated_topic="quantum physics"
    )

    # Verify the call
    mock_chat_service.ask.assert_called_once_with(
        "What is the vacation policy?",
        conversation_id="__hallucination_test__"
    )

    # Verify result (should be True because "quantum physics" is not in answer)
    assert result is True


def test_test_no_hallucination_detected(monkeypatch):
    """Test RagEvaluator.test_no_hallucination method when hallucination detected."""
    # Mock chat service and response
    mock_chat_service = Mock()
    mock_response = Mock()
    mock_response.answer = "The vacation policy is 20 days per year and includes quantum physics training."
    mock_chat_service.ask.return_value = mock_response

    # Create evaluator with mock chat service
    evaluator = RagEvaluator(mock_chat_service)

    # Test hallucination detected (unrelated topic IS in answer)
    result = evaluator.test_no_hallucination(
        question="What is the vacation policy?",
        unrelated_topic="quantum physics"
    )

    # Verify result (should be False because "quantum physics" IS in answer)
    assert result is False