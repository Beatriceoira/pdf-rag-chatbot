"""Unit tests for prompt engineering."""

from __future__ import annotations

from src.llm.prompts import build_prompt, build_system_prompt


class TestPrompts:
    """Tests for prompt templates."""

    def test_build_prompt_basic(self):
        """Test basic prompt construction."""
        context = ["The vacation policy is 20 days.", "Sick leave is 10 days."]
        question = "How many vacation days are provided?"

        prompt = build_prompt(context, question)

        assert "The vacation policy is 20 days." in prompt
        assert "How many vacation days are provided?" in prompt
        assert "Context:" in prompt
        assert "Question:" in prompt

    def test_build_prompt_empty_context(self):
        """Test prompt with no context."""
        prompt = build_prompt([], "What is the policy?")
        assert "What is the policy?" in prompt

    def test_build_prompt_single_context(self):
        """Test prompt with single context passage."""
        context = ["The answer is 42."]
        prompt = build_prompt(context, "What is the answer?")
        assert "The answer is 42." in prompt

    def test_system_prompt_contains_security_rules(self):
        """System prompt should contain security/injection rules."""
        system = build_system_prompt()
        assert "untrusted" in system.lower() or "ignore" in system.lower()
        assert "system prompt" in system.lower() or "secret" in system.lower()

    def test_system_prompt_does_not_leak_keys(self):
        """System prompt should not contain API key placeholders."""
        system = build_system_prompt()
        assert "sk-" not in system
        assert "api_key" not in system.lower()
