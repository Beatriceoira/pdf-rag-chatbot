"""LLM provider factory with OpenAI, Ollama, and Anthropic support."""

from __future__ import annotations

from src.config.settings import Settings
from src.llm.openai_chat import OpenAIChat
from src.utils.exceptions import LLMError
from src.utils.logging import get_logger

logger = get_logger("llm")

try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None  # type: ignore

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None  # type: ignore


class LLMFactory:
    """Factory for creating LLM chat models based on provider settings."""

    @staticmethod
    def create(settings: Settings):
        """Create and return the appropriate chat model."""
        provider = settings.llm_provider

        if provider == "openai":
            if not settings.openai_api_key:
                raise LLMError("OpenAI API key is required. Set OPENAI_API_KEY in your environment.")
            return OpenAIChat(model=settings.llm_model)

        elif provider == "ollama":
            if ChatOllama is None:
                raise LLMError("langchain-ollama is not installed. Run: pip install langchain-ollama")
            return ChatOllama(model=settings.llm_model, temperature=0.2)

        elif provider == "anthropic":
            if ChatAnthropic is None:
                raise LLMError("langchain-anthropic is not installed. Run: pip install langchain-anthropic")
            api_key = settings.openai_api_key  # anthropic uses its own key, stored in same env var pattern
            import os

            api_key = os.environ.get("ANTHROPIC_API_KEY", settings.openai_api_key)
            if not api_key:
                raise LLMError("Anthropic API key is required. Set ANTHROPIC_API_KEY in your environment.")
            return ChatAnthropic(model=settings.llm_model, temperature=0.2)

        else:
            raise LLMError(f"Unknown LLM provider: {provider}")
