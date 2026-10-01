"""OpenAI chat model wrapper."""

from __future__ import annotations

from typing import Any

from src.utils.logging import get_logger

logger = get_logger("llm")

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover
    ChatOpenAI = None  # type: ignore[assignment,misc]


class OpenAIChat:
    """Thin wrapper around LangChain's ChatOpenAI."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.2):
        if ChatOpenAI is None:
            raise ImportError("langchain-openai is not installed. Run: pip install langchain-openai")
        self._model = ChatOpenAI(model=model, temperature=temperature)
        self._model_name = model

    @property
    def model_name(self) -> str:
        return self._model_name

    def invoke(self, messages) -> Any:
        """Invoke the model with messages."""
        logger.debug("Invoking OpenAI model '%s'", self._model_name)
        return self._model.invoke(messages)

    def stream(self, messages) -> Any:
        """Stream the model response."""
        logger.debug("Streaming OpenAI model '%s'", self._model_name)
        return self._model.stream(messages)

    def __repr__(self) -> str:
        return f"OpenAIChat(model={self._model_name!r})"
