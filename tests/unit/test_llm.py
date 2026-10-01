"""Unit tests for LLM factory and OpenAI chat wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.config.settings import Settings
from src.llm.factory import LLMFactory
from src.llm.openai_chat import OpenAIChat
from src.utils.exceptions import LLMError


class TestLLMFactory:
    """Tests for LLMFactory."""

    def test_create_openai(self):
        """Test creating OpenAI LLM."""
        settings = Settings(llm_provider="openai", llm_model="gpt-4o-mini", _env_file=None)

        with patch("src.llm.factory.OpenAIChat") as mock_chat:
            result = LLMFactory.create(settings)
            mock_chat.assert_called_once_with(model="gpt-4o-mini")
            assert result == mock_chat.return_value

    def test_create_openai_without_key_raises(self):
        """Test that OpenAI LLM raises when API key is missing."""
        with patch.dict("os.environ", {}, clear=True):
            settings = Settings(llm_provider="openai", openai_api_key="", _env_file=None)

            with patch("src.llm.factory.OpenAIChat") as mock_chat:
                with pytest.raises(LLMError, match="OpenAI API key is required"):
                    LLMFactory.create(settings)
                mock_chat.assert_not_called()

    def test_create_ollama(self):
        """Test creating Ollama LLM."""
        settings = Settings(llm_provider="ollama", llm_model="llama3", _env_file=None)

        mock_ollama = MagicMock()
        with patch("src.llm.factory.ChatOllama", mock_ollama):
            result = LLMFactory.create(settings)
            mock_ollama.assert_called_once_with(model="llama3", temperature=0.2)
            assert result == mock_ollama.return_value

    def test_create_ollama_without_library_raises(self):
        """Test Ollama raises when library not installed."""
        settings = Settings(llm_provider="ollama", _env_file=None)

        with pytest.raises(LLMError, match="langchain-ollama"), patch("src.llm.factory.ChatOllama", None):
            LLMFactory.create(settings)

    def test_create_anthropic(self):
        """Test creating Anthropic LLM."""
        settings = Settings(llm_provider="anthropic", llm_model="claude-3-sonnet", _env_file=None)

        mock_anthropic = MagicMock()
        with patch("src.llm.factory.ChatAnthropic", mock_anthropic), patch.dict(
            "os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}
        ):
            result = LLMFactory.create(settings)
            mock_anthropic.assert_called_once_with(model="claude-3-sonnet", temperature=0.2)
            assert result == mock_anthropic.return_value

    def test_create_anthropic_without_key_raises(self):
        """Test Anthropic raises when API key is missing."""
        with patch.dict("os.environ", {}, clear=True):
            settings = Settings(llm_provider="anthropic", openai_api_key="", _env_file=None)
            with patch("src.llm.factory.ChatAnthropic", MagicMock()), pytest.raises(
                LLMError, match=r"Anthropic API key is required|langchain-anthropic"
            ):
                LLMFactory.create(settings)

    def test_create_anthropic_without_library_raises(self):
        """Test Anthropic raises when langchain-anthropic is not installed."""
        settings = Settings(llm_provider="anthropic", _env_file=None)

        with pytest.raises(LLMError, match="langchain-anthropic"), patch(
            "src.llm.factory.ChatAnthropic", None
        ), patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}):
            LLMFactory.create(settings)

    def test_create_unknown_provider_raises(self):
        """Test that an unknown provider raises LLMError."""
        settings = Settings(llm_provider="openai", _env_file=None)
        settings.llm_provider = "unknown"  # bypass pydantic validation

        with pytest.raises(LLMError, match="Unknown LLM provider"):
            LLMFactory.create(settings)

    def test_create_ollama_with_custom_model(self):
        """Test Ollama with custom model name."""
        settings = Settings(llm_provider="ollama", llm_model="custom-model", _env_file=None)

        mock_ollama = MagicMock()
        with patch("src.llm.factory.ChatOllama", mock_ollama):
            LLMFactory.create(settings)
            mock_ollama.assert_called_once_with(model="custom-model", temperature=0.2)


class TestOpenAIChat:
    """Tests for OpenAIChat wrapper."""

    def test_init(self):
        """Test initialization."""
        with patch("src.llm.openai_chat.ChatOpenAI") as mock_chat:
            mock_instance = MagicMock()
            mock_chat.return_value = mock_instance
            chat = OpenAIChat(model="gpt-4o-mini", temperature=0.5)
            assert chat.model_name == "gpt-4o-mini"
            mock_chat.assert_called_once_with(model="gpt-4o-mini", temperature=0.5)

    def test_model_name_property(self):
        """Test model_name property."""
        with patch("src.llm.openai_chat.ChatOpenAI") as mock_chat:
            mock_chat.return_value = MagicMock()
            chat = OpenAIChat(model="gpt-4o")
            assert chat.model_name == "gpt-4o"

    def test_invoke(self):
        """Test invoke method."""
        with patch("src.llm.openai_chat.ChatOpenAI") as mock_chat:
            mock_instance = MagicMock()
            mock_chat.return_value = mock_instance

            mock_response = MagicMock()
            mock_response.content = "Hi there!"
            mock_instance.invoke.return_value = mock_response

            chat = OpenAIChat()
            messages = [{"role": "user", "content": "Hello"}]

            result = chat.invoke(messages)

            assert result == mock_response
            mock_instance.invoke.assert_called_once_with(messages)

    def test_stream(self):
        """Test stream method."""
        with patch("src.llm.openai_chat.ChatOpenAI") as mock_chat:
            mock_instance = MagicMock()
            mock_chat.return_value = mock_instance

            mock_stream = MagicMock()
            mock_stream.__iter__ = MagicMock(return_value=iter([MagicMock(content="Hi")]))
            mock_instance.stream.return_value = mock_stream

            chat = OpenAIChat()
            messages = [{"role": "user", "content": "Hello"}]

            result = chat.stream(messages)

            assert result == mock_stream
            mock_instance.stream.assert_called_once_with(messages)

    def test_repr(self):
        """Test string representation."""
        with patch("src.llm.openai_chat.ChatOpenAI") as mock_chat:
            mock_chat.return_value = MagicMock()
            chat = OpenAIChat(model="gpt-4o-mini")
            assert "OpenAIChat" in repr(chat)
            assert "gpt-4o-mini" in repr(chat)

    def test_init_without_langchain_raises(self):
        """Test that init raises when langchain-openai is not installed."""
        with pytest.raises(ImportError, match="langchain-openai"), patch(
            "src.llm.openai_chat.ChatOpenAI", None
        ):
            OpenAIChat()

    def test_invoke_returns_content(self):
        """Test invoke returns response content."""
        with patch("src.llm.openai_chat.ChatOpenAI") as mock_chat:
            mock_instance = MagicMock()
            mock_chat.return_value = mock_instance
            mock_response = MagicMock()
            mock_response.content = "Test answer"
            mock_instance.invoke.return_value = mock_response

            chat = OpenAIChat()
            result = chat.invoke([{"role": "user", "content": "Hi"}])

            assert result.content == "Test answer"

    def test_stream_yields_chunks(self):
        """Test stream yields chunks."""
        with patch("src.llm.openai_chat.ChatOpenAI") as mock_chat:
            mock_instance = MagicMock()
            mock_chat.return_value = mock_instance
            mock_instance.stream.return_value = [
                MagicMock(content="Hello"),
                MagicMock(content=" world"),
            ]

            chat = OpenAIChat()
            chunks = list(chat.stream([{"role": "user", "content": "Hi"}]))

            assert len(chunks) == 2
            assert chunks[0].content == "Hello"
            assert chunks[1].content == " world"
