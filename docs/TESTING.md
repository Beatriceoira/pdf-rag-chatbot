# Testing Guide

## Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit -v

# Integration tests (require Qdrant)
pytest tests/integration -v

# RAG-specific tests
pytest tests/rag -v

# Security tests
pytest tests/security -v

# With coverage
pytest --cov=src --cov-report=term-missing -v
```

## Test Categories

| Directory | Description | Requires Services |
|-----------|-------------|-------------------|
| `tests/unit/` | Pure logic tests (chunking, hashing, config, validation, prompts) | No |
| `tests/integration/` | Qdrant store, ingestion pipeline, retrieval | Qdrant (in-memory) |
| `tests/rag/` | Grounding, citations, prompt injection, reranking | No |
| `tests/security/` | File validation, secret handling, resource limits | No |

## Adding Tests

1. Place unit tests in `tests/unit/test_<module>.py`
2. Place integration tests in `tests/integration/test_<module>.py`
3. Use `Settings(qdrant_mode="memory")` for isolated Qdrant tests
4. Mark integration tests with `@pytest.mark.integration`
