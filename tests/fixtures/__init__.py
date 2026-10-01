"""Configuration fixtures for testing."""

from __future__ import annotations

from pathlib import Path

# Test fixtures directory
FIXTURES_DIR = Path(__file__).parent.parent.parent / "tests" / "fixtures"


def get_fixture_path(filename: str) -> Path:
    """Get absolute path to a test fixture file."""
    path = FIXTURES_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Fixture not found: {path}")
    return path


# Evaluation dataset fixture
EVALUATION_DATASET = [
    {
        "question": "What is the vacation allowance?",
        "expected_answer": "20 days",
        "expected_sources": ["company_policy.pdf:2"],
    },
    {
        "question": "What is the refund period?",
        "expected_answer": "30 days",
        "expected_sources": ["policy.pdf:4"],
    },
    {
        "question": "What is the employee dress code?",
        "expected_answer": "",
        "expected_sources": [],
    },
]
