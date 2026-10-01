"""RAG evaluation script."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chat.memory import ConversationMemory
from src.chat.service import ChatService
from src.config.settings import Settings, reset_settings_cache
from src.evaluation.metrics import EvaluationRunner
from src.llm.factory import LLMFactory
from src.vectorstore.qdrant_store import QdrantStore

SAMPLE_DATASET = [
    {
        "question": "What is the vacation allowance?",
        "expected_answer": "20 days",
        "expected_sources": ["company_policy.pdf:2"],
    },
    {
        "question": "What is the refund period?",
        "expected_answer": "30 days",
        "expected_sources": ["product_manual.pdf:4"],
    },
    {
        "question": "What is the employee dress code?",
        "expected_answer": "",
        "expected_sources": [],
    },
]


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG system")
    parser.add_argument("--dataset", type=str, help="Path to JSON evaluation dataset")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output path")
    parser.add_argument("--k", type=int, default=5, help="Number of documents to retrieve")
    args = parser.parse_args()

    # Initialize components
    settings = Settings()
    reset_settings_cache()
    llm = LLMFactory.create(settings)
    store = QdrantStore(settings)
    memory = ConversationMemory(settings.conversation_db_path)
    chat = ChatService(settings, store, llm, memory)

    # Load dataset
    if args.dataset:
        with open(args.dataset) as f:
            dataset = json.load(f)
    else:
        dataset = SAMPLE_DATASET
        print("Using sample dataset (no real documents indexed).")

    # Run evaluation
    runner = EvaluationRunner(chat)
    results = runner.evaluate(dataset, k=args.k)
    summary = runner.summary(results)

    # Export
    runner.export_results(results, args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("=" * 60)
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
