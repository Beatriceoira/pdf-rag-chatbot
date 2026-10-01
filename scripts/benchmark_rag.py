#!/usr/bin/env python3
"""RAG performance benchmarking script."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import psutil
from src.chat.memory import ConversationMemory
from src.chat.service import ChatService
from src.config.settings import Settings, reset_settings_cache
from src.embeddings.factory import EmbeddingFactory
from src.ingestion.pipeline import IngestionPipeline
from src.llm.factory import LLMFactory
from src.vectorstore.qdrant_store import QdrantStore


def get_system_info() -> Dict[str, Any]:
    """Get system information for benchmark reproducibility."""
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "python_version": sys.version,
        "platform": sys.platform,
        "cpu_count": psutil.cpu_count(),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "process_id": os.getpid(),
    }


def benchmark_ingestion(
    pipeline: IngestionPipeline,
    store: QdrantStore,
    pdf_path: Path,
    iterations: int = 3,
) -> Dict[str, Any]:
    """Benchmark PDF ingestion pipeline."""
    times = {
        "validation": [],
        "parsing": [],
        "chunking": [],
        "embedding": [],
        "vector_insertion": [],
        "total": [],
    }

    # Warm-up run
    if pdf_path.exists():
        pipeline.ingest_from_path(pdf_path)

    for i in range(iterations):
        start_time = time.perf_counter()

        # Validation
        validation_start = time.perf_counter()
        # We'll measure this indirectly through the pipeline
        validation_end = time.perf_counter()
        times["validation"].append(validation_end - validation_start)

        # Full pipeline
        result = pipeline.ingest_from_path(pdf_path)

        end_time = time.perf_counter()
        total_time = end_time - start_time

        if result.success:
            times["total"].append(total_time)
            # For now, we'll attribute most time to the full pipeline
            # In a more detailed version, we'd instrument each step
        else:
            print(f"Iteration {i+1} failed: {result.error}")

    # Calculate statistics
    def stats(times_list: List[float]) -> Dict[str, float]:
        if not times_list:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}
        sorted_times = sorted(times_list)
        return {
            "min": min(times_list),
            "max": max(times_list),
            "mean": sum(times_list) / len(times_list),
            "median": sorted_times[len(sorted_times) // 2],
        }

    return {
        "validation": stats(times["validation"]),
        "parsing": stats(times["parsing"]),  # Placeholder
        "chunking": stats(times["chunking"]),  # Placeholder
        "embedding": stats(times["embedding"]),  # Placeholder
        "vector_insertion": stats(times["vector_insertion"]),  # Placeholder
        "total": stats(times["total"]),
        "iterations": iterations,
        "successful_runs": len([t for t in times["total"] if t > 0]),
    }


def benchmark_retrieval(
    chat_service: ChatService,
    queries: List[str],
    iterations: int = 10,
) -> Dict[str, Any]:
    """Benchmark query processing latency."""
    retrieval_times = []
    generation_times = []
    end_to_end_times = []

    # Warm-up
    if queries:
        chat_service.ask(queries[0], "benchmark_warmup")

    for query in queries:
        for _ in range(iterations):
            start_time = time.perf_counter()

            # Retrieve
            retrieval_start = time.perf_counter()
            # We'll measure this through the chat service
            retrieval_end = time.perf_counter()

            # Generate
            generation_start = time.perf_counter()
            response = chat_service.ask(query, f"benchmark_{int(time.time())}")
            generation_end = time.perf_counter()

            end_time = time.perf_counter()

            retrieval_times.append((retrieval_end - retrieval_start) * 1000)  # ms
            generation_times.append((generation_end - generation_start) * 1000)  # ms
            end_to_end_times.append((end_time - start_time) * 1000)  # ms

    def stats(times_list: List[float]) -> Dict[str, float]:
        if not times_list:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
        sorted_times = sorted(times_list)
        n = len(times_list)
        return {
            "mean": sum(times_list) / n,
            "p50": sorted_times[int(n * 0.5)],
            "p95": sorted_times[int(n * 0.95)] if n >= 20 else sorted_times[-1],
            "p99": sorted_times[int(n * 0.99)] if n >= 100 else sorted_times[-1],
        }

    return {
        "retrieval_latency_ms": stats(retrieval_times),
        "generation_latency_ms": stats(generation_times),
        "end_to_end_latency_ms": stats(end_to_end_times),
        "iterations_per_query": iterations,
        "total_queries": len(queries),
        "total_measurements": len(retrieval_times),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark RAG performance")
    parser.add_argument(
        "--document",
        type=str,
        help="Path to PDF document for ingestion benchmark",
    )
    parser.add_argument(
        "--queries",
        type=str,
        nargs="+",
        default=[
            "What is the main topic of this document?",
            "Summarize the key findings.",
            "What are the conclusions?",
        ],
        help="Queries for retrieval benchmark",
    )
    parser.add_argument(
        "--ingestion-iterations",
        type=int,
        default=3,
        help="Number of iterations for ingestion benchmark",
    )
    parser.add_argument(
        "--retrieval-iterations",
        type=int,
        default=10,
        help="Number of iterations for retrieval benchmark",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/benchmarks/benchmark_results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--qdrant-mode",
        type=str,
        choices=["local", "server", "memory"],
        default="local",
        help="Qdrant mode to use",
    )
    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize settings
    settings = Settings()
    settings.qdrant_mode = args.qdrant_mode
    reset_settings_cache()

    print(f"Initializing benchmark with Qdrant mode: {args.qdrant_mode}")
    print(f"System info: {get_system_info()}")

    # Initialize components
    try:
        llm = LLMFactory.create(settings)
        store = QdrantStore(settings)
        memory = ConversationMemory(settings.conversation_db_path)
        chat_service = ChatService(settings, store, llm, memory)
        pipeline = IngestionPipeline(settings)
        print("Components initialized successfully")
    except Exception as e:
        print(f"Failed to initialize components: {e}")
        return 1

    results = {
        "system_info": get_system_info(),
        "configuration": {
            "qdrant_mode": args.qdrant_mode,
            "embedding_provider": settings.embedding_provider,
            "llm_provider": settings.llm_provider,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "retrieval_k": settings.retrieval_k,
        },
        "benchmarks": {},
    }

    # Run ingestion benchmark if document provided
    if args.document and Path(args.document).exists():
        pdf_path = Path(args.document)
        print(f"Running ingestion benchmark on {pdf_path}...")
        ingestion_results = benchmark_ingestion(
            pipeline, store, pdf_path, args.ingestion_iterations
        )
        results["benchmarks"]["ingestion"] = ingestion_results
        print(f"Ingestion benchmark completed: {ingestion_results['total']['mean']:.2f}s avg")
    elif args.document:
        print(f"Document not found: {args.document}")

    # Run retrieval benchmark
    print(f"Running retrieval benchmark with {len(args.queries)} queries...")
    retrieval_results = benchmark_retrieval(
        chat_service, args.queries, args.retrieval_iterations
    )
    results["benchmarks"]["retrieval"] = retrieval_results
    print(f"Retrieval benchmark completed: {retrieval_results['end_to_end_latency_ms']['mean']:.2f}ms avg")

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())