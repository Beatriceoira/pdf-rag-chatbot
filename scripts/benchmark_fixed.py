#!/usr/bin/env python3
"""Fixed RAG performance benchmarking script with proper collection handling."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import psutil
from src.chat.memory import ConversationMemory
from src.chat.service import ChatService
from src.config.settings import Settings
from src.embeddings.factory import EmbeddingFactory
from src.ingestion.pipeline import IngestionPipeline
from src.llm.factory import LLMFactory
from src.vectorstore.qdrant_store import QdrantStore


def get_system_info() -> Dict[str, Any]:
    """Get system information for benchmark reproducibility."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": sys.platform,
        "cpu_count": psutil.cpu_count(),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "process_id": os.getpid(),
    }


def benchmark_ingestion(
    pipeline: IngestionPipeline,
    pdf_path: Path,
    store: QdrantStore,
    embedding_model,
    iterations: int = 3,
) -> Dict[str, Any]:
    """Benchmark PDF ingestion pipeline."""
    times = []

    for i in range(iterations):
        # Clear the collection before each iteration to ensure we are measuring
        # the ingestion of a single document into an empty store.
        store.clear()

        start_time = time.perf_counter()

        # Step 1: Ingestion pipeline (validation, parsing, chunking)
        result = pipeline.ingest_from_path(pdf_path)
        if not result.success:
            print(f"Iteration {i+1} ingestion failed: {result.error}")
            continue

        # Step 2: Embedding
        texts = [doc.page_content for doc in result.chunks]
        embeddings = embedding_model.embed_documents(texts)

        # Step 3: Insertion
        from langchain_core.documents import Document as LCDocument
        embedded_docs = []
        for j, doc in enumerate(result.chunks):
            embedded_doc = LCDocument(
                page_content=doc.page_content,
                metadata={**doc.metadata, "embedding": embeddings[j]},
            )
            embedded_docs.append(embedded_doc)
        store.add_documents(embedded_docs)

        end_time = time.perf_counter()
        times.append(end_time - start_time)

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
        "total_time_seconds": stats(times),
        "iterations": iterations,
        "successful_runs": len(times),
    }


def benchmark_retrieval(
    chat_service: ChatService,
    queries: List[str],
    iterations: int = 10,
) -> Dict[str, Any]:
    """Benchmark query processing latency using the public API."""
    retrieval_times = []
    generation_times = []
    end_to_end_times = []

    # Warm-up
    if queries:
        chat_service.ask(queries[0], "benchmark_warmup")

    for query in queries:
        for _ in range(iterations):
            start_time = time.perf_counter()

            # Retrieve and generate
            response = chat_service.ask(query, f"benchmark_{int(time.time())}")

            end_time = time.perf_counter()

            # The ChatService.ask method does not separate retrieval and
            # generation latency in the returned ChatResponse? Actually,
            # it does: the ChatResponse has retrieval_latency_ms and
            # generation_latency_ms.
            # Let's use those.
            retrieval_times.append(response.retrieval_latency_ms)
            generation_times.append(response.generation_latency_ms)
            end_to_end_times.append(response.retrieval_latency_ms + response.generation_latency_ms)

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
    parser = argparse.ArgumentParser(description="Benchmark RAG performance with fixed collection handling")
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
        default="reports/benchmarks/benchmark_results_fixed.json",
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

    # Generate a unique collection name for this benchmark run to avoid interference
    benchmark_collection_name = f"benchmark_{int(time.time())}"
    original_collection_name = settings.collection_name
    settings.collection_name = benchmark_collection_name

    print(f"Initializing benchmark with Qdrant mode: {args.qdrant_mode}")
    print(f"Using collection: {benchmark_collection_name}")
    print(f"System info: {get_system_info()}")

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

    try:
        # Initialize components that we will reuse
        llm = LLMFactory.create(settings)
        store = QdrantStore(settings)
        memory = ConversationMemory(settings.conversation_db_path)
        pipeline = IngestionPipeline(settings)
        embedding_model = EmbeddingFactory.create(settings)
        print("Components initialized successfully")

        # Run ingestion benchmark if document provided
        if args.document and Path(args.document).exists():
            pdf_path = Path(args.document)
            print(f"Running ingestion benchmark on {pdf_path}...")
            ingestion_results = benchmark_ingestion(
                pipeline, pdf_path, store, embedding_model, args.ingestion_iterations
            )
            results["benchmarks"]["ingestion"] = ingestion_results
            print(f"Ingestion benchmark completed: {ingestion_results['total_time_seconds']['mean']:.2f}s avg")
        elif args.document:
            print(f"Document not found: {args.document}")

        # Initialize chat service for retrieval benchmark (using the same store and llm)
        chat_service = ChatService(settings, store, llm, memory)
        print("Chat service initialized for retrieval benchmark")

        # Run retrieval benchmark
        print(f"Running retrieval benchmark with {len(args.queries)} queries...")
        retrieval_results = benchmark_retrieval(chat_service, args.queries, args.retrieval_iterations)
        results["benchmarks"]["retrieval"] = retrieval_results
        print(f"Retrieval benchmark completed: {retrieval_results['end_to_end_latency_ms']['mean']:.2f}ms avg")

    finally:
        # Clean up: delete the benchmark collection and restore the original collection name
        try:
            store.delete_collection()
        except Exception as e:
            print(f"Warning: failed to delete benchmark collection: {e}")
        settings.collection_name = original_collection_name

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())