#!/usr/bin/env python3
"""Simple RAG performance benchmarking script using public APIs."""

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
from src.config.settings import Settings, reset_settings_cache
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
    iterations: int = 3,
    settings: Settings = None,
) -> Dict[str, Any]:
    """Benchmark PDF ingestion pipeline using the public API."""
    if settings is None:
        # This should not happen, but we'll create a default settings object
        settings = Settings()
    times = []

    for i in range(iterations):
        # Create a fresh pipeline instance for each iteration to avoid
        # interference from previous runs (e.g., vector store state).
        # However, note that the pipeline uses the same settings, so if
        # we are using a persistent Qdrant mode (local or server), we
        # will be adding to the same collection. To avoid that, we can
        # use a unique collection name for each iteration by modifying
        # the settings, but that's complex.
        # Alternatively, we can clear the collection before each iteration.
        # Let's do that by getting the vector store from the settings?
        # We'll create a temporary Qdrant store to clear the collection.
        # But note: the pipeline creates its own vector store internally.
        # Actually, looking at the IngestionPipeline, it does not create
        # a vector store; it returns chunks and metadata. The caller
        # (in our case, the benchmark) is responsible for inserting
        # into the vector store. Wait, let's check the IngestionPipeline
        # in the actual code.

        # In the actual IngestionPipeline (src/ingestion/pipeline.py),
        # the `ingest_from_path` method returns an IngestionResult with
        # chunks and metadata. It does not insert into the vector store.
        # The insertion is done by the caller (in app.py, _process_uploaded_files).
        # So our benchmark should reflect that: we measure the time to
        # produce the chunks and metadata, and then we can optionally
        # measure the time to insert into the vector store.

        # However, for an end-to-end benchmark of the ingestion pipeline
        # as used in the application, we should include the insertion
        # step. But to keep things simple and focused on the pipeline
        # itself, we'll measure only the pipeline's internal steps
        # (validation, parsing, chunking, embedding) and note that
        # insertion is a separate step.

        # Let's change approach: we'll benchmark the entire process
        # as it happens in the application: we'll call the pipeline
        # to get the result, and then we'll insert the chunks into
        # the vector store (using a vector store created from the same
        # settings). We'll do this in a loop and measure the total time.

        # We'll create a new pipeline and a new vector store for each
        # iteration to avoid state issues.

        # Create fresh settings for each iteration to avoid any
        # caching issues? The Settings class is cached via lru_cache.
        # We'll reset the cache before each iteration? That might be
        # overkill. Instead, we'll use the same settings and rely on
        # the fact that the pipeline and vector store are stateless
        # except for the vector store's collection.

        # To avoid interference, we'll use a unique collection name
        # for each iteration by appending the iteration number to the
        # collection name in the settings. But we cannot change the
        # settings because they are cached and used by other parts.
        # Instead, we'll create a custom settings object for each
        # iteration? That breaks the singleton pattern.

        # Given the complexity, and since the benchmark is meant to
        # give a rough idea of performance, we'll run the ingestion
        # benchmark in a way that is similar to how the application
        # works: we'll process a document and insert it into the
        # vector store, and we'll do this multiple times, but we'll
        # use different document names (or we'll accept that we are
        # re-ingesting the same document and getting duplicate
        # detection).

        # For simplicity, we'll just measure the pipeline's internal
        # steps (ingest_from_path) and note that the insertion step
        # is separate and can be benchmarked similarly.

        # Let's look at the IngestionPipeline.ingest_from_path method:
        # It calls validate_file, parse_pdf, chunk_documents.
        # It does not do embedding or insertion. The embedding and
        # insertion are done by the EmbeddingFactory and the vector
        # store in the caller.

        # Therefore, to benchmark the full ingestion as used in the
        # application, we need to:
        # 1. Validate
        # 2. Parse
        # 3. Chunk
        # 4. Embed
        # 5. Insert

        # We'll do all of these steps in the benchmark and measure
        # the total time.

        # We'll create a function that does all these steps using
        # the public APIs from the ingestion module and the
        # embedding and vector store factories.

        # However, to save time, let's benchmark the pipeline's
        # `ingest_from_path` method (which does validation, parsing,
        # chunking) and then separately benchmark the embedding and
        # insertion steps.

        # Given the time constraints, we'll benchmark the end-to-end
        # process using the same approach as in the application's
        # `_process_uploaded_files` function, but we'll do it in a
        # loop and measure the time.

        # We'll create a fresh vector store for each iteration by
        # using a unique collection name. We can do this by
        # temporarily modifying the settings' collection name.
        # Since the settings are a singleton, we'll store the
        # original value, change it, and then restore it.

        original_collection_name = settings.collection_name
        try:
            settings.collection_name = f"{settings.collection_name}_bench_{i}"
            # Now create a fresh pipeline and vector store
            pipe = IngestionPipeline(settings)
            store = QdrantStore(settings)
            # Note: the pipeline does not use the store; we'll
            # manually do the steps that the application does.

            start_time = time.perf_counter()

            # Step 1: Ingestion pipeline (validation, parsing, chunking)
            result = pipe.ingest_from_path(pdf_path)
            if not result.success:
                print(f"Iteration {i+1} ingestion failed: {result.error}")
                continue

            # Step 2: Embedding
            embedding_model = EmbeddingFactory.create(settings)
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
        finally:
            settings.collection_name = original_collection_name

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
    conversation_id: str = None,
) -> Dict[str, Any]:
    """Benchmark query processing latency using the public API."""
    retrieval_times = []
    generation_times = []
    end_to_end_times = []

    # Warm-up
    if queries and conversation_id:
        chat_service.ask(queries[0], conversation_id)

    for query in queries:
        for _ in range(iterations):
            start_time = time.perf_counter()

            # Retrieve and generate
            response = chat_service.ask(query, conversation_id)

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
    parser = argparse.ArgumentParser(description="Benchmark RAG performance using public APIs")
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
        ingestion_results = benchmark_ingestion(IngestionPipeline(settings), pdf_path, args.ingestion_iterations, settings)
        results["benchmarks"]["ingestion"] = ingestion_results
        print(f"Ingestion benchmark completed: {ingestion_results['total_time_seconds']['mean']:.2f}s avg")
    elif args.document:
        print(f"Document not found: {args.document}")

    # Initialize components for retrieval benchmark
    try:
        llm = LLMFactory.create(settings)
        store = QdrantStore(settings)
        memory = ConversationMemory(settings.conversation_db_path)
        chat_service = ChatService(settings, store, llm, memory)
        # Create a conversation for benchmarking
        conv_id = memory.create_conversation("benchmark")
        print(f"Components initialized successfully for retrieval benchmark (conversation ID: {conv_id})")
    except Exception as e:
        print(f"Failed to initialize components for retrieval benchmark: {e}")
        return 1

    # Run retrieval benchmark
    print(f"Running retrieval benchmark with {len(args.queries)} queries...")
    retrieval_results = benchmark_retrieval(chat_service, args.queries, args.retrieval_iterations, conv_id)
    results["benchmarks"]["retrieval"] = retrieval_results
    print(f"Retrieval benchmark completed: {retrieval_results['end_to_end_latency_ms']['mean']:.2f}ms avg")

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())