#!/usr/bin/env python3
"""Detailed RAG performance benchmarking script."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import psutil
from src.chat.memory import ConversationMemory
from src.chat.service import ChatService
from src.config.settings import Settings, reset_settings_cache
from src.embeddings.factory import EmbeddingFactory
from src.ingestion.chunker import chunk_documents
from src.ingestion.metadata import DocumentMetadata
from src.ingestion.parser import parse_pdf_from_bytes
from src.ingestion.pdf_loader import validate_in_memory
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


def benchmark_ingestion_detailed(
    pipeline_settings: Settings,
    pdf_path: Path,
    iterations: int = 3,
) -> Dict[str, Any]:
    """Benchmark PDF ingestion pipeline with detailed step timing."""
    # We'll use a unique collection name for this benchmark run
    benchmark_collection_name = f"benchmark_ingestion_{int(time.time())}"
    original_collection_name = pipeline_settings.collection_name
    pipeline_settings.collection_name = benchmark_collection_name

    times = {
        "validation": [],
        "parsing": [],
        "chunking": [],
        "embedding": [],
        "vector_insertion": [],
        "total": [],
    }

    # We need to create the embedding model and vector store for this benchmark
    embedding_model = EmbeddingFactory.create(pipeline_settings)
    vector_store = QdrantStore(pipeline_settings)

    # Warm-up run
    if pdf_path.exists():
        with open(pdf_path, "rb") as f:
            file_bytes = f.read()
        # We'll do a warm-up ingestion to initialize the collection
        _ = ingest_document_pipeline(
            file_bytes,
            pdf_path.name,
            pipeline_settings,
            embedding_model,
            vector_store,
        )
        # Clear the collection for the actual benchmark runs
        vector_store.delete_collection()

    for i in range(iterations):
        start_time = time.perf_counter()

        with open(pdf_path, "rb") as f:
            file_bytes = f.read()

        # We'll break down the ingestion into steps
        validation_start = time.perf_counter()
        errors = validate_in_memory(
            file_bytes,
            pdf_path.name,
            max_size_mb=pipeline_settings.max_file_size_mb,
        )
        validation_end = time.perf_counter()
        if errors:
            print(f"Iteration {i+1} validation failed: {errors}")
            # Skip this iteration if validation fails
            continue
        validation_end = time.perf_counter()
        times["validation"].append(validation_end - validation_start)

        parsing_start = time.perf_counter()
        try:
            docs, meta = parse_pdf_from_bytes(file_bytes, pdf_path.name)
        except Exception as exc:
            print(f"Iteration {i+1} parsing failed: {exc}")
            continue
        parsing_end = time.perf_counter()
        times["parsing"].append(parsing_end - parsing_start)

        chunking_start = time.perf_counter()
        try:
            chunks, chunk_meta = chunk_documents(
                docs,
                chunk_size=pipeline_settings.chunk_size,
                chunk_overlap=pipeline_settings.chunk_overlap,
                separators=pipeline_settings.chunk_separators,
                document_name=pdf_path.name,
            )
        except Exception as exc:
            print(f"Iteration {i+1} chunking failed: {exc}")
            continue
        chunking_end = time.perf_counter()
        times["chunking"].append(chunking_end - chunking_start)

        # Enrich chunks with metadata (as in the pipeline)
        for doc in chunks:
            doc.metadata.setdefault("document_name", pdf_path.name)
            doc.metadata.setdefault("file_hash", meta.file_hash)

        embedding_start = time.perf_counter()
        # Embed the chunks
        texts = [doc.page_content for doc in chunks]
        try:
            embeddings = embedding_model.embed_documents(texts)
        except Exception as exc:
            print(f"Iteration {i+1} embedding failed: {exc}")
            continue
        embedding_end = time.perf_counter()
        times["embedding"].append(embedding_end - embedding_start)

        vector_insertion_start = time.perf_counter()
        try:
            # We need to create LCDocuments with embeddings
            from langchain_core.documents import Document as LCDocument

            embedded_docs = []
            for i, doc in enumerate(chunks):
                embedded_doc = LCDocument(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "embedding": embeddings[i]},
                )
                embedded_docs.append(embedded_doc)
            vector_store.add_documents(embedded_docs)
        except Exception as exc:
            print(f"Iteration {i+1} vector insertion failed: {exc}")
            continue
        vector_insertion_end = time.perf_counter()
        times["vector_insertion"].append(vector_insertion_end - vector_insertion_start)

        end_time = time.perf_counter()
        total_time = end_time - start_time
        times["total"].append(total_time)

    # Restore the original collection name
    pipeline_settings.collection_name = original_collection_name
    # Clean up the benchmark collection
    try:
        vector_store.delete_collection()
    except Exception:
        pass  # Ignore errors during cleanup

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
        "parsing": stats(times["parsing"]),
        "chunking": stats(times["chunking"]),
        "embedding": stats(times["embedding"]),
        "vector_insertion": stats(times["vector_insertion"]),
        "total": stats(times["total"]),
        "iterations": iterations,
        "successful_runs": len([t for t in times["total"] if t > 0]),
    }


def benchmark_retrieval_detailed(
    settings: Settings,
    queries: List[str],
    iterations: int = 10,
) -> Dict[str, Any]:
    """Benchmark query processing latency (embedding + search)."""
    # We'll use a unique collection name for this benchmark run
    benchmark_collection_name = f"benchmark_retrieval_{int(time.time())}"
    original_collection_name = settings.collection_name
    settings.collection_name = benchmark_collection_name

    # We need to initialize the embedding model and vector store
    embedding_model = EmbeddingFactory.create(settings)
    vector_store = QdrantStore(settings)

    # We need to have some data in the vector store to search
    # We'll ingest a small document for the benchmark
    benchmark_doc_path = Path(__file__).parent.parent / "test_document.pdf"
    if benchmark_doc_path.exists():
        with open(benchmark_doc_path, "rb") as f:
            file_bytes = f.read()
        # Ingest the benchmark document
        _ = ingest_document_pipeline(
            file_bytes,
            benchmark_doc_path.name,
            settings,
            embedding_model,
            vector_store,
        )
    else:
        print("Warning: benchmark document not found, retrieval benchmark will be on empty store")

    retrieval_times = []
    embedding_times = []
    search_times = []

    # Warm-up
    if queries:
        _ = embed_and_search(queries[0], embedding_model, vector_store)

    for query in queries:
        for _ in range(iterations):
            start_time = time.perf_counter()

            # Embed the query
            embedding_start = time.perf_counter()
            try:
                query_embedding = embedding_model.embed_query(query)
            except Exception as exc:
                print(f"Query embedding failed: {exc}")
                continue
            embedding_end = time.perf_counter()

            # Search the vector store
            search_start = time.perf_counter()
            try:
                results = vector_store.search(
                    query=query,  # We pass the query string, the search method will embed it again
                    k=settings.retrieval_k,
                )
            except Exception as exc:
                print(f"Vector search failed: {exc}")
                continue
            search_end = time.perf_counter()

            end_time = time.perf_counter()

            # Note: the search method embeds the query again, so we are measuring embedding twice?
            # Actually, in the search method, it does: query_embedding = self.embedding.embed_query(query)
            # So we are measuring the embedding twice in this benchmark.
            # To avoid that, we could change the search method to accept a pre-computed embedding?
            # But we don't want to change the production code.
            # Alternatively, we can measure the embedding time and then the search time without the embedding?
            # We'll adjust: we'll measure the time for the search method (which includes embedding) and then
            # we'll also measure the embedding time separately for reporting.
            # We'll report:
            #   query_embedding_time: time to embed the query (using the embedding model directly)
            #   vector_search_time: time to search the vector store (which includes embedding again)
            #   total_retrieval_time: query_embedding_time + vector_search_time
            # But note: the vector_search_time includes an embedding step, so we are double counting the embedding.
            # We'll note this in the limitations.

            embedding_times.append((embedding_end - embedding_start) * 1000)  # ms
            search_times.append((search_end - search_start) * 1000)  # ms
            retrieval_times.append((end_time - start_time) * 1000)  # ms

    # Restore the original collection name
    settings.collection_name = original_collection_name
    # Clean up the benchmark collection
    try:
        vector_store.delete_collection()
    except Exception:
        pass  # Ignore errors during cleanup

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
        "query_embedding_latency_ms": stats(embedding_times),
        "vector_search_latency_ms": stats(search_times),
        "total_retrieval_latency_ms": stats(retrieval_times),
        "iterations_per_query": iterations,
        "total_queries": len(queries),
        "total_measurements": len(retrieval_times),
        "note": "The vector search latency includes an additional query embedding step (so total retrieval latency is not exactly the sum of the two).",
    }


def ingest_document_pipeline(
    file_bytes: bytes,
    filename: str,
    settings: Settings,
    embedding_model,
    vector_store: QdrantStore,
) -> Tuple[bool, str]:
    """Helper function to ingest a document using the provided components."""
    # Validate
    errors = validate_in_memory(
        file_bytes,
        filename,
        max_size_mb=settings.max_file_size_mb,
    )
    if errors:
        return False, "; ".join(errors)

    # Parse
    try:
        docs, meta = parse_pdf_from_bytes(file_bytes, filename)
    except Exception as exc:
        return False, str(exc)

    # Chunk
    try:
        chunks, chunk_meta = chunk_documents(
            docs,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=settings.chunk_separators,
            document_name=filename,
        )
    except Exception as exc:
        return False, str(exc)

    # Enrich chunks with metadata
    for doc in chunks:
        doc.metadata.setdefault("document_name", filename)
        doc.metadata.setdefault("file_hash", meta.file_hash)

    # Embed
    try:
        texts = [doc.page_content for doc in chunks]
        embeddings = embedding_model.embed_documents(texts)
    except Exception as exc:
        return False, str(exc)

    # Insert into vector store
    try:
        from langchain_core.documents import Document as LCDocument

        embedded_docs = []
        for i, doc in enumerate(chunks):
            embedded_doc = LCDocument(
                page_content=doc.page_content,
                metadata={**doc.metadata, "embedding": embeddings[i]},
            )
            embedded_docs.append(embedded_doc)
        vector_store.add_documents(embedded_docs)
    except Exception as exc:
        return False, str(exc)

    return True, ""


def embed_and_search(
    query: str,
    embedding_model,
    vector_store: QdrantStore,
) -> List[Dict]:
    """Helper function to embed a query and search the vector store."""
    query_embedding = embedding_model.embed_query(query)
    # We cannot directly use the embedding because the search method expects a query string
    # and will embed it again. So we just call the search method with the query string.
    return vector_store.search(query=query, k=5)


def main():
    parser = argparse.ArgumentParser(description="Benchmark RAG performance with detailed timing")
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
        default="reports/benchmarks/benchmark_results_detailed.json",
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
        ingestion_results = benchmark_ingestion_detailed(
            settings, pdf_path, args.ingestion_iterations
        )
        results["benchmarks"]["ingestion"] = ingestion_results
        print(f"Ingestion benchmark completed: {ingestion_results['total']['mean']:.2f}s avg")
    elif args.document:
        print(f"Document not found: {args.document}")

    # Run retrieval benchmark
    print(f"Running retrieval benchmark with {len(args.queries)} queries...")
    retrieval_results = benchmark_retrieval_detailed(
        settings, args.queries, args.retrieval_iterations
    )
    results["benchmarks"]["retrieval"] = retrieval_results
    print(f"Retrieval benchmark completed: {retrieval_results['total_retrieval_latency_ms']['mean']:.2f}ms avg")

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())