#!/usr/bin/env python3
"""Create benchmark PDF documents of various sizes."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def create_pdf_with_content(text: str, pages: int = 1) -> bytes:
    """Create a PDF with the given text repeated across the specified number of pages."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    for page in range(pages):
        y = height - 50  # Start near the top
        lines = text.strip().split('\n')
        for line in lines:
            if y < 50:  # Bottom margin
                c.showPage()
                y = height - 50
            c.drawString(50, y, line)
            y -= 20  # Line spacing
        c.showPage()  # End the page

    c.save()
    buffer.seek(0)
    return buffer.read()


def create_benchmark_pdfs(directory: Path) -> dict[str, Path]:
    """Create sample PDFs for benchmarking.

    Returns dict mapping filename → path.
    """
    directory.mkdir(parents=True, exist_ok=True)

    # Small document (~1 page)
    small_text = """
    This is a small test document for benchmarking.
    It contains basic information about RAG systems.
    The quick brown fox jumps over the lazy dog.
    """ * 20  # Repeat to get reasonable content

    # Medium document (~5 pages)
    medium_text = """
    This is a medium-sized test document for performance benchmarking.
    It contains multiple paragraphs discussing various aspects of
    Retrieval-Augmented Generation (RAG) systems.

    Section 1: Introduction
    Retrieval-Augmented Generation combines the strengths of
    information retrieval and text generation. Unlike standard
    language models that rely solely on parametric knowledge,
    RAG systems retrieve relevant information from an external
    knowledge base to generate more accurate and grounded responses.

    Section 2: Architecture
    A typical RAG system consists of four main components:
    1. Document ingestion pipeline (PDF parsing, chunking, embedding)
    2. Vector storage system (for efficient similarity search)
    3. Retrieval mechanism (to find relevant context)
    4. Generation component (LLM that produces grounded answers)

    Section 3: Embedding Models
    Embedding models convert text into numerical vectors that
    capture semantic meaning. Common choices include:
    - OpenAI's text-embedding-3-small (384 dimensions)
    - HuggingFace's all-MiniLM-L6-v2 (384 dimensions)
    - Local sentence-transformers models

    Section 4: Vector Stores
    Vector stores enable efficient similarity search in high-dimensional spaces.
    Popular options include:
    - Qdrant (used in this project)
    - FAISS
    - Pinecone
    - Weaviate
    """ * 3

    # Large document (~15 pages)
    large_text = medium_text + """

    Section 5: Chunking Strategies
    Effective chunking is crucial for RAG performance. Strategies include:
    - Fixed-size chunking (simple but may break semantic units)
    - Recursive chunking (respects paragraph/sentence boundaries)
    - Semantic chunking (uses embedding similarity to group related text)
    - Layout-aware chunking (preserves document structure)

    Section 6: Retrieval Techniques
    Beyond basic vector search, advanced retrieval methods include:
    - Hybrid search (combines vector and keyword search)
    - Re-ranking (uses cross-encoders to improve initial results)
    - Query expansion (expands the original query with related terms)
    - Metadata filtering (narrows search by document attributes)

    Section 7: Evaluation Metrics
    Standard metrics for evaluating RAG systems:
    - Hit@K: Proportion of queries where relevant docs appear in top-K
    - Precision@K: Proportion of retrieved docs that are relevant
    - Recall@K: Proportion of relevant docs that are retrieved
    - MRR: Mean Reciprocal Rank of first relevant document
    - Groundedness: Whether answers are supported by retrieved context
    - Citation correctness: Whether cited sources actually contain claimed info

    Section 8: Optimization Techniques
    Performance can be improved through:
    - Batch embedding generation
    - Efficient vector indexing (IVF, HNSW)
    - Model quantization
    - Caching frequent queries
    - Asynchronous processing

    Section 9: Security Considerations
    Important security aspects of RAG systems:
    - Prompt injection prevention (treat retrieved content as untrusted)
    - Input validation and sanitization
    - Secure handling of API keys and credentials
    - File type and size validation for uploads
    - Protection against path traversal attacks
    - Safe handling of potentially malicious PDF content

    Section 10: Future Directions
    Emerging trends in RAG research:
    - Multimodal RAG (handling images, tables, and layouts)
    - Agentic RAG (using LLMs to plan and execute retrieval strategies)
    - Real-time RAG (continuous updates to the knowledge base)
    - Federated RAG (distributed knowledge sources with privacy preservation)
    - Specialized RAG (domain-specific adaptations for medicine, law, etc.)
    """ * 2

    files = {
        "benchmark_small.pdf": small_text,
        "benchmark_medium.pdf": medium_text,
        "benchmark_large.pdf": large_text,
    }

    results = {}
    for name, content in files.items():
        path = directory / name
        pdf_bytes = create_pdf_with_content(content, pages=1 if "small" in name else (5 if "medium" in name else 15))
        path.write_bytes(pdf_bytes)
        results[name] = path
        print(f"Created {name}: {len(pdf_bytes)} bytes")

    return results


if __name__ == "__main__":
    base_dir = Path(__file__).parent
    create_benchmark_pdfs(base_dir)