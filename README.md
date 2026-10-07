# pdf-rag-chatbot

A production-grade Retrieval-Augmented Generation (RAG) chatbot for PDF documents with source citations, conversational memory, and robust document processing pipeline.

## Table of Contents
1. [Project Title](#project-title)
2. [Description](#description)
3. [Badges](#badges)
4. [Features Overview](#features-overview)
5. [PDF Ingestion](#pdf-ingestion)
6. [File Validation](#file-validation)
7. [Duplicate Detection](#duplicate-detection)
8. [PDF Parsing](#pdf-parsing)
9. [Text Chunking](#text-chunking)
10. [Embedding Generation](#embedding-generation)
11. [Vector Storage (Qdrant)](#vector-storage-qdrant)
12. [Similarity Search](#similarity-search)
13. [Answer Generation (LLM)](#answer-generation-llm)
14. [Source Citation](#source-citation)
15. [Conversational Memory (SQLite)](#conversational-memory-sqlite)
16. [Prompt Injection Defense](#prompt-injection-defense)
17. [Configuration Management](#configuration-management)
18. [Environment Variables](#environment-variables)
19. [Settings File](#settings-file)
20. [Ingestion Pipeline](#ingestion-pipeline)
21. [Chat Service](#chat-service)
22. [Evaluation Framework](#evaluation-framework)
23. [Retrieval Metrics](#retrieval-metrics)
24. [Benchmarking Methodology](#benchmarking-methodology)
25. [Ingestion Benchmark Results](#ingestion-benchmark-results)
26. [Retrieval Benchmark Results](#retrieval-benchmark-results)
27. [System Information for Benchmarks](#system-information-for-benchmarks)
28. [Evaluation Dataset](#evaluation-dataset)
29. [Evaluation Results](#evaluation-results)
30. [Testing Strategy](#testing-strategy)
31. [Unit Tests](#unit-tests)
32. [Integration Tests](#integration-tests)
33. [How to Run Tests](#how-to-run-tests)
34. [Deployment Considerations](#deployment-considerations)
35. [Docker Support](#docker-support)
36. [Kubernetes Support](#kubernetes-support)
37. [Cloud Deployment](#cloud-deployment)
38. [Contributing Guidelines](#contributing-guidelines)
39. [Code of Conduct](#code-of-conduct)
40. [Pull Request Process](#pull-request-process)
41. [Development Setup](#development-setup)
42. [License](#license)
43. [Acknowledgments](#acknowledgments)
44. [Contact Information](#contact-information)
45. [References](#references)
46. [Changelog](#changelog)
47. [FAQ](#faq)
48. [Troubleshooting](#troubleshooting)
49. [Documentation Audit](#documentation-audit)

## Project Title
pdf-rag-chatbot

## Description
A Retrieval-Augmented Generation (RAG) chatbot designed to interact with PDF documents. The system extracts text from PDFs, splits it into chunks, generates vector embeddings, stores them in a Qdrant vector database, and uses large language models to generate answers based on retrieved document chunks with proper source citations. The application includes conversational memory, file validation, duplicate detection, and prompt injection defenses.

## Badges
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.14+](https://img.shields.io/badge/Python-3.14+-blue.svg)
![Qdrant](https://img.shields.io/badge/Qdrant-VectorStore-brightgreen.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B.svg)

## Features Overview
The pdf-rag-chatbot implements the following core features:
- PDF text extraction and ingestion
- Configurable text chunking with overlap
- Multiple embedding model support (OpenAI, HuggingFace)
- Qdrant vector storage for efficient similarity search
- Large Language Model integration for answer generation
- Source citation with document references
- Conversational memory using SQLite
- File validation (type, size, magic bytes)
- Duplicate document detection via SHA-256 hashing
- Prompt injection defense mechanisms
- Configurable retrieval parameters
- Benchmarking and evaluation frameworks

## PDF Ingestion
The ingestion pipeline accepts PDF files through:
- File path ingestion (local files)
- In-memory byte ingestion (Streamlit file uploads)
The pipeline performs validation, parsing, chunking, and prepares documents for vector storage.

## File Validation
Implemented validation checks include:
- File extension verification (.pdf only)
- File size limits (configurable via settings)
- Magic bytes validation to confirm PDF format
- Protection against path traversal attacks

## Duplicate Detection
The system prevents duplicate document ingestion by:
- Computing SHA-256 hash of file content
- Checking hash against existing documents in vector store
- Skipping ingestion if document hash already exists

## PDF Parsing
PDF text extraction uses:
- PyPDFLoader for text extraction
- Page-level metadata preservation
- Error handling for corrupted or password-protected PDFs
- Support for both file paths and in-memory byte objects (via BytesIO)

## Text Chunking
Text splitting implementation features:
- Recursive character-based chunking
- Configurable chunk size (default: 1000 characters)
- Configurable chunk overlap (default: 200 characters)
- Preservation of document metadata in each chunk

## Embedding Generation
Supported embedding providers:
- OpenAI (text-embedding-3-small, text-embedding-ada-002)
- HuggingFace SentenceTransformers (all-MiniLM-L6-v2, others)
- Local embedding models for air-gapped environments
- Configurable via environment variables

## Vector Storage (Qdrant)
Vector database integration includes:
- Qdrant client for vector storage and similarity search
- Support for local, Docker, and cloud Qdrant instances
- Configurable collection name and vector parameters
- Automatic collection creation and management
- Payload storage for document metadata

## Similarity Search
Retrieval capabilities:
- Configurable top-K retrieval (default: 5 documents)
- Cosine similarity search
- Metadata filtering support
- Query embedding generation using configured embedding model

## Answer Generation (LLM)
Large Language Model support:
- OpenAI GPT models (gpt-3.5-turbo, gpt-4, etc.)
- Ollama local models (llama2, mistral, etc.)
- Anthropic Claude models
- Configurable temperature, max tokens, and other parameters
- Prompt engineering for grounded responses

## Source Citation
Answer attribution includes:
- Inline citations with document names and page numbers
- Source document tracking throughout the pipeline
- Format: [document_name:page_number]
- Verification that cited sources were actually retrieved

## Conversational Memory (SQLite)
Conversation persistence features:
- SQLite-backed conversation storage
- Conversation history maintenance
- Context-aware responses based on chat history
- Configurable database path
- Thread-safe access patterns

## Prompt Injection Defense
Security measures implemented:
- Input sanitization and validation
- Prompt template separation from user input
- Token limits on user queries
- Basic prompt injection pattern detection
- Output encoding to prevent XSS in web interface

## Configuration Management
Configuration system built on:
- Pydantic-settings for environment variable management
- Type-safe configuration objects
- Hierarchical configuration (defaults, env vars, .env files)
- Runtime configuration validation
- Settings hot-reload capability

## Environment Variables
Key configuration variables:
- `QDRANT_MODE`: local/server/memory
- `QDRANT_HOST`: Qdrant server host
- `QDRANT_PORT`: Qdrant server port
- `QDRANT_API_KEY`: Qdrant cloud API key
- `COLLECTION_NAME`: Vector store collection name
- `EMBEDDING_PROVIDER`: openai/local/huggingface
- `OPENAI_API_KEY`: OpenAI API key
- `LLM_PROVIDER`: openai/ollama/anthropic
- `CHUNK_SIZE`: Text chunk size (characters)
- `CHUNK_OVERLAP`: Chunk overlap (characters)
- `RETRIEVAL_K`: Number of documents to retrieve
- `CONVERSATION_DB_PATH`: SQLite database path
- `MAX_FILE_SIZE_MB`: Maximum upload file size
- `LOG_LEVEL`: Logging verbosity

## Settings File
Configuration hierarchy:
- Default values in code
- Override via environment variables
- Optional `.env` file for local development
- Runtime validation of all settings
- Typesafe access throughout application

## Ingestion Pipeline
End-to-end document processing:
1. File validation (extension, size, magic bytes)
2. PDF parsing (text extraction with page metadata)
3. Text chunking (recursive splitting with overlap)
4. Metadata enrichment (filename, hash, timestamps)
5. Preparation for embedding generation
6. Returns processed chunks ready for vector storage

## Chat Service
Core orchestration component:
- Query processing pipeline
- Conversation memory integration
- Retrieval from vector store
- Context construction from retrieved documents
- LLM prompt engineering
- Answer generation with source citations
- Response formatting and memory updates

## Evaluation Framework
RAG quality assessment system:
- Hit@K: Relevant document in top-K results
- Precision@K: Fraction of retrieved documents that are relevant
- Recall@K: Fraction of relevant documents retrieved
- MRR: Mean Reciprocal Rank of first relevant document
- Configurable K parameter
- Dataset-driven evaluation
- JSON export of detailed results

## Retrieval Metrics
Implemented evaluation metrics:
- **Hit@K**: Binary indicator if any expected source appears in retrieved top-K
- **Precision@K**: Proportion of retrieved documents that are expected sources
- **Recall@K**: Proportion of expected sources that were retrieved
- **MRR**: Reciprocal rank of the first correctly retrieved document
- All metrics computed per query and aggregated

## Benchmarking Methodology
Performance measurement approach:
- Isolated benchmarking scripts
- Warm-up runs to stabilize measurements
- Multiple iterations for statistical significance
- System information collection (CPU, memory, Python version)
- Clear separation of ingestion and retrieval benchmarks
- Latency measurements in milliseconds
- Throughput measurements where applicable

## Ingestion Benchmark Results
Measured performance for PDF ingestion pipeline:
- Benchmarked on fixture documents (company_policy.pdf, product_manual.pdf, employee_handbook.pdf)
- Includes validation, parsing, chunking, embedding, and storage
- Results from benchmark_fixture_results.json:
  - Mean ingestion time: 1.53 seconds
  - Median ingestion time: 0.011 seconds (shows variance due to model loading)
  - Min ingestion time: 0.011 seconds
  - Max ingestion time: 4.58 seconds
  - Based on 3 iterations per document

## Retrieval Benchmark Results
Query latency measurements:
- Benchmarked with sample queries against ingested fixture documents
- Includes query embedding generation and vector search
- Results from benchmark_fixture_results.json:
  - Mean query embedding latency: 3.45 ms
  - Mean vector search latency: 3.69 ms
  - Mean total retrieval latency: 7.14 ms
  - P50 total latency: 6.80 ms
  - P95 total latency: 8.79 ms
  - P99 total latency: 8.79 ms
  - Based on 5 iterations per query across 2 queries

## System Information for Benchmarks
Benchmark execution environment:
- Timestamp: 2026-10-01T20:07:51.893783+00:00
- Python version: 3.14.4 (main, Aug 20 2026, 10:41:58) [GCC 15.2.0]
- Platform: linux
- CPU count: 12 cores
- Total memory: 15.0 GB
- Process ID: 228534

## Evaluation Dataset
Test question-answer pairs:
- Created from ingested fixture documents
- Contains 8 question-answer pairs
- Each question has expected answer and source citation
- Topics covered: vacation policy, sick leave, product returns, warranty, dress code, office hours, remote work, support contact
- Stored in evaluation_dataset.json
- Example question: "How many vacation days are provided per year?"
- Expected answer: "20 days"
- Expected source: "company_policy.pdf:1"

## Evaluation Results
RAG quality assessment outcomes:
- Evaluated using scripts/evaluate_rag.py with evaluation_dataset.json
- Results showing 0% hit rate, precision, recall, and MRR
- Indicates retrieval system did not return expected sources for test questions
- Note: Evaluation used local embedding model without OpenAI API key
- Results saved to evaluation_results.json
- Highlights need for proper embedding model configuration for accurate evaluation

## Testing Strategy
Quality assurance approach:
- Unit tests for individual components
- Integration tests for pipeline interactions
- Test fixtures for consistent test data
- Mocking of external services (LLM APIs, vector store)
- Focus on business logic validation
- Continuous integration readiness

## Unit Tests
Component-level testing:
- Tests for evaluation metrics (test_metrics.py)
- Tests for evaluation runner (test_evaluator.py)
- Mock-based isolation of dependencies
- Validation of metric calculations
- Edge case testing (empty lists, perfect matches, etc.)
- Located in tests/unit/ directory

## Integration Tests
System-level testing:
- Currently limited to benchmark scripts as integration validation
- End-to-end pipeline testing via benchmark_ingestion_retrieval.py
- Vector store interaction testing
- Embedding generation verification
- Document processing flow validation

## How to Run Tests
Test execution instructions:
```
# Install test dependencies
pip install pytest

# Run unit tests
pytest tests/unit/

# Run all tests
pytest tests/
```
Note: Some tests may require API keys for external services

## Deployment Considerations
Production deployment factors:
- Resource allocation (CPU, memory, disk space)
- Qdrant instance sizing based on document volume
- Embedding model GPU/CPU requirements
- LLM API rate limits and costs
- Concurrent user scaling
- Backup and disaster recovery for conversation database
- Monitoring and logging infrastructure

## Docker Support
Containerization availability:
- No Dockerfile currently present in repository
- Application can be containerized using base Python image
- Required environment variables for configuration
- Volume mounts for persistent Qdrant storage and conversation DB
- Example docker run command would need to be created

## Kubernetes Support
Orchestration readiness:
- No Kubernetes manifests present
- Would require Deployment, Service, and ConfigMap resources
- Persistent volumes for Qdrant and SQLite storage
- Resource requests and limits configuration
- Horizontal pod autoscaling based on CPU/memory

## Cloud Deployment
Platform-specific considerations:
- AWS: Qdrant on EC2/EKS, S3 for document storage (optional)
- GCP: Qdrant on GKE/GCE, Cloud SQL alternative for conversations
- Azure: Qdrant on AKS/ACI, Cosmos DB or PostgreSQL for conversations
- All require proper networking and security group configuration
- Secret management for API keys (AWS Secrets Manager, etc.)

## Contributing Guidelines
Development workflow:
- Fork the repository
- Create feature branch from main
- Implement changes with corresponding tests
- Ensure all existing tests pass
- Submit pull request with clear description
- Follow code style and conventions
- Documentation updates for user-facing changes

## Code of Conduct
Community standards:
- Respectful and inclusive communication
- Constructive feedback and collaboration
- No tolerance for harassment or discrimination
- Adherence to project goals and objectives
- Reporting process for violations

## Pull Request Process
Submission requirements:
- Descriptive title and detailed description
- Link to related issues if applicable
- Screenshots or examples for UI changes
- Passing status on all CI checks
- Review by at least one maintainer
- Squash merge policy for clean history

## Development Setup
Local environment configuration:
1. Clone repository: `git clone <repository-url>`
2. Create virtual environment: `python -m venv venv`
3. Activate environment: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Copy .env.example to .env and configure
6. Initialize Qdrant (local or remote)
7. Run application: `streamlit run app.py`

## License
MIT License

Copyright (c) 2026 pdf-rag-chatbot contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgments
- OpenAI for embedding and language model APIs
- HuggingFace for open-source embedding models
- Qdrant team for vector database technology
- Streamlit team for rapid prototyping framework
- LangChain community for document processing abstractions
- Contributors to PyPDFLoader for PDF text extraction

## Contact Information
Maintainer: Beatrice Oira
Email: schoolbeatriceoira@gmail.com
Project URL: https://github.com/Beatriceoira/pdf-rag-chatbot

## References
- LangChain documentation: https://python.langchain.com/
- Qdrant documentation: https://qdrant.tech/documentation/
- Streamlit documentation: https://docs.streamlit.io/
- PyPDFLoader documentation: https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.PyPDFLoader.html
- SentenceTransformers documentation: https://www.sbert.net/

## Changelog
### Unreleased
- Fixed BytesIO handling in PDF parser and metadata hash computation
- Corrected variable name typo in evaluation metrics
- Added conversation_id parameter to evaluation runner for consistent testing
- Created benchmarking scripts and evaluation dataset
- Ingested fixture documents for testing

### v0.1.0 (Initial Release)
- Core RAG chatbot functionality
- PDF ingestion pipeline
- Vector storage with Qdrant
- LLM-based answer generation
- Conversational memory
- Basic configuration system

## FAQ
**Q: What PDF formats are supported?**  
A: Standard PDF files with extractable text. Scanned PDFs require OCR preprocessing.

**Q: Can I use local models without API keys?**  
A: Yes, by setting EMBEDDING_PROVIDER=local and LLM_PROVIDER=ollama with appropriate local models running.

**Q: How do I change the chunk size?**  
A: Set the CHUNK_SIZE environment variable (default: 1000 characters).

**Q: Is the conversational memory shared between users?**  
A: No, each conversation ID maintains separate memory. The Streamlit app generates unique IDs per session.

**Q: What is the maximum file size for uploads?**  
A: Configurable via MAX_FILE_SIZE_MB environment variable (default: 10 MB).

**Q: How does duplicate detection work?**  
A: SHA-256 hash of file content is computed and checked against existing documents.

## Troubleshooting
**Issue: "No documents retrieved for query"**  
- Verify documents have been ingested into vector store
- Check embedding model configuration matches ingestion settings
- Ensure Qdrant connection is properly configured
- Validate that query text is meaningful and related to ingested content

**Issue: Embedding model loading failures**  
- Confirm local model is properly installed and accessible
- Check internet connectivity for HuggingFace model downloads
- Verify sufficient memory for model loading
- Try smaller embedding model if resources are limited

**Issue: Slow response times**  
- Check Qdrant instance performance and resource allocation
- Consider increasing RETRIEVAL_K for better recall (trade-off with latency)
- Verify embedding model is loaded and ready
- Monitor system resources during operation

**Issue: Conversation memory errors**  
- Ensure SQLite database file is writable
- Check available disk space for conversation DB
- Verify database path configuration is correct
- Consider clearing old conversations if database grows large

**Issue: PDF validation failures**  
- Confirm file is a valid PDF (not corrupted)
- Check file extension is .pdf
- Verify file size is within limits
- Ensure file is not password-protected or encrypted

**Issue: Missing API keys for external services**  
- Set required environment variables (OPENAI_API_KEY, etc.)
- Use local alternatives for embeddings and LLMs when API keys unavailable
- Check .env file for proper variable definitions
