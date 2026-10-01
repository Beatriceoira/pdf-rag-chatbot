# PDF RAG Chatbot

A production-grade, document-grounded AI assistant for querying PDF collections using LangChain, Qdrant, embeddings, and LLMs.

## Overview

This application allows users to upload PDF documents, automatically extract and chunk text, generate embeddings, store vectors in Qdrant, and then chat with their documents using retrieval-augmented generation (RAG). Answers are grounded in the retrieved context with source citations.

## Features

- **Multi-document support** — Upload and query multiple PDFs simultaneously
- **Grounded answers** — LLM responses are constrained to retrieved document context
- **Source citations** — Each answer includes document name, page number, and similarity score
- **Duplicate detection** — SHA-256 hashing prevents re-indexing identical files
- **Page-aware chunking** — Chunks preserve page numbers for accurate citations
- **Configurable chunking** — Adjustable chunk size, overlap, and separators
- **Hybrid retrieval** — Vector similarity search with metadata filtering
- **Optional reranking** — Cross-encoder reranking for improved relevance
- **Conversation memory** — SQLite-backed persistent chat history
- **Anti-injection hardening** — System prompt guards against malicious document content
- **Local & cloud modes** — Swap between local embeddings (HuggingFace) and OpenAI
- **Streaming responses** — Token-by-token output when supported by the LLM
- **Export conversations** — Download as JSON or Markdown
- **Health checks** — Verify Qdrant, embedding model, and LLM status
- **Docker support** — Run with `docker compose up`
- **CI/CD** — GitHub Actions for linting and testing

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full architecture diagram.

```
PDF Upload → Validation → Parsing → Chunking → Embedding → Qdrant
                                                                  ↓
User Query → Query Processing → Retrieval → (Reranking) → LLM → Grounded Answer
                                                                  ↓
                                                           Source Citations
```

## Technology Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| RAG Framework | LangChain |
| Vector Store | Qdrant |
| Embeddings (Cloud) | OpenAI `text-embedding-3-small` |
| Embeddings (Local) | HuggingFace `all-MiniLM-L6-v2` |
| LLM (Cloud) | OpenAI (`gpt-4o-mini`, `gpt-4o`) |
| LLM (Local) | Ollama (optional) |
| Database | SQLite (conversation history) |
| PDF Parsing | PyPDF |
| Testing | pytest |
| Linting | Ruff |

## Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Required:
- `OPENAI_API_KEY` — For chat model (and optional embeddings)

Optional:
- `QDRANT_MODE` — `local` (default), `server`, or `memory`
- `EMBEDDING_PROVIDER` — `local` (default) or `openai`
- `LLM_PROVIDER` — `openai` (default), `ollama`, or `anthropic`
- `CHUNK_SIZE`, `CHUNK_OVERLAP`, `RETRIEVAL_K`

## Running Locally

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Qdrant Setup

By default, Qdrant runs in embedded mode (`./qdrant_data/`). For a standalone server:

```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

Then set `QDRANT_MODE=server` and `QDRANT_URL=http://localhost:6333`.

## Local LLM Setup (Ollama)

```bash
# Install Ollama from https://ollama.com
ollama pull llama3.2
```

Set `LLM_PROVIDER=ollama` and `LLM_MODEL=llama3.2` in your `.env`.

## Usage

1. Start the app: `streamlit run app.py`
2. Enter your OpenAI API key in the sidebar (or use local embeddings + local LLM)
3. Configure chunk size, overlap, and top-k in the sidebar
4. Upload one or more PDFs and click **Process documents**
5. Ask questions in the chat — answers cite sources with page numbers

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=term-missing -v

# Run specific test suites
pytest tests/unit -v
pytest tests/integration -v
pytest tests/rag -v
pytest tests/security -v
```

## RAG Evaluation

```bash
python scripts/evaluate_rag.py --dataset my_dataset.json --output results.json
```

See [docs/EVALUATION.md](docs/EVALUATION.md) for details.

## Security

- **File validation**: Extension, magic bytes, size checks on upload
- **Duplicate detection**: SHA-256 hashing prevents re-indexing
- **Prompt injection resistance**: System prompt treats all document content as untrusted data
- **Secret protection**: API keys never appear in logs, UI, or string representations
- **Resource limits**: Configurable max file size and query length

## Performance

- Embeddings are batched where supported
- Qdrant collections use COSINE distance for semantic similarity
- Lazy initialization of models and clients
- SQLite for conversation persistence (no external DB required)

## Project Structure

```
pdf-rag-chatbot/
├── app.py                     # Streamlit entry point
├── src/
│   ├── config/settings.py     # Typed configuration
│   ├── ingestion/             # PDF parsing, chunking
│   ├── embeddings/            # Embedding model factory
│   ├── vectorstore/           # Qdrant integration
│   ├── retrieval/             # Query processing, retrieval, reranking
│   ├── llm/                   # LLM factory, prompts, answer generation
│   ├── chat/                  # Conversation memory, chat service
│   ├── evaluation/            # RAG evaluation metrics
│   └── utils/                 # Logging, hashing, exceptions
├── tests/
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   ├── rag/                   # RAG-specific tests
│   └── security/              # Security tests
├── docs/                      # Documentation
├── scripts/                   # Utility scripts
├── data/                      # Persistent data (SQLite, exports)
├── qdrant_data/               # Qdrant embedded storage
├── .env.example               # Environment variable template
├── Dockerfile                 # Container build
├── docker-compose.yml         # Multi-service deployment
└── requirements.txt           # Python dependencies
```

## Docker

```bash
docker compose up
```

Access at http://localhost:8501.

## CI/CD

GitHub Actions runs on every push/PR:
- Ruff linting and formatting check
- Unit tests with coverage
- RAG and security tests

## Future Improvements

- [ ] PostgreSQL/Redis backend for conversation memory
- [ ] Multi-user support with session isolation
- [ ] Real-time document sync (watch for file changes)
- [ ] Advanced reranking with Cohere or custom models
- [ ] Agent-based follow-up questioning
- [ ] Web interface for management without Streamlit
- [ ] PDF text extraction improvements (layout-aware)
- [ ] Evaluation dashboard with plots

## License

MIT
