"""Streamlit application entry point for the PDF RAG Chatbot."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ── Ensure project root is on sys.path ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

load_dotenv()

# ── Import application modules ───────────────────────────────────────────────
from src.chat.memory import ConversationMemory
from src.chat.service import ChatService
from src.config.settings import get_settings
from src.ingestion.pipeline import IngestionPipeline
from src.llm.factory import LLMFactory
from src.utils.logging import get_logger, setup_logging

# ── Logging ──────────────────────────────────────────────────────────────────
settings = get_settings()
setup_logging(level=settings.log_level)
logger = get_logger("app")

# ── Streamlit Page Config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session State Initialization ─────────────────────────────────────────────
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "chat_service" not in st.session_state:
    st.session_state.chat_service = None
if "ingestion_pipeline" not in st.session_state:
    st.session_state.ingestion_pipeline = None
if "store" not in st.session_state:
    st.session_state.store = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "document_registry" not in st.session_state:
    st.session_state.document_registry = {}  # {file_hash: meta}


# ── Helper Functions ──────────────────────────────────────────────────────────


def _get_memory() -> ConversationMemory:
    """Get or create the conversation memory instance."""
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory(settings.conversation_db_path)
    return st.session_state.memory


def _new_conversation():
    """Create a new conversation."""
    mem = _get_memory()
    conv_id = mem.create_conversation()
    st.session_state.conversation_id = conv_id
    st.session_state.document_registry = {}
    st.toast("New conversation created")
    st.rerun()


def _clear_conversation():
    """Clear current conversation messages."""
    if not st.session_state.conversation_id:
        return
    mem = _get_memory()
    st.session_state.conversation_id = mem.create_conversation("New chat")
    st.toast("Conversation cleared")
    st.rerun()


def _get_chat_service() -> ChatService | None:
    """Lazily initialize the chat service."""
    if st.session_state.chat_service is not None:
        return st.session_state.chat_service

    try:
        from src.vectorstore.qdrant_store import QdrantStore

        llm = LLMFactory.create(settings)
        store = QdrantStore(settings)
        mem = _get_memory()
        chat_service = ChatService(settings, store, llm, mem)
        st.session_state.chat_service = chat_service
        st.session_state.store = store
        st.session_state.llm = llm
        return chat_service
    except Exception as exc:
        logger.error("Failed to initialize chat service: %s", exc)
        st.error(f"Failed to initialize chat service: {exc}")
        return None


def _render_document_list():
    """Render the document registry in the sidebar."""
    mem = _get_memory()
    conv_id = st.session_state.conversation_id
    if not conv_id:
        st.caption("No conversation active")
        return

    docs = mem.get_documents(conv_id)
    if not docs:
        st.caption("No documents uploaded")
        return

    for doc in docs:
        name = doc["document_name"]
        pages = doc.get("page_count", "?")
        chunks = doc.get("chunk_count", "?")
        st.markdown(f"✅ **{name}**")
        st.caption(f"{pages} pages · {chunks} chunks")

        col_del, col_reidx = st.columns([1, 1])
        with col_del:
            if st.button("🗑️", key=f"del_{doc['id']}", help="Remove document"):
                st.session_state.document_registry.pop(doc["file_hash"], None)
                st.toast(f"Removed {name}")
                st.rerun()
        with col_reidx:
            if st.button("🔄", key=f"reidx_{doc['id']}", help="Re-index document"):
                st.session_state.document_registry.pop(doc["file_hash"], None)
                st.toast(f"Re-indexing {name}")
                st.rerun()


def _export_conversation(fmt: str):
    """Export the current conversation."""
    mem = _get_memory()
    conv_id = st.session_state.conversation_id
    if not conv_id:
        st.warning("No active conversation to export.")
        return
    data = mem.export_conversation(conv_id, fmt=fmt)
    ext = "json" if fmt == "json" else "md"
    st.download_button(
        label=f"Download conversation.{ext}",
        data=data.encode("utf-8"),
        file_name=f"conversation.{ext}",
        mime=f"text/{'json' if fmt == 'json' else 'markdown'}",
    )


def _show_health_check():
    """Show the health check panel."""
    cs = _get_chat_service()
    if cs is None:
        st.error("Chat service not initialized.")
        return
    with st.spinner("Checking health..."):
        status = cs.health_check()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Application", "✓" if status["application"] else "✗")
        st.metric("Qdrant", "✓" if status["qdrant"] else "✗")
        st.metric("Embedding", "✓" if status["embedding_model"] else "✗")
    with col2:
        st.metric("LLM", "✓" if status["llm"] else "✗")
        st.metric("Documents", status.get("vectors", 0))
        if status.get("embedding_dimension"):
            st.metric("Embedding dim", status["embedding_dimension"])


def _process_uploaded_files(files):
    """Ingest uploaded PDF files into the vector store."""
    pipeline = IngestionPipeline(settings)
    store = st.session_state.store or _get_chat_service()
    if store is None:
        st.error("Could not initialize vector store.")
        return

    mem = _get_memory()
    conv_id = st.session_state.conversation_id or mem.create_conversation()
    st.session_state.conversation_id = conv_id

    success_count = 0
    error_count = 0
    total_chunks = 0

    for uploaded_file in files:
        file_bytes = uploaded_file.read()
        filename = uploaded_file.name

        result = pipeline.ingest_from_bytes(file_bytes, filename)

        if result.success:
            # Check for duplicates
            file_hash = result.document_meta.file_hash
            if file_hash in st.session_state.document_registry:
                st.warning(f"⚠️  '{filename}' is a duplicate — already indexed.")
                continue

            store.add_documents(result.chunks)
            mem.register_document(
                conversation_id=conv_id,
                document_name=filename,
                file_hash=file_hash,
                page_count=result.document_meta.page_count,
                chunk_count=result.document_meta.chunk_count,
            )
            st.session_state.document_registry[file_hash] = result.document_meta
            total_chunks += len(result.chunks)
            success_count += 1
            logger.info("Indexed '%s': %d chunks", filename, len(result.chunks))
        else:
            error_count += 1
            logger.warning("Failed to ingest '%s': %s", filename, result.error)
            st.error(f"Failed to process '{filename}': {result.error}")

    return success_count, error_count, total_chunks


def _render_sources(sources: list[dict]):
    """Render source citations as an expander."""
    with st.expander(f"📚 Sources ({len(sources)})"):
        for i, src in enumerate(sources, 1):
            doc_name = src.get("document_name", "unknown")
            page = src.get("page_number", "?")
            score = src.get("score")
            preview = src.get("text_preview", "")

            score_str = f" · similarity: {score:.4f}" if score is not None else ""
            st.markdown(f"**{i}.** `{doc_name}` — Page {page}{score_str}")
            if preview:
                st.caption(preview)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    # ── Mode Indicator ─────────────────────────────────────────────────────
    mode = "☁️ Cloud Mode" if settings.embedding_provider == "openai" else "🖥️ Local Mode"
    st.markdown(f"**Mode:** {mode}")

    # ── API Key ────────────────────────────────────────────────────────────
    api_key = st.text_input(
        "OpenAI API key",
        type="password",
        value=os.environ.get("OPENAI_API_KEY", ""),
        help="Needed for the chat model and/or OpenAI embeddings.",
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        settings.openai_api_key = api_key

    # ── Embedding Provider ─────────────────────────────────────────────────
    embedding_choice = st.selectbox(
        "Embeddings",
        ["OpenAI (text-embedding-3-small)", "Local (HuggingFace all-MiniLM-L6-v2)"],
        help="Local embeddings run on your machine — no API key or cost, but slower on first run.",
    )
    settings.embedding_provider = "openai" if embedding_choice.startswith("OpenAI") else "local"

    # ── Chat Model ─────────────────────────────────────────────────────────
    model_choice = st.selectbox(
        "Chat model",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    )
    settings.llm_model = model_choice

    # ── Chunking ───────────────────────────────────────────────────────────
    chunk_size = st.slider("Chunk size", 200, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 500, 150, 50)
    settings.chunk_size = chunk_size
    settings.chunk_overlap = chunk_overlap

    # ── Retrieval ──────────────────────────────────────────────────────────
    retrieval_k = st.slider("Top-k retrieval", 1, 10, 5, 1)
    settings.retrieval_k = retrieval_k

    enable_reranking = st.toggle("Enable reranking", value=False)
    settings.enable_reranking = enable_reranking

    st.divider()

    # ── Document Upload ────────────────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "Upload PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
    )
    process_btn = st.button("🚀 Process documents", use_container_width=True)

    st.divider()

    # ── Conversation Controls ──────────────────────────────────────────────
    col_new, col_clear = st.columns(2)
    with col_new:
        if st.button("🆕 New conversation", use_container_width=True):
            _new_conversation()
    with col_clear:
        if st.button("🗑️ Clear chat", use_container_width=True):
            _clear_conversation()

    st.divider()

    # ── Document List ──────────────────────────────────────────────────────
    st.subheader("📚 Documents")
    _render_document_list()

    st.divider()

    # ── Export ─────────────────────────────────────────────────────────────
    if st.session_state.conversation_id:
        if st.button("📥 Export JSON", use_container_width=True):
            _export_conversation("json")
        if st.button("📥 Export Markdown", use_container_width=True):
            _export_conversation("markdown")

    st.divider()

    # ── Health Check ───────────────────────────────────────────────────────
    if st.button("🏥 Health Check", use_container_width=True):
        _show_health_check()


# ── Process Uploaded PDFs ─────────────────────────────────────────────────────
if process_btn and uploaded_files:
    chat_service = _get_chat_service()
    if chat_service is None:
        st.sidebar.error("Failed to initialize chat service. Check settings.")
    elif not settings.openai_api_key and settings.embedding_provider == "openai":
        st.sidebar.error("Enter your OpenAI API key first.")
    else:
        with st.spinner("Reading, chunking & embedding PDFs..."):
            success, errors, chunks = _process_uploaded_files(uploaded_files)
        if success > 0:
            st.sidebar.success(f"Indexed {chunks} chunks from {success} file(s).")
        if errors > 0:
            st.sidebar.warning(f"{errors} file(s) failed to process.")

# ── Main Chat UI ─────────────────────────────────────────────────────────────
st.title("📄 Chat with your PDFs")
st.caption("Document-grounded AI assistant · LangChain + Qdrant RAG")

# Ensure conversation exists
mem = _get_memory()
if not st.session_state.conversation_id:
    st.session_state.conversation_id = mem.create_conversation()

chat_service = _get_chat_service()

# ── Render chat history ──────────────────────────────────────────────────────
messages = mem.get_messages(st.session_state.conversation_id) if st.session_state.conversation_id else []
for msg in messages:
    role = msg["role"]
    content = msg["content"]
    sources = msg.get("sources")
    with st.chat_message(role):
        st.markdown(content)
        if sources:
            _render_sources(sources)

# ── Handle new user input ────────────────────────────────────────────────────
if prompt := st.chat_input("Ask something about your documents..."):
    if chat_service is None:
        st.warning("Upload and process a PDF first (see the sidebar).")
    else:
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_service.ask(
                    question=prompt,
                    conversation_id=st.session_state.conversation_id,
                )
            if response.success:
                st.markdown(response.answer)
                if response.sources:
                    _render_sources(response.sources)
                if st.toggle("Show debug info", key="debug"):
                    with st.expander("Debug Info"):
                        st.json(
                            {
                                "model": response.model_name,
                                "retrieval_latency_ms": round(response.retrieval_latency_ms, 1),
                                "generation_latency_ms": round(response.generation_latency_ms, 1),
                                "num_sources": len(response.sources),
                            }
                        )
            else:
                st.error(f"Error: {response.error}")

        # Re-render to show the new message
        st.rerun()


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Built with LangChain, Qdrant, Streamlit · "
    f"Chunk size: {settings.chunk_size} · "
    f"Overlap: {settings.chunk_overlap} · "
    f"Top-k: {settings.retrieval_k}"
)
