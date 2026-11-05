"""Sidebar component for Alethia UI.

Handles application branding, chat session controls, multi-format document upload (PDF/TXT),
direct text note ingestion, per-document management, and knowledge base capacity metrics.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import streamlit as st

from src.chunker import chunk_document
from src.config import is_groq_configured
from src.document_loader import (
    Document,
    create_document_from_text,
    load_document_from_bytes,
)
from src.store import VectorStore, get_vector_store
from src.text_cleaner import clean_document


def get_database_stats(store: Optional[VectorStore] = None) -> Dict[str, Any]:
    """Retrieve actual document and chunk statistics from the persistent ChromaDB collection.

    Args:
        store: Optional VectorStore instance.

    Returns:
        Dict with 'chunks', 'documents', 'sources', and 'doc_details'.
    """
    target_store = store or get_vector_store()
    try:
        total_chunks = target_store.count()
        if total_chunks == 0:
            return {"chunks": 0, "documents": 0, "sources": [], "doc_details": {}}

        # Query all metadatas to discover unique indexed document sources and their metrics
        records = target_store.collection.get(include=["metadatas"])
        metadatas = records.get("metadatas", [])

        doc_details: Dict[str, Dict[str, Any]] = {}
        for m in metadatas:
            if isinstance(m, dict) and m.get("source"):
                src = m["source"]
                if src not in doc_details:
                    ext = Path(src).suffix.replace(".", "").upper() or "TXT"
                    doc_details[src] = {"chunks": 0, "chars": 0, "type": ext, "status": "Indexed"}
                doc_details[src]["chunks"] += 1
                doc_details[src]["chars"] += int(m.get("char_count", 0))

        sources = sorted(list(doc_details.keys()))

        # Convert character counts to approximate file size in KB
        for src, info in doc_details.items():
            info["size_kb"] = round(info["chars"] / 1024, 1) if info["chars"] > 0 else 0.1

        return {
            "chunks": total_chunks,
            "documents": len(sources),
            "sources": sources,
            "doc_details": doc_details,
        }
    except Exception:
        return {"chunks": 0, "documents": 0, "sources": [], "doc_details": {}}


def handle_document_upload(uploaded_files: List[Any], store: Optional[VectorStore] = None) -> None:
    """Validate, parse, chunk, and index uploaded .pdf and .txt files into ChromaDB.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects.
        store: Optional VectorStore instance.
    """
    if not uploaded_files:
        return

    target_store = store or get_vector_store()
    indexed_total = 0

    for file in uploaded_files:
        filename = getattr(file, "name", "uploaded_document.txt")
        suffix = Path(filename).suffix.lower()

        # 1. Validate file extension (.txt or .pdf)
        if suffix not in (".txt", ".pdf"):
            st.sidebar.error(f"Unsupported format: `{filename}`. Supported formats: `.pdf`, `.txt`.")
            continue

        # 2. Read bytes and parse using document loader
        try:
            raw_bytes = file.read()
            if not raw_bytes or not raw_bytes.strip():
                st.sidebar.warning(f"File `{filename}` is empty. Skipped.")
                continue

            doc = load_document_from_bytes(raw_bytes, filename)
        except ValueError as e:
            st.sidebar.warning(f"File `{filename}` is empty. Skipped.")
            continue
        except Exception as e:
            st.sidebar.error(f"Failed to parse `{filename}`: {str(e)}")
            continue

        # 3. Clean and chunk using backend pipeline
        try:
            cleaned = clean_document(doc)
            chunks = chunk_document(cleaned)

            if not chunks:
                st.sidebar.warning(f"No valid text chunks generated from `{filename}`.")
                continue

            # 4. Store in ChromaDB with intelligent content-change detection
            newly_indexed = target_store.index_chunks(chunks, overwrite=False)
            if newly_indexed > 0:
                indexed_total += newly_indexed
                st.sidebar.success(f"Indexed {newly_indexed} chunks from `{filename}`.")
                # Synchronize BM25 index with ChromaDB to keep hybrid retrieval consistent
                try:
                    from src.hybrid_retriever import get_hybrid_retriever
                    get_hybrid_retriever().sync_bm25()
                except Exception:
                    pass
            else:
                st.sidebar.info(f"`{filename}` already indexed.")

        except Exception as e:
            st.sidebar.error(f"Error indexing `{filename}`: {str(e)}")

    if indexed_total > 0:
        st.sidebar.success(f"Indexing complete: {indexed_total} new chunks added.")
        st.rerun()


def handle_direct_text_input(
    text: str,
    title: str = "quick_notes.txt",
    store: Optional[VectorStore] = None,
) -> None:
    """Clean, chunk, and index directly pasted or typed text into ChromaDB.

    Args:
        text: Raw text string provided by user.
        title: Target document identifier.
        store: Optional VectorStore instance.
    """
    cleaned_input = (text or "").strip()
    if not cleaned_input:
        st.sidebar.warning("Note content cannot be empty.")
        return

    safe_title = (title or "quick_notes.txt").strip()
    if not safe_title.endswith((".txt", ".md")):
        safe_title = f"{safe_title}.txt"

    target_store = store or get_vector_store()
    try:
        doc = create_document_from_text(cleaned_input, safe_title)
        cleaned = clean_document(doc)
        chunks = chunk_document(cleaned)
        if not chunks:
            st.sidebar.warning("No valid text chunks generated.")
            return

        newly_indexed = target_store.index_chunks(chunks, overwrite=True)
        if newly_indexed > 0:
            try:
                from src.hybrid_retriever import get_hybrid_retriever
                get_hybrid_retriever().sync_bm25()
            except Exception:
                pass
            st.sidebar.success(f"Indexed note '{safe_title}' ({newly_indexed} chunks).")
            st.rerun()
        else:
            st.sidebar.info("Note already indexed.")
    except Exception as e:
        st.sidebar.error(f"Failed to index note: {str(e)}")


def delete_document_action(source: str, store: Optional[VectorStore] = None) -> None:
    """Delete a document from ChromaDB and resynchronize the BM25 index.

    Args:
        source: Name of the document source to remove.
        store: Optional VectorStore instance.
    """
    target_store = store or get_vector_store()
    removed_count = target_store.delete_document(source)
    try:
        from src.hybrid_retriever import get_hybrid_retriever
        get_hybrid_retriever().sync_bm25()
    except Exception:
        pass
    st.sidebar.success(f"Removed '{source}' ({removed_count} chunks deleted).")
    st.rerun()


def render_sidebar(store: Optional[VectorStore] = None) -> None:
    """Render the sidebar with controls, upload interface, and document management.

    Args:
        store: Optional VectorStore instance for dependency injection.
    """
    target_store = store or get_vector_store()

    with st.sidebar:
        # Branding
        st.markdown(
            """
            <div class="sidebar-brand-box">
                <div class="sidebar-brand-title">ALETHIA</div>
                <div class="sidebar-brand-sub">Self-Correcting Precision RAG</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Chat session controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button("New Chat", use_container_width=True, type="secondary"):
                st.session_state.messages = []
                st.rerun()

        with col2:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # Document Ingestion Section
        st.markdown('<div class="sidebar-section-title">Documents & Ingestion</div>', unsafe_allow_html=True)

        tab_upload, tab_paste = st.tabs(["Upload Files", "Paste Text"])

        with tab_upload:
            uploaded_files = st.file_uploader(
                "Upload documents",
                type=["pdf", "txt"],
                accept_multiple_files=True,
                help="Supported file types: PDF documents (.pdf) and Plain text (.txt).",
                label_visibility="collapsed",
            )
            st.caption("Supported formats: `.pdf`, `.txt`")

            if uploaded_files:
                if st.button("Index Files", use_container_width=True, type="primary"):
                    with st.spinner("Processing & indexing documents..."):
                        handle_document_upload(uploaded_files, store=target_store)

        with tab_paste:
            note_title = st.text_input("Document Name", value="meeting_notes.txt", help="Identifier for this note")
            note_body = st.text_area("Paste or draft text", placeholder="Paste document text or notes here...", height=120)
            if st.button("Index Text Note", use_container_width=True):
                if note_body:
                    with st.spinner("Indexing text note..."):
                        handle_direct_text_input(note_body, note_title, store=target_store)
                else:
                    st.warning("Please paste or type text first.")

        # Live document list & management
        stats = get_database_stats(target_store)
        doc_details = stats.get("doc_details", {})

        if stats["sources"]:
            with st.expander(f"Managed Documents ({len(stats['sources'])})", expanded=True):
                for src in stats["sources"]:
                    info = doc_details.get(src, {"chunks": 0, "size_kb": 0.1, "type": "TXT", "status": "Indexed"})
                    ext = info.get("type", "TXT")
                    col_info, col_del = st.columns([4, 1])
                    with col_info:
                        st.markdown(
                            f"""
                            <div class="sidebar-doc-row">
                                <div class="sidebar-doc-name">
                                    <span class="file-type-pill">{ext}</span> {src}
                                </div>
                                <div class="sidebar-doc-meta">
                                    {info.get('size_kb', 0.1)} KB · {info.get('chunks', 0)} chunks
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with col_del:
                        if st.button("✕", key=f"del_{src}", help=f"Remove '{src}'"):
                            delete_document_action(src, store=target_store)

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # Knowledge Base Summary Section
        st.markdown('<div class="sidebar-section-title">Knowledge Base</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="kb-metrics-card">
                <div class="kb-metric-row">
                    <span>Indexed Documents</span>
                    <strong>{stats['documents']}</strong>
                </div>
                <div class="kb-metric-row">
                    <span>Total Passages</span>
                    <strong>{stats['chunks']}</strong>
                </div>
                <div class="kb-metric-row">
                    <span>Index Status</span>
                    <strong style="color: #10B981;">Online</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # About & Capacity Details
        with st.expander("System Specs & Capacity", expanded=False):
            st.markdown(
                """
                **Ingestion Capacity**
                - File types: `.pdf`, `.txt`, direct text
                - File size: Up to 25 MB per document
                - Length: 100,000+ words per file
                - Chunking: Semantic 500-char windows (100-char overlap)

                **Hybrid Engine**
                - Dense: ChromaDB (`all-MiniLM-L6-v2`)
                - Sparse: Pure-Python Okapi BM25
                - Fusion: Reciprocal Rank Fusion ($k=60$)
                - LLM: Groq Llama 3 70B Versatile
                """
            )

        # System API status check
        configured = is_groq_configured()
        if not configured:
            st.warning("`GROQ_API_KEY` is not configured in `.env`.")

