"""Sidebar component for Alethia UI.

Handles application branding, chat session controls, document upload & indexing,
per-document management (viewing size, status, and deletion), and knowledge base statistics.
"""

from typing import Any, Dict, List, Optional
import streamlit as st

from src.chunker import chunk_document
from src.config import is_groq_configured
from src.document_loader import Document
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
                    doc_details[src] = {"chunks": 0, "chars": 0, "status": "Indexed"}
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
    """Validate, clean, chunk, and index uploaded .txt files into ChromaDB.

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

        # 1. Validate file extension (.txt only)
        if not filename.lower().endswith(".txt"):
            st.sidebar.error(f"Unsupported format: `{filename}`. Only `.txt` files are supported.")
            continue

        # 2. Read and validate content (with UTF-8 and UTF-8-sig fallback)
        try:
            raw_bytes = file.read()
            if not raw_bytes or not raw_bytes.strip():
                st.sidebar.warning(f"File `{filename}` is empty. Skipped.")
                continue

            try:
                content = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                content = raw_bytes.decode("utf-8-sig")
        except Exception as e:
            st.sidebar.error(f"Failed to read `{filename}`: {str(e)}")
            continue

        # 3. Clean and chunk using established backend pipeline
        try:
            doc = Document(text=content, metadata={"source": filename})
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
                st.sidebar.info(f"`{filename}` already indexed (no changes detected).")

        except Exception as e:
            st.sidebar.error(f"Error indexing `{filename}`: {str(e)}")

    if indexed_total > 0:
        st.sidebar.success(f"Indexing complete: {indexed_total} new chunks added to database.")
        st.rerun()


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
            <div style="padding-bottom: 0.5rem; margin-bottom: 1rem; border-bottom: 1px solid #E2E8F0;">
                <h2 style="margin: 0; font-size: 1.35rem; font-weight: 700; color: #172033;">⚡ Alethia</h2>
                <p style="margin: 0.2rem 0 0 0; font-size: 0.8rem; color: #64748B;">Document AI Assistant</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Chat session controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ New Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        st.markdown("---")

        # 1. Your Documents Section
        st.markdown("#### 📁 Your Documents")
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["txt"],
            accept_multiple_files=True,
            help="Supported file types: Plain text (.txt, UTF-8).",
            label_visibility="collapsed",
        )
        st.caption("Supported formats: `.txt` (UTF-8)")

        if uploaded_files:
            if st.button("Index Uploaded Files", use_container_width=True, type="primary"):
                with st.spinner("Processing & indexing documents..."):
                    handle_document_upload(uploaded_files, store=target_store)

        # Retrieve live document statistics
        stats = get_database_stats(target_store)
        doc_details = stats.get("doc_details", {})

        if stats["sources"]:
            with st.expander(f"Manage Documents ({len(stats['sources'])})", expanded=True):
                for src in stats["sources"]:
                    info = doc_details.get(src, {"chunks": 0, "size_kb": 0.0, "status": "Indexed"})
                    col_info, col_del = st.columns([4, 1])
                    with col_info:
                        st.markdown(
                            f"""
                            <div style="font-size: 0.825rem; font-weight: 600; color: #172033; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                                📄 {src}
                            </div>
                            <div style="font-size: 0.725rem; color: #64748B;">
                                {info.get('size_kb', 0.1)} KB · {info.get('chunks', 0)} chunks · <span style="color: #15803D;">● Indexed</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with col_del:
                        if st.button("🗑️", key=f"del_{src}", help=f"Remove '{src}' from knowledge base"):
                            delete_document_action(src, store=target_store)

        st.markdown("---")

        # 2. Knowledge Base Section
        st.markdown("#### 📊 Knowledge Base")
        st.markdown(
            f"""
            <div class="diagnostic-card">
                <strong>Indexed Documents:</strong> {stats['documents']}<br/>
                <strong>Total Chunks:</strong> {stats['chunks']}<br/>
                <strong>Status:</strong> <span style="color: #15803D; font-weight: 600;">● Active</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # System API status
        configured = is_groq_configured()
        if not configured:
            st.warning("⚠️ `GROQ_API_KEY` is not configured in `.env`.")
