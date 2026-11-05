"""Source references presentation component for Alethia UI.

Presents clean, readable excerpt cards with source document names,
section indexes, and high-contrast typography.
"""

from typing import Any, Dict, List, Optional
import streamlit as st


def render_sources(chunks: Optional[List[Dict[str, Any]]]) -> None:
    """Render an expandable source citations section for retrieved chunks.

    Args:
        chunks: List of retrieved chunk dictionaries from state["chunks"].
    """
    if not chunks:
        with st.expander("📄 Retrieved Sources (0)", expanded=False):
            st.info("No relevant document sources were retrieved for this question.")
        return

    num_sources = len(chunks)
    label = f"📄 Retrieved Sources ({num_sources})"

    with st.expander(label, expanded=False):
        # If document inspection retrieved many chunks, limit preview or show cleanly
        display_chunks = chunks if num_sources <= 10 else chunks[:10]

        for idx, chunk in enumerate(display_chunks, 1):
            source = chunk.get("source", "Unknown Document")
            chunk_id = chunk.get("chunk_id", f"chunk_{idx}")
            text = chunk.get("text", "").strip()
            sim = chunk.get("similarity") or chunk.get("dense_similarity")
            score_info = f" · Similarity: {sim}" if sim is not None else ""

            st.markdown(
                f"""<div class="source-card">
<div class="source-meta">
<span><strong>Document:</strong> {source} · <code class="mono">{chunk_id}</code>{score_info}</span>
</div>
<div class="source-text">{text}</div>
</div>""",
                unsafe_allow_html=True,
            )

        if num_sources > 10:
            st.caption(f"*Showing 10 of {num_sources} referenced sections.*")
