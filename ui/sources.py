"""Source references presentation component for Alethia UI.

Presents clean, readable excerpt cards with source document names,
section indexes, and high-contrast typography.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import streamlit as st


def render_sources(chunks: Optional[List[Dict[str, Any]]]) -> None:
    """Render an expandable source citations section for retrieved chunks.

    Args:
        chunks: List of retrieved chunk dictionaries from state["chunks"].
    """
    if not chunks:
        with st.expander("Retrieved Sources (0)", expanded=False):
            st.info("No relevant document sources were retrieved for this question.")
        return

    num_sources = len(chunks)
    label = f"Retrieved Sources ({num_sources})"

    with st.expander(label, expanded=False):
        # If document inspection retrieved many chunks, limit preview or show cleanly
        display_chunks = chunks if num_sources <= 10 else chunks[:10]

        for idx, chunk in enumerate(display_chunks, 1):
            source = chunk.get("source", "Unknown Document")
            chunk_id = chunk.get("chunk_id", f"chunk_{idx}")
            text = chunk.get("text", "").strip()

            ext = Path(source).suffix.replace(".", "").upper() or "DOC"

            sim = chunk.get("similarity") or chunk.get("dense_similarity")
            if sim is not None:
                sim_float = float(sim)
                if 0.0 <= sim_float <= 1.0:
                    sim_pct = round(sim_float * 100, 1)
                    score_badge = f'<span class="source-score-pill">Match: {sim_pct}%</span>'
                else:
                    score_badge = f'<span class="source-score-pill">Score: {sim_float:.2f}</span>'
            else:
                score_badge = ""

            st.markdown(
                f"""<div class="source-card">
<div class="source-meta">
    <div style="display: flex; align-items: center; gap: 0.5rem;">
        <span class="file-type-pill">{ext}</span>
        <strong style="color: #E2E8F0;">{source}</strong>
        <span style="color: #64748B;">·</span>
        <code class="mono" style="color: #94A3B8;">{chunk_id}</code>
    </div>
    {score_badge}
</div>
<div class="source-text">{text}</div>
</div>""",
                unsafe_allow_html=True,
            )

        if num_sources > 10:
            st.caption(f"Showing top 10 of {num_sources} referenced sections.")
