"""Alethia Streamlit Application Entrypoint.

Self-Correcting RAG Engine with LangGraph, ChromaDB, Sentence Transformers, and Groq.
"""

import sys
from pathlib import Path
import streamlit as st

# Ensure root directory is on Python path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui.styles import apply_custom_styles
from ui.sidebar import render_sidebar
from ui.chat import render_chat_interface


def main() -> None:
    """Main application layout and execution flow."""
    st.set_page_config(
        page_title="Alethia · Precision RAG",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 1. Apply custom SaaS theme styles
    apply_custom_styles()

    # 2. Render sidebar (controls, document management & database stats)
    render_sidebar()

    # 3. Main header
    st.markdown(
        """
        <div class="alethia-header">
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <div class="alethia-title">
                    Alethia
                </div>
                <div class="header-engine-tag">
                    Self-Correcting RAG
                </div>
            </div>
            <div class="alethia-subtitle">
                Deterministic factual retrieval and verification across multi-format documents.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4. Render interactive chat interface
    render_chat_interface()


if __name__ == "__main__":
    main()
