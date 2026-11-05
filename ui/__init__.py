"""Alethia UI Package.

Modular Streamlit components for the Alethia Self-Correcting RAG interface.
"""

from ui.styles import apply_custom_styles
from ui.sidebar import render_sidebar
from ui.chat import render_chat_interface, init_session_state
from ui.sources import render_sources
from ui.trace import render_trace

__all__ = [
    "apply_custom_styles",
    "render_sidebar",
    "render_chat_interface",
    "init_session_state",
    "render_sources",
    "render_trace",
]

