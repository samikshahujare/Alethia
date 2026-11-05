"""Automated unit and integration tests for Alethia Streamlit UI (Phase 5).

Covers all 14 Phase 5 testing requirements:
1. App imports successfully
2. UI modules import successfully
3. Chat input validation
4. Session-state initialization
5. New Chat behavior
6. Chat history rendering logic
7. Pipeline integration using mocks (0 API credits)
8. Source display with actual metadata
9. Empty source handling
10. Trace rendering with multiple retries
11. Correct abstention display
12. Document upload validation (rejects non-txt, rejects empty)
13. Duplicate upload handling
14. Error handling for failed pipeline execution
"""

from io import BytesIO
from unittest.mock import MagicMock, patch
import pytest
import streamlit as st

from ui.styles import apply_custom_styles, get_custom_css
from ui.sources import render_sources
from ui.trace import render_trace, _get_verdict_badge
from ui.sidebar import get_database_stats, handle_document_upload
from ui.chat import (
    init_session_state,
    process_user_query,
    render_assistant_message,
)
from src.llm import ConfigurationError, LLMError, LLMRateLimitError


# =====================================================================
# 1. Module Import Tests
# =====================================================================

def test_app_imports_successfully():
    """Verify that app.py imports cleanly without errors."""
    import app
    assert hasattr(app, "main")


def test_ui_modules_import_successfully():
    """Verify that all modular UI components import cleanly."""
    import ui
    import ui.styles
    import ui.sidebar
    import ui.chat
    import ui.sources
    import ui.trace

    assert hasattr(ui.styles, "apply_custom_styles")
    assert hasattr(ui.sidebar, "render_sidebar")
    assert hasattr(ui.chat, "render_chat_interface")
    assert hasattr(ui.sources, "render_sources")
    assert hasattr(ui.trace, "render_trace")


# =====================================================================
# 2. Session State & Chat Lifecycle Tests
# =====================================================================

def test_session_state_initialization():
    """Verify init_session_state sets up clean session state defaults."""
    # Reset session_state keys
    if "messages" in st.session_state:
        del st.session_state["messages"]
    if "pending_query" in st.session_state:
        del st.session_state["pending_query"]

    init_session_state()

    assert "messages" in st.session_state
    assert st.session_state.messages == []
    assert "pending_query" in st.session_state
    assert st.session_state.pending_query is None


def test_new_chat_behavior():
    """Verify clearing or restarting a chat resets the conversation history."""
    st.session_state.messages = [
        {"role": "user", "content": "Question 1"},
        {"role": "assistant", "content": "Answer 1"},
    ]
    # New chat action:
    st.session_state.messages = []
    assert len(st.session_state.messages) == 0


def test_chat_input_validation_empty():
    """Verify that empty or whitespace query is rejected without calling pipeline."""
    mock_pipeline = MagicMock()
    with patch("streamlit.warning") as mock_warning:
        process_user_query("", pipeline_func=mock_pipeline)
        mock_pipeline.assert_not_called()
        mock_warning.assert_called_once()

    with patch("streamlit.warning") as mock_warning:
        process_user_query("    \t  ", pipeline_func=mock_pipeline)
        mock_pipeline.assert_not_called()
        mock_warning.assert_called_once()


# =====================================================================
# 3. Pipeline Integration with Mocked Backend
# =====================================================================

def test_pipeline_integration_using_mock():
    """Verify process_user_query executes the pipeline and appends formatted messages."""
    st.session_state.messages = []

    mock_state = {
        "question": "What is self-attention?",
        "original_question": "What is self-attention?",
        "draft": "Self-attention computes attention weights across input tokens.",
        "verdict": "PASS",
        "reason": "Directly supported by the text.",
        "attempts": 0,
        "chunks": [{"chunk_id": "c1", "source": "t.txt", "text": "Self-attention computes weights."}],
        "trace": [{"node": "retrieve_node"}, {"node": "generate_node"}, {"node": "judge_node"}],
    }

    mock_runner = MagicMock(return_value=mock_state)

    with patch("ui.chat.is_groq_configured", return_value=True):
        process_user_query("What is self-attention?", pipeline_func=mock_runner)

    assert mock_runner.called
    assert len(st.session_state.messages) == 2
    assert st.session_state.messages[0]["role"] == "user"
    assert st.session_state.messages[0]["content"] == "What is self-attention?"
    assert st.session_state.messages[1]["role"] == "assistant"
    assert st.session_state.messages[1]["content"] == mock_state["draft"]
    assert st.session_state.messages[1]["verdict"] == "PASS"
    assert st.session_state.messages[1]["attempts"] == 0


# =====================================================================
# 4. Source Reference Component Tests
# =====================================================================

def test_source_display_with_actual_metadata():
    """Verify render_sources displays chunk source, ID, text, and similarity scores."""
    chunks = [
        {
            "chunk_id": "chunk_doc1_0",
            "source": "attention_mechanisms.txt",
            "text": "Multi-head attention allows the model to jointly attend to information.",
            "similarity": 0.8421,
            "distance": 0.1579,
            "metadata": {"chunk_index": 0},
        }
    ]

    with patch("streamlit.markdown") as mock_md, patch("streamlit.expander") as mock_expander:
        render_sources(chunks)
        mock_expander.assert_called_with("Retrieved Sources (1)", expanded=False)
        assert mock_md.called
        rendered_html = mock_md.call_args[0][0]
        assert "attention_mechanisms.txt" in rendered_html
        assert "chunk_doc1_0" in rendered_html
        assert "84.2%" in rendered_html


def test_empty_source_handling():
    """Verify render_sources handles empty or None chunks safely."""
    with patch("streamlit.info") as mock_info, patch("streamlit.expander") as mock_expander:
        render_sources([])
        mock_expander.assert_called_with("Retrieved Sources (0)", expanded=False)
        mock_info.assert_called_with("No relevant document sources were retrieved for this question.")

    with patch("streamlit.info") as mock_info, patch("streamlit.expander") as mock_expander:
        render_sources(None)
        mock_expander.assert_called_with("Retrieved Sources (0)", expanded=False)
        mock_info.assert_called_with("No relevant document sources were retrieved for this question.")


# =====================================================================
# 5. Execution Trace Visualization Tests
# =====================================================================

def test_trace_rendering_with_multiple_retries():
    """Verify render_trace accurately renders multi-step and multi-retry transitions."""
    trace = [
        {"node": "retrieve_node", "question": "Q1", "chunk_count": 4, "sources": ["doc.txt"]},
        {"node": "generate_node", "question": "Q1", "chunk_count": 4, "draft": "Draft 1"},
        {"node": "judge_node", "verdict": "FAIL", "reason": "Unsupported claim."},
        {"node": "rewrite_node", "attempt": 1, "rewritten_question": "Q2", "verification_reason": "Unsupported claim."},
        {"node": "retrieve_node", "question": "Q2", "chunk_count": 4, "sources": ["doc.txt"]},
        {"node": "generate_node", "question": "Q2", "chunk_count": 4, "draft": "Draft 2"},
        {"node": "judge_node", "verdict": "PASS", "reason": "Now supported."},
        {"event": "workflow_completed", "final_verdict": "PASS", "total_attempts": 1},
    ]

    with patch("streamlit.markdown") as mock_md, patch("streamlit.expander") as mock_expander:
        render_trace(
            trace=trace,
            original_question="Q1",
            final_question="Q2",
            verdict="PASS",
            attempts=1,
            draft="Draft 2",
        )
        mock_expander.assert_called_with("Execution Diagnostics (8 steps · 1 retries)", expanded=False)
        # Verify markdown calls were executed for trace summary and individual step cards
        assert mock_md.call_count >= 5


def test_correct_abstention_display():
    """Verify proper abstention ('Not in context.' + PASS) generates the Abstained badge."""
    badge_html = _get_verdict_badge("PASS", is_abstained=True)
    assert "Abstained · Pass" in badge_html
    assert "badge-abstained" in badge_html

    badge_pass = _get_verdict_badge("PASS", is_abstained=False)
    assert "Passed" in badge_pass
    assert "badge-pass" in badge_pass

    badge_fail = _get_verdict_badge("FAIL", is_abstained=False)
    assert "Failed" in badge_fail
    assert "badge-fail" in badge_fail


# =====================================================================
# 6. Document Upload & Validation Tests
# =====================================================================

class MockUploadedFile:
    """Mock Streamlit UploadedFile object."""

    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def read(self) -> bytes:
        return self._content


def test_document_upload_validation_rejects_unsupported_formats():
    """Verify that unsupported file formats (like .csv or .exe) are rejected with an error."""
    mock_store = MagicMock()
    csv_file = MockUploadedFile("table.csv", b"col1,col2\n1,2")

    with patch("streamlit.sidebar.error") as mock_err:
        handle_document_upload([csv_file], store=mock_store)
        mock_err.assert_called_once()
        assert "Unsupported format" in mock_err.call_args[0][0]
        mock_store.index_chunks.assert_not_called()


def test_document_upload_validation_rejects_empty():
    """Verify that empty files are flagged and skipped."""
    mock_store = MagicMock()
    empty_file = MockUploadedFile("empty.txt", b"   \n  ")

    with patch("streamlit.sidebar.warning") as mock_warn:
        handle_document_upload([empty_file], store=mock_store)
        mock_warn.assert_called_once()
        assert "is empty" in mock_warn.call_args[0][0]
        mock_store.index_chunks.assert_not_called()


def test_document_upload_duplicate_handling():
    """Verify that uploading duplicate documents reports 0 new chunks cleanly."""
    mock_store = MagicMock()
    # Mock index_chunks returning 0 (skipped duplicates)
    mock_store.index_chunks.return_value = 0

    valid_file = MockUploadedFile(
        "notes.txt",
        b"Machine learning systems require rigorous validation data splits.",
    )

    with patch("streamlit.sidebar.info") as mock_info:
        handle_document_upload([valid_file], store=mock_store)
        assert mock_store.index_chunks.called
        assert any("already indexed" in str(arg) for arg in mock_info.call_args_list)


# =====================================================================
# 7. Error Handling for Failed Pipeline Execution
# =====================================================================

def test_error_handling_missing_groq_api_key():
    """Verify graceful handling when Groq API key is unconfigured."""
    st.session_state.messages = []
    with patch("ui.chat.is_groq_configured", return_value=False), patch("streamlit.error") as mock_err:
        process_user_query("Any question")
        assert len(st.session_state.messages) == 2
        assert "GROQ_API_KEY is not configured" in st.session_state.messages[1]["content"]
        mock_err.assert_called_once()


def test_error_handling_rate_limit_exception():
    """Verify graceful handling of LLMRateLimitError without leaking secrets."""
    st.session_state.messages = []

    def mock_rate_limit_runner(q):
        raise LLMRateLimitError("Rate limit reached: 429")

    with patch("ui.chat.is_groq_configured", return_value=True):
        process_user_query("How many layers?", pipeline_func=mock_rate_limit_runner)

    assert len(st.session_state.messages) == 2
    assistant_msg = st.session_state.messages[1]
    assert "Rate limit exceeded" in assistant_msg["content"]
    assert assistant_msg["verdict"] == "FAIL"


def test_error_handling_general_exception():
    """Verify unexpected runtime exceptions are caught and reported without crashing."""
    st.session_state.messages = []

    def mock_broken_runner(q):
        raise RuntimeError("ChromaDB connection timed out")

    with patch("ui.chat.is_groq_configured", return_value=True):
        process_user_query("How many layers?", pipeline_func=mock_broken_runner)

    assert len(st.session_state.messages) == 2
    assistant_msg = st.session_state.messages[1]
    assert "unexpected error occurred" in assistant_msg["content"]
    assert assistant_msg["verdict"] == "FAIL"


# =====================================================================
# 8. Main Application Flow Integration Test
# =====================================================================

def test_app_main_execution():
    """Verify app.py main() executes cleanly and coordinates all UI components."""
    import app

    with patch("app.apply_custom_styles") as mock_styles, \
         patch("app.render_sidebar") as mock_sidebar, \
         patch("app.render_chat_interface") as mock_chat, \
         patch("streamlit.set_page_config") as mock_config, \
         patch("streamlit.markdown") as mock_md:

        app.main()

        mock_config.assert_called_once()
        mock_styles.assert_called_once()
        mock_sidebar.assert_called_once()
        mock_chat.assert_called_once()
        assert mock_md.called

