"""Chat interface and message lifecycle component for Alethia UI.

Presents conversational messages with accessible user-facing status indicators,
collapsible document sources, and expandable diagnostic details.
"""

from typing import Any, Callable, Dict, List, Optional
import streamlit as st

from src.config import is_groq_configured
from src.graph import run_rag_pipeline
from src.llm import ConfigurationError, LLMError, LLMRateLimitError
from ui.sources import render_sources
from ui.trace import render_trace


def init_session_state() -> None:
    """Initialize Streamlit session state variables for chat history and settings."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None


def render_welcome_screen() -> None:
    """Render welcome prompt suggestions when chat history is empty."""
    st.markdown(
        """
        <div class="welcome-card">
            <h3 style="margin: 0 0 0.5rem 0; font-weight: 600; letter-spacing: -0.01em;">Knowledge Base Reasoning</h3>
            <p style="margin: 0 0 1.25rem 0; line-height: 1.6; font-size: 0.9rem;">
                Upload documents (PDF, TXT, or notes) and query any specific claim, comparison, or concept.
                Alethia indexes exact passages and verifies factual consistency against source evidence.
            </p>
            <div style="font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.7; margin-bottom: 0.5rem;">
                Example Prompts
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sample_queries = [
        "Summarize this document.",
        "What are the primary findings?",
        "Extract key questions from this document.",
        "What are the core technical constraints?",
    ]

    cols = st.columns(2)
    for idx, query in enumerate(sample_queries):
        col = cols[idx % 2]
        with col:
            if st.button(query, key=f"sample_{idx}", use_container_width=True):
                st.session_state.pending_query = query
                st.rerun()


def render_assistant_message(msg: Dict[str, Any]) -> None:
    """Render an assistant message card with answer, status badge, sources, and diagnostics.

    Args:
        msg: Assistant message state dictionary from session_state.
    """
    draft = msg.get("content", "")
    verdict = msg.get("verdict", "")
    reason = msg.get("reason", "")
    attempts = msg.get("attempts", 0)
    chunks = msg.get("chunks", [])
    trace = msg.get("trace", [])
    orig_q = msg.get("original_question", "")
    final_q = msg.get("question", orig_q)

    is_abstained = "not in context" in draft.lower() and verdict.upper() == "PASS"

    with st.chat_message("assistant"):
        # Header user-friendly status indicator
        if is_abstained:
            st.markdown(
                '<div class="status-pill status-notfound">Not in context</div>',
                unsafe_allow_html=True,
            )
        elif verdict.upper() == "PASS":
            st.markdown(
                '<div class="status-pill status-verified">Verified</div>',
                unsafe_allow_html=True,
            )
        elif verdict.upper() == "FAIL":
            st.markdown(
                '<div class="status-pill status-partial">Partially answered</div>',
                unsafe_allow_html=True,
            )

        # Answer body
        st.markdown(draft)

        # Verification note if relevant and non-technical
        if reason and not reason.startswith("Verified verbatim"):
            st.caption(f"Verification Note: {reason}")

        # Expandable Sources Section
        render_sources(chunks)

        # Advanced Execution Details (collapsed diagnostics)
        render_trace(
            trace=trace,
            original_question=orig_q,
            final_question=final_q,
            verdict=verdict,
            attempts=attempts,
            draft=draft,
            decomposed_questions=msg.get("decomposed_questions", []),
        )


def process_user_query(
    query: str,
    pipeline_func: Optional[Callable[[str], Dict[str, Any]]] = None,
) -> None:
    """Execute the self-correcting RAG pipeline for a user question and update session state.

    Args:
        query: User question string.
        pipeline_func: Optional callable (default: run_rag_pipeline) for dependency injection.
    """
    clean_query = query.strip() if query else ""
    if not clean_query:
        st.warning("Please enter a non-empty question.")
        return

    # 1. Append user message to history
    st.session_state.messages.append({"role": "user", "content": clean_query})

    # 2. Render user message in UI
    with st.chat_message("user"):
        st.write(clean_query)

    # 3. Check configuration upfront to provide clear guidance
    if not is_groq_configured():
        error_msg = {
            "role": "assistant",
            "content": "GROQ_API_KEY is not configured in `.env`. Please add your API key to proceed.",
            "verdict": "FAIL",
            "reason": "Missing GROQ_API_KEY environment variable.",
            "attempts": 0,
            "chunks": [],
            "trace": [],
            "original_question": clean_query,
            "question": clean_query,
        }
        st.session_state.messages.append(error_msg)
        with st.chat_message("assistant"):
            st.error(error_msg["content"])
        return

    # 4. Execute pipeline with stepped progressive loading indicator
    runner = pipeline_func or run_rag_pipeline
    with st.status("Analyzing question & context...", expanded=True) as status_box:
        st.write("1. Scanning document chunks & search indices...")
        try:
            st.write("2. Identifying relevant passages & citations...")
            final_state = runner(clean_query)
            st.write("3. Synthesizing response & verifying factual consistency...")
            if status_box is not None and hasattr(status_box, "update"):
                status_box.update(label="Response ready", state="complete", expanded=False)
        except ConfigurationError as e:
            if status_box is not None and hasattr(status_box, "update"):
                status_box.update(label="Configuration error", state="error", expanded=False)
            final_state = {
                "question": clean_query,
                "original_question": clean_query,
                "draft": f"Configuration Error: {str(e)}",
                "verdict": "FAIL",
                "reason": "Groq configuration error.",
                "attempts": 0,
                "chunks": [],
                "trace": [],
            }
        except LLMRateLimitError:
            if status_box is not None and hasattr(status_box, "update"):
                status_box.update(label="Rate limit reached", state="error", expanded=False)
            final_state = {
                "question": clean_query,
                "original_question": clean_query,
                "draft": "Rate limit exceeded on Groq API. Please wait a moment and try again.",
                "verdict": "FAIL",
                "reason": "Groq API rate limit (429).",
                "attempts": 0,
                "chunks": [],
                "trace": [],
            }
        except LLMError as e:
            if status_box is not None and hasattr(status_box, "update"):
                status_box.update(label="Inference error", state="error", expanded=False)
            final_state = {
                "question": clean_query,
                "original_question": clean_query,
                "draft": f"API Error: Unable to complete inference ({str(e)}).",
                "verdict": "FAIL",
                "reason": "Groq API error encountered.",
                "attempts": 0,
                "chunks": [],
                "trace": [],
            }
        except Exception as e:
            if status_box is not None and hasattr(status_box, "update"):
                status_box.update(label="Execution error", state="error", expanded=False)
            final_state = {
                "question": clean_query,
                "original_question": clean_query,
                "draft": f"An unexpected error occurred during execution: {str(e)}",
                "verdict": "FAIL",
                "reason": "Internal execution error.",
                "attempts": 0,
                "chunks": [],
                "trace": [],
            }

    # 5. Format and store assistant response
    assistant_msg = {
        "role": "assistant",
        "content": final_state.get("draft", "No answer generated."),
        "verdict": final_state.get("verdict", "FAIL"),
        "reason": final_state.get("reason", ""),
        "attempts": final_state.get("attempts", 0),
        "chunks": final_state.get("chunks", []),
        "trace": final_state.get("trace", []),
        "original_question": final_state.get("original_question", clean_query),
        "question": final_state.get("question", clean_query),
        "decomposed_questions": final_state.get("decomposed_questions", []),
        "retrieval_strategy": final_state.get("retrieval_strategy", "hybrid"),
    }
    st.session_state.messages.append(assistant_msg)

    # 6. Render assistant response immediately
    render_assistant_message(assistant_msg)


def render_chat_interface(
    pipeline_func: Optional[Callable[[str], Dict[str, Any]]] = None,
) -> None:
    """Render the main chat area with historical messages and input bar.

    Args:
        pipeline_func: Optional custom pipeline runner for testing.
    """
    init_session_state()

    # If there are no messages, show welcome card
    if not st.session_state.messages:
        render_welcome_screen()
    else:
        # Render existing conversation history (never re-invoking LLM on reruns)
        for msg in st.session_state.messages:
            if msg.get("role") == "user":
                with st.chat_message("user"):
                    st.write(msg.get("content", ""))
            elif msg.get("role") == "assistant":
                render_assistant_message(msg)

    # Check if a sample query was clicked
    active_query = None
    if st.session_state.pending_query:
        active_query = st.session_state.pending_query
        st.session_state.pending_query = None

    # Chat input bar at bottom
    user_input = st.chat_input("Ask a question about your documents...")
    query_to_run = user_input or active_query

    if query_to_run:
        process_user_query(query_to_run, pipeline_func=pipeline_func)
