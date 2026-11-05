"""State definitions for the Alethia self-correcting RAG engine.

Defines the core data contract passed between retrieval, generation, judging,
and rewriting nodes. Compatible with LangGraph StateGraph execution.
"""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class GraphState(TypedDict, total=False):
    """Shared state dictionary representing the RAG workflow pipeline state."""

    question: str
    original_question: str
    chunks: List[Dict[str, Any]]
    draft: str
    verdict: str  # "PASS" or "FAIL"
    reason: str
    attempts: int
    trace: List[Dict[str, Any]]
    decomposed_questions: List[str]
    subquestion_results: Dict[str, List[Dict[str, Any]]]
    retrieval_strategy: str
    retrieval_diagnostics: Dict[str, Any]
    query_history: List[str]
    current_rewrite: str
    initial_chunk_ids: List[str]
    subsequent_chunk_ids: List[str]
    newly_retrieved_chunk_ids: List[str]
    newly_retrieved_relevant_chunks: List[Dict[str, Any]]
    total_latency_ms: float
    document_intent: Optional[str]
    target_document: Optional[str]
    extracted_questions: List[str]


def create_initial_state(question: str) -> GraphState:
    """Create a standardized initial GraphState for a user query.

    Args:
        question: The initial question submitted by the user.

    Returns:
        GraphState initialized with default empty values.
    """
    clean_q = question.strip() if question else ""
    return {
        "question": clean_q,
        "original_question": clean_q,
        "chunks": [],
        "draft": "",
        "verdict": "",
        "reason": "",
        "attempts": 0,
        "trace": [],
        "decomposed_questions": [],
        "subquestion_results": {},
        "retrieval_strategy": "hybrid",
        "retrieval_diagnostics": {},
        "query_history": [clean_q] if clean_q else [],
        "current_rewrite": "",
        "initial_chunk_ids": [],
        "subsequent_chunk_ids": [],
        "newly_retrieved_chunk_ids": [],
        "newly_retrieved_relevant_chunks": [],
        "total_latency_ms": 0.0,
        "document_intent": None,
        "target_document": None,
        "extracted_questions": [],
    }
