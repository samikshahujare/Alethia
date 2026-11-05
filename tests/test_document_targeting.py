"""Tests for explicit document targeting and similarity isolation during retrieval."""

from unittest.mock import MagicMock, patch
import pytest

from src.nodes import retrieve_node
from src.state import GraphState


def test_retrieve_node_targets_referenced_document():
    """Verify that querying about abc.txt retrieves chunks exclusively from abc.txt."""
    abc_chunks = [
        {
            "chunk_id": "abc_c0",
            "source": "abc.txt",
            "text": "The capital of France is Paris.",
            "chunk_index": 0,
            "similarity": 0.89,
            "distance": 0.11,
            "dense_similarity": 0.89,
            "metadata": {"source": "abc.txt"},
        }
    ]

    mock_collection = MagicMock()
    mock_collection.get.return_value = {
        "metadatas": [
            {"source": "abc.txt"},
            {"source": "bcd.txt"},
            {"source": "solar_system.txt"},
        ]
    }

    mock_store = MagicMock()
    mock_store.collection = mock_collection

    mock_hybrid_retrieve = MagicMock(return_value=abc_chunks)

    state: GraphState = {
        "question": "According to abc.txt, what is the capital of France?",
        "original_question": "According to abc.txt, what is the capital of France?",
        "attempts": 0,
        "trace": [],
    }

    with patch("src.store.get_vector_store", return_value=mock_store), \
         patch("src.hybrid_retriever.hybrid_retrieve", mock_hybrid_retrieve):

        updated_state = retrieve_node(state)

    # Verify that hybrid_retrieve was called with source_filter='abc.txt'
    assert mock_hybrid_retrieve.called
    kwargs = mock_hybrid_retrieve.call_args[1]
    assert kwargs.get("source_filter") == "abc.txt"

    # Verify state reflects targeted document
    assert updated_state.get("target_document") == "abc.txt"
    assert len(updated_state.get("chunks", [])) == 1
    assert updated_state["chunks"][0]["source"] == "abc.txt"
    assert updated_state["chunks"][0]["chunk_id"] == "abc_c0"


def test_retrieve_node_global_fallback_when_no_document_named():
    """Verify standard global retrieval occurs when no document is explicitly mentioned."""
    global_chunks = [
        {
            "chunk_id": "solar_0",
            "source": "solar_system.txt",
            "text": "The solar system contains eight planets.",
            "chunk_index": 0,
            "similarity": 0.85,
            "distance": 0.15,
            "metadata": {"source": "solar_system.txt"},
        }
    ]

    mock_collection = MagicMock()
    mock_collection.get.return_value = {
        "metadatas": [
            {"source": "abc.txt"},
            {"source": "bcd.txt"},
            {"source": "solar_system.txt"},
        ]
    }

    mock_store = MagicMock()
    mock_store.collection = mock_collection

    state: GraphState = {
        "question": "How many planets are in the solar system?",
        "original_question": "How many planets are in the solar system?",
        "attempts": 0,
        "trace": [],
    }

    mock_decomp = MagicMock(return_value={
        "chunks": global_chunks,
        "decomposed_questions": ["How many planets are in the solar system?"],
        "subquestion_results": {},
        "retrieval_diagnostics": {},
    })

    with patch("src.store.get_vector_store", return_value=mock_store), \
         patch("src.decomposition.retrieve_decomposed", mock_decomp):

        updated_state = retrieve_node(state)

    assert updated_state.get("target_document") is None
    assert len(updated_state.get("chunks", [])) == 1
    assert updated_state["chunks"][0]["chunk_id"] == "solar_0"

