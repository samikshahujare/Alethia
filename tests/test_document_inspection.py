"""Tests for the document inspection and extraction module."""

from unittest.mock import MagicMock
import pytest

from src.document_inspection import (
    detect_document_intent,
    resolve_target_document,
    extract_questions_from_chunks,
    generate_document_overview,
    extract_document_list,
)


def test_detect_document_intent_question_extraction():
    """Verify various question extraction queries are classified correctly."""
    assert detect_document_intent("Which questions are present in the file?") == "question_extraction"
    assert detect_document_intent("List all interview questions in the document") == "question_extraction"
    assert detect_document_intent("Extract every question from the uploaded file") == "question_extraction"
    assert detect_document_intent("What questions are in this document?") == "question_extraction"


def test_detect_document_intent_summary():
    """Verify document overview and summarization queries are classified correctly."""
    assert detect_document_intent("What is this document about?") == "document_summary"
    assert detect_document_intent("Summarize this document") == "document_summary"
    assert detect_document_intent("Give an overview of the file") == "document_summary"
    assert detect_document_intent("Summary of the uploaded document") == "document_summary"


def test_detect_document_intent_list_extraction():
    """Verify list extraction queries for skills, topics, and projects."""
    assert detect_document_intent("List all skills mentioned in the document") == "list_extraction"
    assert detect_document_intent("What topics are covered in the file?") == "list_extraction"
    assert detect_document_intent("List the projects described in the document") == "list_extraction"


def test_detect_document_intent_none_for_standard_rag():
    """Verify standard factual and concept queries return None."""
    assert detect_document_intent("What is machine learning?") is None
    assert detect_document_intent("What is KYC?") is None
    assert detect_document_intent("Explain precision, recall and F1-score") is None
    assert detect_document_intent("What is the capital city of Australia?") is None


def test_resolve_target_document():
    """Verify resolution of target document from available sources."""
    sources = ["solar_system.txt", "transformer_summary.txt", "Tell me about yourself.txt"]

    # Explicit mention in query
    assert resolve_target_document("What is in solar_system.txt?", sources) == "solar_system.txt"
    assert resolve_target_document("Questions in Tell me about yourself?", sources) == "Tell me about yourself.txt"

    # User uploaded document prioritized when no document named
    assert resolve_target_document("Which questions are present in the file?", sources) == "Tell me about yourself.txt"


def test_extract_questions_from_chunks():
    """Verify question extraction preserves wording, document order, and deduplicates."""
    chunks = [
        {
            "chunk_id": "chunk_01",
            "source": "interview.txt",
            "chunk_index": 0,
            "text": "Tell me about yourself:\nI am a data analyst.\n\nWhy should we hire you? Because I work hard.",
        },
        {
            "chunk_id": "chunk_02",
            "source": "interview.txt",
            "chunk_index": 1,
            "text": "What is KYC:\nKYC stands for Know Your Customer.\n\nWhy should we hire you? Because I work hard.",
        },
        {
            "chunk_id": "chunk_03",
            "source": "interview.txt",
            "chunk_index": 2,
            "text": "What all can you do in Excel? I can use VLOOKUP and pivot tables.",
        },
    ]

    res = extract_questions_from_chunks(chunks, "interview.txt")
    assert res["source"] == "interview.txt"
    assert res["total_count"] == 4
    # Check deduplication and ordering
    assert res["questions"] == [
        "Tell me about yourself:",
        "Why should we hire you?",
        "What is KYC:",
        "What all can you do in Excel?",
    ]
    assert "Questions present in **interview.txt** (4 total):" in res["answer"]


def test_extract_questions_empty_when_no_questions():
    """Verify clean abstention message when document contains no questions."""
    chunks = [
        {
            "chunk_id": "chunk_01",
            "source": "notes.txt",
            "chunk_index": 0,
            "text": "The transformer architecture relies entirely on attention mechanisms.",
        }
    ]
    res = extract_questions_from_chunks(chunks, "notes.txt")
    assert res["total_count"] == 0
    assert "No explicit questions were found in **notes.txt**" in res["answer"]


def test_generate_document_overview_extractive():
    """Verify document overview generator produces clear summary."""
    chunks = [
        {
            "chunk_id": "chunk_01",
            "source": "notes.txt",
            "chunk_index": 0,
            "text": "Overview of neural networks and gradient descent.",
        }
    ]
    summary = generate_document_overview(chunks, "notes.txt")
    assert "Summary of **notes.txt**" in summary
    assert "Overview of neural networks" in summary

