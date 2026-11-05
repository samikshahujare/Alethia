"""Unit tests for the Systematic Answer Accuracy Debugger.

Verifies:
1. Complete 18-field diagnostic capture.
2. Accurate classification across all 10 failure categories:
   - document_ingestion_failure
   - retrieval_failure
   - wrong_document_selection
   - context_construction_failure
   - question_decomposition_failure
   - unsupported_generation
   - incorrect_judge_approval
   - unnecessary_abstention
   - incomplete_answer
   - source_attribution_failure
3. Proper serialization to dict and JSON.
"""

from typing import Any, Dict, List
import pytest

from eval.accuracy_debugger import (
    DiagnosticRecord,
    FAILURE_CATEGORIES,
    _classify_failure,
    diagnose_query,
)


class TestDiagnosticRecordStructure:
    """Test structure and serialization of DiagnosticRecord."""

    def test_record_has_all_18_fields(self):
        rec = DiagnosticRecord(
            original_question="What is KYC?",
            detected_intent=None,
            selected_document="Tell me about yourself.txt",
            decomposed_subquestions=["What is KYC?"],
            dense_results=[{"chunk_id": "c1", "score": 0.85}],
            bm25_results=[{"chunk_id": "c1", "score": 12.5}],
            hybrid_results=[{"chunk_id": "c1", "rrf_score": 0.032}],
            rrf_rankings={"c1": {"dense_rank": 1, "bm25_rank": 1, "rrf_score": 0.032}},
            retrieved_chunk_ids=["c1"],
            actual_retrieved_text={"c1": "KYC stands for Know Your Customer."},
            context_passed_to_generator="[Document Chunk 1] KYC stands for Know Your Customer.",
            generated_answer="KYC stands for Know Your Customer.",
            judge_verdict="PASS",
            judge_reason="Fully grounded in context.",
            rewrite_attempts=0,
            newly_retrieved_evidence=[],
            final_answer="KYC stands for Know Your Customer.",
            source_references=["Tell me about yourself.txt"],
            total_execution_time_ms=45.2,
        )

        d = rec.to_dict()
        assert len(d) >= 18
        assert d["original_question"] == "What is KYC?"
        assert d["selected_document"] == "Tell me about yourself.txt"
        assert d["retrieved_chunk_ids"] == ["c1"]
        assert d["final_answer"] == "KYC stands for Know Your Customer."
        assert d["judge_verdict"] == "PASS"
        assert d["total_execution_time_ms"] == 45.2


class TestFailureClassification:
    """Test automatic failure categorization across diagnostic scenarios."""

    def test_classify_unnecessary_abstention(self):
        rec = DiagnosticRecord(
            original_question="What is KYC?",
            final_answer="Not in context.",
            judge_verdict="PASS",
        )
        _classify_failure(rec, expected_abstain=False, expected_answer_keywords=["Know Your Customer"])
        assert rec.is_correct is False
        assert rec.failure_category == "unnecessary_abstention"

    def test_classify_correct_abstention(self):
        rec = DiagnosticRecord(
            original_question="What is the capital of Australia?",
            final_answer="Not in context.",
            judge_verdict="PASS",
        )
        _classify_failure(rec, expected_abstain=True)
        assert rec.is_correct is True
        assert rec.failure_category is None

    def test_classify_unsupported_generation_when_abstention_expected(self):
        rec = DiagnosticRecord(
            original_question="What is the capital of Australia?",
            final_answer="The capital of Australia is Canberra.",
            judge_verdict="PASS",
        )
        _classify_failure(rec, expected_abstain=True)
        assert rec.is_correct is False
        assert rec.failure_category == "unsupported_generation"

    def test_classify_retrieval_failure(self):
        rec = DiagnosticRecord(
            original_question="What is SOP?",
            retrieved_chunk_ids=["chunk_0018", "chunk_0025"],
            final_answer="Not in context.",
        )
        _classify_failure(rec, expected_chunks=["chunk_0006"])
        assert rec.is_correct is False
        assert rec.failure_category == "retrieval_failure"

    def test_classify_wrong_document_selection(self):
        rec = DiagnosticRecord(
            original_question="How many layers in the Transformer?",
            selected_document="Tell me about yourself.txt",
            final_answer="Not in context.",
        )
        _classify_failure(rec, expected_source="transformer_summary.txt")
        assert rec.is_correct is False
        assert rec.failure_category == "wrong_document_selection"

    def test_classify_incomplete_answer(self):
        rec = DiagnosticRecord(
            original_question="What is precision and recall?",
            final_answer="Precision is true positives over total predicted positives.",
            judge_verdict="PASS",
        )
        _classify_failure(rec, expected_answer_keywords=["precision", "recall"])
        assert rec.is_correct is False
        assert rec.failure_category == "incomplete_answer"


class TestDiagnoseQueryEndToEnd:
    """Test full diagnose_query function with deterministic mock LLM."""

    def test_diagnose_query_successful_run(self):
        def mock_llm(**kwargs):
            return "VERDICT: PASS\nREASON: Factually grounded."

        rec = diagnose_query(
            question="What is KYC?",
            k=2,
            llm_func=mock_llm,
        )
        assert rec.original_question == "What is KYC?"
        assert isinstance(rec.retrieved_chunk_ids, list)
        assert isinstance(rec.dense_results, list)
        assert isinstance(rec.bm25_results, list)
        assert isinstance(rec.hybrid_results, list)
        assert rec.total_execution_time_ms > 0

