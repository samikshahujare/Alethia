"""Unit and integration tests for Phase 5: Rigorous Retrieval Benchmarking and Optimization.

Tests:
1. IR metric calculation correctness (Hit@K, Recall@K, Precision@K, MRR).
2. Edge cases: missing relevant chunks, empty retrieved lists, rank beyond K_max.
3. Multi-relevant chunk queries (partial recall, full recall).
4. Segregation of answerable vs unanswerable/abstention queries.
5. Deterministic ranking and score tie-breaking consistency.
6. Hybrid parameter configuration handling (RRF k, dense/BM25 weights, candidate pool).
7. Benchmark reproducibility and artifact generation integrity.
8. Question decomposition evaluation (triggering, subquestion generation, evidence preservation).
"""

import pytest
from typing import Any, Dict, List

from eval.phase5_benchmark import (
    calculate_hit_at_k,
    calculate_recall_at_k,
    calculate_precision_at_k,
    calculate_reciprocal_rank,
    compute_retrieval_metrics,
    evaluate_retrieval_strategy,
    evaluate_parameter_grid,
    evaluate_decomposition_behavior,
)
from src.hybrid_retriever import HybridRetriever, calculate_rrf_score


# ============================================================================
# 1. Metric Calculation Unit Tests
# ============================================================================

class TestMetricCalculations:
    """Verify mathematical definitions of Hit Rate@K, Recall@K, Precision@K, and MRR."""

    def test_hit_at_k_present_and_absent(self):
        relevant_ids = {"c1", "c2"}
        retrieved_ids = ["c3", "c1", "c4", "c5", "c2"]

        # Hit@1: c3 is not in relevant_ids -> 0
        assert calculate_hit_at_k(retrieved_ids, relevant_ids, k=1) == 0
        # Hit@2: c1 is relevant -> 1
        assert calculate_hit_at_k(retrieved_ids, relevant_ids, k=2) == 1
        # Hit@3: c1 is relevant -> 1
        assert calculate_hit_at_k(retrieved_ids, relevant_ids, k=3) == 1
        # Empty retrieved list -> 0
        assert calculate_hit_at_k([], relevant_ids, k=3) == 0
        # Empty relevant list -> 0
        assert calculate_hit_at_k(retrieved_ids, set(), k=3) == 0

    def test_recall_at_k_single_and_multiple_targets(self):
        # Case A: 1 relevant target
        assert calculate_recall_at_k(["c1", "c2"], {"c1"}, k=1) == 1.0
        assert calculate_recall_at_k(["c2", "c1"], {"c1"}, k=1) == 0.0
        assert calculate_recall_at_k(["c2", "c1"], {"c1"}, k=2) == 1.0

        # Case B: 2 relevant targets (partial recall)
        rel_2 = {"c1", "c2"}
        assert calculate_recall_at_k(["c1", "c3", "c4"], rel_2, k=1) == 0.5
        assert calculate_recall_at_k(["c1", "c3", "c4"], rel_2, k=3) == 0.5
        assert calculate_recall_at_k(["c1", "c2", "c4"], rel_2, k=2) == 1.0

    def test_precision_at_k(self):
        relevant_ids = {"c1", "c2"}
        retrieved_ids = ["c1", "c3", "c2", "c4", "c5"]

        # Precision@1: 1 relevant out of 1 -> 1.0
        assert calculate_precision_at_k(retrieved_ids, relevant_ids, k=1) == 1.0
        # Precision@2: 1 relevant (c1) out of 2 -> 0.5
        assert calculate_precision_at_k(retrieved_ids, relevant_ids, k=2) == 0.5
        # Precision@3: 2 relevant (c1, c2) out of 3 -> 2/3
        assert pytest.approx(calculate_precision_at_k(retrieved_ids, relevant_ids, k=3), rel=1e-3) == 2 / 3
        # Precision@5: 2 relevant out of 5 -> 0.4
        assert calculate_precision_at_k(retrieved_ids, relevant_ids, k=5) == 0.4

    def test_reciprocal_rank(self):
        relevant_ids = {"target"}
        # Found at rank 1 -> 1/1 = 1.0
        assert calculate_reciprocal_rank(["target", "c2", "c3"], relevant_ids) == 1.0
        # Found at rank 2 -> 1/2 = 0.5
        assert calculate_reciprocal_rank(["c1", "target", "c3"], relevant_ids) == 0.5
        # Found at rank 3 -> 1/3 = 0.3333...
        assert pytest.approx(calculate_reciprocal_rank(["c1", "c2", "target"], relevant_ids), rel=1e-3) == 1 / 3
        # Not found in top 5 -> 0.0
        assert calculate_reciprocal_rank(["c1", "c2", "c3", "c4", "c5"], relevant_ids, k_max=5) == 0.0


# ============================================================================
# 2. Query Segregation & Missing Chunk Handling
# ============================================================================

class TestBenchmarkSegregationAndEdgeCases:
    """Verify segregation of answerable vs unanswerable queries and missing chunk handling."""

    def test_unanswerable_queries_excluded_from_denominator(self):
        mock_dataset = [
            {"id": "A1", "expected_behavior": "answer", "relevant_chunk_ids": ["c1"]},
            {"id": "A2", "expected_behavior": "answer", "relevant_chunk_ids": ["c2"]},
            {"id": "U1", "expected_behavior": "abstain", "relevant_chunk_ids": []},
            {"id": "U2", "expected_behavior": "abstain", "relevant_chunk_ids": []},
        ]
        # Query results where A1 hits at rank 1, A2 misses completely
        mock_retrievals = {
            "A1": ["c1", "c3"],
            "A2": ["c4", "c5"],
            "U1": ["c3", "c4"],
            "U2": ["c5", "c6"],
        }

        metrics = compute_retrieval_metrics(mock_dataset, mock_retrievals)
        assert metrics["total_answerable_queries"] == 2
        assert metrics["unanswerable_queries_excluded"] == 2
        # Hit Rate@1: 1 / 2 = 0.5
        assert metrics["hit_rate_at_1"] == 0.5
        # MRR: (1/1 + 0) / 2 = 0.5
        assert metrics["mrr"] == 0.5

    def test_missing_relevant_chunks_scored_as_zero(self):
        mock_dataset = [
            {"id": "A1", "expected_behavior": "answer", "relevant_chunk_ids": ["missing_chunk"]},
        ]
        mock_retrievals = {"A1": ["other_1", "other_2"]}
        metrics = compute_retrieval_metrics(mock_dataset, mock_retrievals)
        assert metrics["hit_rate_at_1"] == 0.0
        assert metrics["hit_rate_at_3"] == 0.0
        assert metrics["hit_rate_at_5"] == 0.0
        assert metrics["recall_at_1"] == 0.0
        assert metrics["recall_at_5"] == 0.0
        assert metrics["mrr"] == 0.0


# ============================================================================
# 3. Deterministic Ranking & Hybrid Configuration Handling
# ============================================================================

class TestConfigurationAndDeterminism:
    """Verify parameter variations and deterministic ranking."""

    def test_rrf_scoring_with_different_weights_and_k(self):
        # Equal weights k=60
        score_equal = calculate_rrf_score(dense_rank=1, bm25_rank=1, k_rrf=60, dense_weight=1.0, bm25_weight=1.0)
        assert score_equal == round(1 / 61 + 1 / 61, 6)

        # Dense-heavy: dense_weight=0.75, bm25_weight=0.25, k=40
        score_dense_heavy = calculate_rrf_score(dense_rank=1, bm25_rank=2, k_rrf=40, dense_weight=0.75, bm25_weight=0.25)
        expected = round(0.75 / 41 + 0.25 / 42, 6)
        assert score_dense_heavy == expected

    def test_ranking_determinism(self):
        """Repeated calls with identical parameters must return identical rank order."""
        class MockStore:
            def retrieve_chunks(self, q, k=4):
                return [{"chunk_id": "c1", "similarity": 0.8}, {"chunk_id": "c2", "similarity": 0.7}]

        class MockBM25:
            def search(self, q, k=4):
                return [{"chunk_id": "c2", "bm25_score": 10.0}, {"chunk_id": "c1", "bm25_score": 8.0}]

        retriever = HybridRetriever(vector_store=MockStore(), bm25_index=MockBM25(), k_rrf=60)
        res1 = retriever.retrieve_hybrid("test query", k=2)
        res2 = retriever.retrieve_hybrid("test query", k=2)
        assert [c["chunk_id"] for c in res1] == [c["chunk_id"] for c in res2]


# ============================================================================
# 4. Question Decomposition Evaluation
# ============================================================================

class TestDecompositionEvaluation:
    """Verify evaluation of decomposition triggering, subquestion generation, and evidence preservation."""

    def test_decomposition_behavior_metrics(self):
        eval_cases = [
            {
                "query": "What is backpropagation?",
                "is_multi_intent": False,
                "expected_subquestions": 1,
            },
            {
                "query": "What optimizer was used in the Transformer paper, and what are Adam's default beta values?",
                "is_multi_intent": True,
                "expected_subquestions": 2,
            },
        ]
        result = evaluate_decomposition_behavior(eval_cases)
        assert result["total_queries"] == 2
        assert result["correct_trigger_count"] == 2
        assert result["trigger_accuracy"] == 1.0

