"""Comprehensive unit and integration tests for HybridRetriever with Reciprocal Rank Fusion (RRF).

Tests:
- RRF calculation unit math.
- Dense-only retrieval results.
- BM25-only retrieval results.
- Hybrid RRF ranking correctness.
- Documents present in only one ranking.
- Duplicate chunk removal across sources.
- Complete metadata and diagnostic preservation.
- Empty query and empty corpus handling.
- Deterministic tie-breaking by chunk_id.
- Configurable top-k parameter.
- Correct handling of missing ranks (None).
- Integration test with persistent ChromaDB and synchronized BM25Index.
- Clear distinction between cosine distance and cosine similarity.
"""

import pytest
from unittest.mock import MagicMock
from src.bm25 import BM25Index
from src.hybrid_retriever import (
    HybridRetriever,
    calculate_rrf_score,
    get_hybrid_retriever,
    hybrid_retrieve,
    DEFAULT_RRF_K,
)


class TestRRFCalculationMath:
    """Verifies that RRF calculations match exact mathematical definitions."""

    def test_both_ranks_present(self):
        # Dense rank 1, BM25 rank 2, k=60
        # 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.01639344 + 0.01612903 = 0.032522
        score = calculate_rrf_score(dense_rank=1, bm25_rank=2, k_rrf=60)
        expected = round(1.0 / 61.0 + 1.0 / 62.0, 6)
        assert score == expected
        assert score == 0.032522

    def test_dense_only_rank_present(self):
        # Dense rank 1, BM25 rank None
        # 1/(60+1) + 0 = 0.016393
        score = calculate_rrf_score(dense_rank=1, bm25_rank=None, k_rrf=60)
        expected = round(1.0 / 61.0, 6)
        assert score == expected
        assert score == 0.016393

    def test_bm25_only_rank_present(self):
        # Dense rank None, BM25 rank 1
        # 0 + 1/(60+1) = 0.016393
        score = calculate_rrf_score(dense_rank=None, bm25_rank=1, k_rrf=60)
        expected = round(1.0 / 61.0, 6)
        assert score == expected
        assert score == 0.016393

    def test_both_ranks_none_or_zero(self):
        assert calculate_rrf_score(dense_rank=None, bm25_rank=None) == 0.0
        assert calculate_rrf_score(dense_rank=0, bm25_rank=0) == 0.0

    def test_custom_weights(self):
        # dense_weight=2.0, bm25_weight=1.0, dense_rank=1, bm25_rank=1
        # 2/(60+1) + 1/(60+1) = 3/61 = 0.049180
        score = calculate_rrf_score(dense_rank=1, bm25_rank=1, k_rrf=60, dense_weight=2.0, bm25_weight=1.0)
        expected = round(3.0 / 61.0, 6)
        assert score == expected


class TestHybridRetrieverUnit:
    """Unit tests for HybridRetriever logic using mock stores."""

    @pytest.fixture
    def mock_store(self):
        store = MagicMock()
        # Mock retrieve_chunks returning 3 chunks
        store.retrieve_chunks.return_value = [
            {
                "chunk_id": "chunk_A",
                "text": "Text A about Transformers",
                "source": "doc1.txt",
                "chunk_index": 0,
                "distance": 0.25,
                "similarity": 0.75,
                "metadata": {"doc": "1"},
            },
            {
                "chunk_id": "chunk_B",
                "text": "Text B about Adam optimizer",
                "source": "doc2.txt",
                "chunk_index": 1,
                "distance": 0.40,
                "similarity": 0.60,
                "metadata": {"doc": "2"},
            },
            {
                "chunk_id": "chunk_C",
                "text": "Text C about Jupiter moons",
                "source": "doc3.txt",
                "chunk_index": 2,
                "distance": 0.50,
                "similarity": 0.50,
                "metadata": {"doc": "3"},
            },
        ]
        return store

    @pytest.fixture
    def mock_bm25(self):
        bm25 = MagicMock(spec=BM25Index)
        # Mock search returning chunk_B as rank 1, chunk_D as rank 2, chunk_A as rank 3
        bm25.search.return_value = [
            {
                "chunk_id": "chunk_B",
                "text": "Text B about Adam optimizer",
                "source": "doc2.txt",
                "chunk_index": 1,
                "bm25_score": 4.5,
                "metadata": {"doc": "2"},
            },
            {
                "chunk_id": "chunk_D",
                "text": "Text D about Mars Olympus Mons",
                "source": "doc4.txt",
                "chunk_index": 3,
                "bm25_score": 3.8,
                "metadata": {"doc": "4"},
            },
            {
                "chunk_id": "chunk_A",
                "text": "Text A about Transformers",
                "source": "doc1.txt",
                "chunk_index": 0,
                "bm25_score": 2.1,
                "metadata": {"doc": "1"},
            },
        ]
        return bm25

    def test_dense_only_strategy(self, mock_store, mock_bm25):
        retriever = HybridRetriever(vector_store=mock_store, bm25_index=mock_bm25)
        results = retriever.retrieve("test query", k=3, strategy="dense")

        assert len(results) == 3
        assert results[0]["chunk_id"] == "chunk_A"
        assert results[0]["dense_rank"] == 1
        assert results[0]["bm25_rank"] is None
        assert results[0]["dense_distance"] == 0.25
        assert results[0]["dense_similarity"] == 0.75
        assert results[0]["retrieval_strategy"] == "dense"
        mock_store.retrieve_chunks.assert_called_once_with("test query", k=3)
        mock_bm25.search.assert_not_called()

    def test_bm25_only_strategy(self, mock_store, mock_bm25):
        retriever = HybridRetriever(vector_store=mock_store, bm25_index=mock_bm25)
        results = retriever.retrieve("test query", k=3, strategy="bm25")

        assert len(results) == 3
        assert results[0]["chunk_id"] == "chunk_B"
        assert results[0]["bm25_rank"] == 1
        assert results[0]["dense_rank"] is None
        assert results[0]["bm25_score"] == 4.5
        assert results[0]["retrieval_strategy"] == "bm25"
        mock_bm25.search.assert_called_once_with("test query", k=3)
        mock_store.retrieve_chunks.assert_not_called()

    def test_hybrid_rrf_ranking_correctness(self, mock_store, mock_bm25):
        retriever = HybridRetriever(vector_store=mock_store, bm25_index=mock_bm25)
        results = retriever.retrieve_hybrid("test query", k=4)

        # Candidates:
        # chunk_A: dense_rank=1, bm25_rank=3 -> 1/61 + 1/63 = 0.016393 + 0.015873 = 0.032266
        # chunk_B: dense_rank=2, bm25_rank=1 -> 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
        # chunk_C: dense_rank=3, bm25_rank=None -> 1/63 = 0.015873
        # chunk_D: dense_rank=None, bm25_rank=2 -> 1/62 = 0.016129

        # Rank 1: chunk_B (0.032522)
        # Rank 2: chunk_A (0.032266)
        # Rank 3: chunk_D (0.016129)
        # Rank 4: chunk_C (0.015873)
        assert len(results) == 4
        assert [r["chunk_id"] for r in results] == ["chunk_B", "chunk_A", "chunk_D", "chunk_C"]
        assert results[0]["rrf_score"] > results[1]["rrf_score"]
        assert results[1]["rrf_score"] > results[2]["rrf_score"]
        assert results[2]["rrf_score"] > results[3]["rrf_score"]

    def test_documents_present_in_only_one_ranking(self, mock_store, mock_bm25):
        retriever = HybridRetriever(vector_store=mock_store, bm25_index=mock_bm25)
        results = retriever.retrieve_hybrid("test query", k=4)

        # chunk_D is BM25 only
        chunk_d = next(r for r in results if r["chunk_id"] == "chunk_D")
        assert chunk_d["dense_rank"] is None
        assert chunk_d["dense_distance"] is None
        assert chunk_d["bm25_rank"] == 2
        assert chunk_d["bm25_score"] == 3.8
        assert chunk_d["rrf_score"] == round(1.0 / 62.0, 6)

        # chunk_C is Dense only
        chunk_c = next(r for r in results if r["chunk_id"] == "chunk_C")
        assert chunk_c["dense_rank"] == 3
        assert chunk_c["dense_distance"] == 0.50
        assert chunk_c["bm25_rank"] is None
        assert chunk_c["bm25_score"] is None
        assert chunk_c["rrf_score"] == round(1.0 / 63.0, 6)

    def test_duplicate_chunk_removal(self, mock_store, mock_bm25):
        retriever = HybridRetriever(vector_store=mock_store, bm25_index=mock_bm25)
        results = retriever.retrieve_hybrid("test query", k=10)

        chunk_ids = [r["chunk_id"] for r in results]
        assert len(chunk_ids) == len(set(chunk_ids)), "Every chunk_id must be unique in fused results"

    def test_metadata_and_diagnostics_preservation(self, mock_store, mock_bm25):
        retriever = HybridRetriever(vector_store=mock_store, bm25_index=mock_bm25)
        results = retriever.retrieve_hybrid("test query", k=1)
        top = results[0]

        required_keys = [
            "chunk_id", "text", "source", "chunk_index", "metadata",
            "dense_rank", "bm25_rank", "dense_distance", "dense_similarity",
            "bm25_score", "rrf_score", "retrieval_strategy"
        ]
        for key in required_keys:
            assert key in top, f"Missing required diagnostic key: {key}"

        assert top["retrieval_strategy"] == "hybrid_rrf"
        # Confirm distance and similarity are properly distinct
        assert top["dense_distance"] == 0.40
        assert top["dense_similarity"] == 0.60
        assert top["dense_distance"] != top["dense_similarity"]

    def test_empty_and_whitespace_query(self, mock_store, mock_bm25):
        retriever = HybridRetriever(vector_store=mock_store, bm25_index=mock_bm25)
        assert retriever.retrieve_hybrid("") == []
        assert retriever.retrieve_hybrid("   \n\t  ") == []
        assert retriever.retrieve_hybrid(None) == []

    def test_deterministic_tie_breaking(self, mock_store, mock_bm25):
        # Two documents with identical RRF score (both rank 1 in their respective single ranking)
        store = MagicMock()
        store.retrieve_chunks.return_value = [
            {"chunk_id": "chunk_z", "text": "Z", "source": "z.txt", "distance": 0.1, "metadata": {}}
        ]
        bm25 = MagicMock(spec=BM25Index)
        bm25.search.return_value = [
            {"chunk_id": "chunk_a", "text": "A", "source": "a.txt", "bm25_score": 10.0, "metadata": {}}
        ]
        retriever = HybridRetriever(vector_store=store, bm25_index=bm25)
        results = retriever.retrieve_hybrid("query", k=2)

        # Both have rank 1 in one retriever and None in the other -> identical RRF score (1/61)
        assert results[0]["rrf_score"] == results[1]["rrf_score"]
        # Tie-breaker must sort chunk_id alphabetically: chunk_a before chunk_z
        assert results[0]["chunk_id"] == "chunk_a"
        assert results[1]["chunk_id"] == "chunk_z"

    def test_configurable_top_k(self, mock_store, mock_bm25):
        retriever = HybridRetriever(vector_store=mock_store, bm25_index=mock_bm25)
        assert len(retriever.retrieve_hybrid("query", k=1)) == 1
        assert len(retriever.retrieve_hybrid("query", k=2)) == 2
        assert len(retriever.retrieve_hybrid("query", k=3)) == 3
        assert len(retriever.retrieve_hybrid("query", k=10)) == 4  # only 4 candidates exist


class TestHybridRetrieverIntegration:
    """Integration tests running against the persistent ChromaDB collection and BM25 index."""

    def test_live_chromadb_and_bm25_integration(self):
        retriever = get_hybrid_retriever()
        assert retriever.vector_store.count() >= 35

        # Query where both dense and sparse terms are informative
        results = retriever.retrieve_hybrid("Olympus Mons Mars height in kilometers", k=4)
        assert len(results) > 0

        top = results[0]
        assert top["chunk_id"] == "solar_system_txt_chunk_0003"
        assert top["source"] == "solar_system.txt"
        assert top["dense_rank"] is not None
        assert top["bm25_rank"] is not None
        assert top["rrf_score"] > 0.0
        assert "21.9" in top["text"]

    def test_retriever_module_compatibility(self):
        from src.retriever import retrieve, hybrid_retrieve

        # Backward compatibility: strategy='dense'
        dense_res = retrieve("What optimizer trained the Transformer?", k=2, strategy="dense")
        assert len(dense_res) == 2
        assert "chunk_id" in dense_res[0]

        # Hybrid strategy via retrieve
        hybrid_res = retrieve("What optimizer trained the Transformer?", k=2, strategy="hybrid")
        assert len(hybrid_res) == 2
        assert hybrid_res[0]["retrieval_strategy"] == "hybrid_rrf"

        # Direct convenience function
        conv_res = hybrid_retrieve("What optimizer trained the Transformer?", k=2)
        assert len(conv_res) == 2
        assert conv_res[0]["retrieval_strategy"] == "hybrid_rrf"

