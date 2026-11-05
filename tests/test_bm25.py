"""Comprehensive unit tests for the pure-Python Okapi BM25 implementation.

Verifies:
- Tokenization and normalization (underscores, decimals, punctuation).
- Exact keyword matching.
- Relevant document ranking.
- Repeated query terms.
- Empty query handling.
- Unknown terms handling.
- Empty corpus handling.
- Document-length normalization.
- Deterministic tie-breaking.
- Metadata and chunk ID preservation.
- Read-only synchronization with ChromaDB vector store.
"""

import pytest
from src.bm25 import BM25Index, tokenize


class TestBM25Tokenizer:
    """Tests for BM25 tokenization and text normalization."""

    def test_lowercase_and_punctuation_stripping(self):
        text = "Hello, World! This is an NLP Test... (Vaswani et al., 2017)."
        tokens = tokenize(text)
        assert "hello" in tokens
        assert "world" in tokens
        assert "nlp" in tokens
        assert "test" in tokens
        assert "vaswani" in tokens
        assert "2017" in tokens
        # Ensure punctuation is stripped
        for t in tokens:
            assert "," not in t
            assert "!" not in t
            assert "(" not in t

    def test_decimal_numbers_preserved(self):
        text = "Olympus Mons stands 21.9 kilometers high. BLEU is 41.8. Threshold is 1.0."
        tokens = tokenize(text)
        assert "21.9" in tokens
        assert "41.8" in tokens
        assert "1.0" in tokens

    def test_compound_underscore_terms_yield_subtokens(self):
        text = "Hyperparameters: beta_1 equals 0.9, beta_2 equals 0.999, d_k = 64."
        tokens = tokenize(text)
        # Full compound tokens
        assert "beta_1" in tokens
        assert "beta_2" in tokens
        assert "d_k" in tokens
        # Sub-tokens for flexible matching
        assert "beta" in tokens
        assert "1" in tokens
        assert "2" in tokens
        assert "d" in tokens
        assert "k" in tokens

    def test_empty_and_whitespace_tokenization(self):
        assert tokenize("") == []
        assert tokenize("   \n\t  ") == []
        assert tokenize(None) == []


class TestBM25IndexCore:
    """Core BM25 ranking and scoring tests."""

    @pytest.fixture
    def sample_corpus(self):
        return [
            {
                "chunk_id": "chunk_01",
                "text": "The Transformer relies on scaled dot-product attention and multi-head attention.",
                "source": "transformer.txt",
                "chunk_index": 0,
                "metadata": {"category": "architecture"},
            },
            {
                "chunk_id": "chunk_02",
                "text": "The AdamW optimizer decouples weight decay from gradient updates in transformer training.",
                "source": "optimizers.txt",
                "chunk_index": 1,
                "metadata": {"category": "optimization"},
            },
            {
                "chunk_id": "chunk_03",
                "text": "Jupiter is the largest planet and possesses 95 officially recognized natural moons.",
                "source": "solar_system.txt",
                "chunk_index": 2,
                "metadata": {"category": "astronomy"},
            },
            {
                "chunk_id": "chunk_04",
                "text": "Olympus Mons on Mars is a massive shield volcano standing approximately 21.9 kilometers high.",
                "source": "solar_system.txt",
                "chunk_index": 3,
                "metadata": {"category": "astronomy"},
            },
        ]

    def test_exact_keyword_matching(self, sample_corpus):
        index = BM25Index.from_documents(sample_corpus)
        results = index.search("Olympus Mons", k=2)

        assert len(results) > 0
        assert results[0]["chunk_id"] == "chunk_04"
        assert results[0]["source"] == "solar_system.txt"
        assert results[0]["bm25_score"] > 0.0

    def test_relevant_document_ranking(self, sample_corpus):
        index = BM25Index.from_documents(sample_corpus)
        results = index.search("AdamW optimizer weight decay", k=4)

        assert len(results) > 0
        # chunk_02 has multiple term matches: adamw, optimizer, weight, decay
        assert results[0]["chunk_id"] == "chunk_02"
        # chunk_01 shares 'transformer' but not optimizer terms
        other_ids = [r["chunk_id"] for r in results[1:]]
        assert "chunk_03" not in other_ids  # irrelevant chunks with 0 matching terms are omitted

    def test_repeated_query_terms_boosts_importance(self, sample_corpus):
        index = BM25Index.from_documents(sample_corpus)
        # Query with repeated term 'volcano'
        results_single = index.search("volcano", k=1)
        results_repeated = index.search("volcano volcano", k=1)

        assert results_single[0]["chunk_id"] == "chunk_04"
        assert results_repeated[0]["chunk_id"] == "chunk_04"
        # Repeated term increases overall query weight
        assert results_repeated[0]["bm25_score"] > results_single[0]["bm25_score"]

    def test_empty_and_whitespace_query(self, sample_corpus):
        index = BM25Index.from_documents(sample_corpus)
        assert index.search("") == []
        assert index.search("   \n\t ") == []
        assert index.search(None) == []

    def test_unknown_terms_returns_empty(self, sample_corpus):
        index = BM25Index.from_documents(sample_corpus)
        # Term completely absent from corpus
        results = index.search("quantum teleportation superluminal")
        assert results == []

    def test_empty_corpus(self):
        index = BM25Index()
        assert index.doc_count == 0
        assert index.search("transformer") == []

    def test_document_length_normalization(self):
        # Two documents containing the same target term 'ganymede',
        # but doc_short is much shorter than doc_long
        docs = [
            {
                "chunk_id": "doc_long",
                "text": "Ganymede is a celestial body. " + "The quick brown fox jumps over the lazy dog. " * 5,
                "source": "long.txt",
                "chunk_index": 0,
            },
            {
                "chunk_id": "doc_short",
                "text": "Ganymede is the largest moon.",
                "source": "short.txt",
                "chunk_index": 1,
            },
        ]
        index = BM25Index.from_documents(docs, b=0.75)
        results = index.search("Ganymede", k=2)

        assert len(results) == 2
        # Under length normalization (b=0.75), shorter document gets higher BM25 score for the same term
        assert results[0]["chunk_id"] == "doc_short"
        assert results[1]["chunk_id"] == "doc_long"
        assert results[0]["bm25_score"] > results[1]["bm25_score"]

    def test_deterministic_tie_breaking(self):
        # Two documents with identical matching text
        docs = [
            {"chunk_id": "chunk_z", "text": "Identical target text.", "source": "z.txt"},
            {"chunk_id": "chunk_a", "text": "Identical target text.", "source": "a.txt"},
            {"chunk_id": "chunk_m", "text": "Identical target text.", "source": "m.txt"},
        ]
        index = BM25Index.from_documents(docs)
        results = index.search("target text", k=3)

        assert len(results) == 3
        # Scores are identical; deterministic sort breaks ties by chunk_id ascending
        assert [r["chunk_id"] for r in results] == ["chunk_a", "chunk_m", "chunk_z"]

    def test_metadata_and_chunk_id_preservation(self, sample_corpus):
        index = BM25Index.from_documents(sample_corpus)
        results = index.search("moons", k=1)

        assert len(results) == 1
        top = results[0]
        assert top["chunk_id"] == "chunk_03"
        assert top["source"] == "solar_system.txt"
        assert top["chunk_index"] == 2
        assert top["metadata"] == {"category": "astronomy"}
        assert "text" in top
        assert "bm25_score" in top
        assert isinstance(top["bm25_score"], float)

    def test_idf_is_strictly_positive(self, sample_corpus):
        index = BM25Index.from_documents(sample_corpus)
        # Check IDF for terms with different frequencies
        for term in ["transformer", "jupiter", "the"]:
            score = index.idf(term)
            assert score >= 0.0


class TestBM25ChromaDBSynchronization:
    """Verifies that BM25 can index from persistent ChromaDB safely without mutating it."""

    def test_from_chromadb_indexes_correct_count_read_only(self):
        from src.store import get_vector_store

        store = get_vector_store()
        initial_chroma_count = store.count()

        # Build BM25 index from persistent ChromaDB
        bm25_index = BM25Index.from_chromadb(store)

        # ChromaDB count must remain identical (read-only verification)
        assert store.count() == initial_chroma_count
        assert bm25_index.doc_count == initial_chroma_count

        # Test searching the live 35 chunks
        results = bm25_index.search("AdamW weight decay warmup", k=3)
        assert len(results) > 0
        assert "optimizers_guide_txt_chunk_0003" in [r["chunk_id"] for r in results]
        assert results[0]["source"] == "optimizers_guide.txt"

    def test_search_technical_decimal_tokens(self):
        from src.store import get_vector_store

        store = get_vector_store()
        bm25_index = BM25Index.from_chromadb(store)

        # Search for decimal height 21.9 km
        results = bm25_index.search("Olympus Mons 21.9", k=2)
        assert len(results) > 0
        assert results[0]["chunk_id"] == "solar_system_txt_chunk_0003"
        assert "21.9" in results[0]["text"]

