"""Unit tests for ChromaDB storage, embeddings, and semantic retrieval.

Tests:
- Embedding generation with all-MiniLM-L6-v2.
- Vector store chunk indexing and duplicate prevention.
- Semantic similarity retrieval and relevance ordering.
- Boundary conditions: empty collection, invalid queries, k > total chunks.
- Disk persistence across VectorStore client re-initialization.
"""

import gc
import shutil
import tempfile
from pathlib import Path
import pytest

from src.chunker import TextChunk
from src.store import VectorStore


@pytest.fixture
def temp_store():
    """Create an isolated temporary VectorStore for each test with Windows-safe teardown."""
    temp_dir = tempfile.mkdtemp()
    store = VectorStore(
        persist_directory=temp_dir,
        collection_name="test_collection",
    )
    try:
        yield store
    finally:
        store.close()
        del store
        gc.collect()
        shutil.rmtree(temp_dir, ignore_errors=True)


class TestVectorStoreAndRetrieval:
    """Test suite for VectorStore and semantic retrieval."""

    def test_embedding_generation_shape(self, temp_store: VectorStore):
        texts = ["Sentence one for testing.", "Sentence two for testing."]
        embeddings = temp_store.generate_embeddings(texts)
        assert len(embeddings) == 2
        # all-MiniLM-L6-v2 produces 384-dimensional vectors
        assert len(embeddings[0]) == 384
        assert len(embeddings[1]) == 384
        assert isinstance(embeddings[0][0], float)

    def test_empty_database_retrieval(self, temp_store: VectorStore):
        results = temp_store.retrieve_chunks("What is attention?")
        assert results == []

    def test_invalid_query_raises_value_error(self, temp_store: VectorStore):
        with pytest.raises(ValueError, match="Question cannot be empty"):
            temp_store.retrieve_chunks("")

        with pytest.raises(ValueError, match="Question cannot be empty"):
            temp_store.retrieve_chunks("   \n\t  ")

    def test_index_chunks_and_count(self, temp_store: VectorStore):
        chunks = [
            TextChunk(
                chunk_id="doc_chunk_0000",
                text="The Transformer employs six encoder layers.",
                source="doc.txt",
                chunk_index=0,
            ),
            TextChunk(
                chunk_id="doc_chunk_0001",
                text="The model is trained using the Adam optimizer.",
                source="doc.txt",
                chunk_index=1,
            ),
        ]
        indexed_count = temp_store.index_chunks(chunks)
        assert indexed_count == 2
        assert temp_store.count() == 2

    def test_reindexing_prevents_duplicates(self, temp_store: VectorStore):
        chunks = [
            TextChunk(
                chunk_id="doc_chunk_0000",
                text="The Transformer employs six encoder layers.",
                source="doc.txt",
                chunk_index=0,
            ),
            TextChunk(
                chunk_id="doc_chunk_0001",
                text="The model is trained using the Adam optimizer.",
                source="doc.txt",
                chunk_index=1,
            ),
        ]
        # First index
        first_count = temp_store.index_chunks(chunks)
        assert first_count == 2
        assert temp_store.count() == 2

        # Re-index same chunks without overwrite
        second_count = temp_store.index_chunks(chunks, overwrite=False)
        assert second_count == 0
        assert temp_store.count() == 2

    def test_reindex_document_clears_orphaned_chunks(self, temp_store: VectorStore):
        # Index document A with 4 chunks and document B with 1 chunk
        doc_a_old = [
            TextChunk(chunk_id=f"docA_chunk_{i:04d}", text=f"Doc A old section {i}", source="docA.txt", chunk_index=i)
            for i in range(4)
        ]
        doc_b = [
            TextChunk(chunk_id="docB_chunk_0000", text="Doc B content", source="docB.txt", chunk_index=0)
        ]
        temp_store.index_chunks(doc_a_old)
        temp_store.index_chunks(doc_b)
        assert temp_store.count() == 5

        # Modify document A: it now has only 2 chunks
        doc_a_updated = [
            TextChunk(chunk_id=f"docA_chunk_{i:04d}", text=f"Doc A updated section {i}", source="docA.txt", chunk_index=i)
            for i in range(2)
        ]
        # Re-index document A
        reindexed_count = temp_store.reindex_document(doc_a_updated)
        assert reindexed_count == 2
        # Total should now be 2 (from docA) + 1 (from docB) = 3
        assert temp_store.count() == 3

        # Verify old chunk IDs 2 and 3 were purged
        res = temp_store.retrieve_chunks("Doc A", k=5)
        retrieved_ids = {r["chunk_id"] for r in res}
        assert "docA_chunk_0002" not in retrieved_ids
        assert "docA_chunk_0003" not in retrieved_ids
        # Verify Doc B remains intact
        b_res = temp_store.retrieve_chunks("Doc B content", k=1)
        assert b_res[0]["chunk_id"] == "docB_chunk_0000"

    def test_semantic_retrieval_relevance(self, temp_store: VectorStore):
        chunks = [
            TextChunk(
                chunk_id="c_layers",
                text="The model consists of six encoder layers and six decoder layers in a stack.",
                source="architecture.txt",
                chunk_index=0,
            ),
            TextChunk(
                chunk_id="c_optimizer",
                text="Optimization was performed using the Adam optimizer with beta1=0.9 and beta2=0.98.",
                source="training.txt",
                chunk_index=1,
            ),
            TextChunk(
                chunk_id="c_recipe",
                text="To make delicious Italian pasta, boil water and add salt before adding noodles.",
                source="cooking.txt",
                chunk_index=2,
            ),
        ]
        temp_store.index_chunks(chunks)

        # Query about optimization
        query = "Which optimizer was used during training?"
        results = temp_store.retrieve_chunks(query, k=2)

        assert len(results) == 2
        top_match = results[0]
        assert top_match["chunk_id"] == "c_optimizer"
        assert "Adam optimizer" in top_match["text"]
        assert "similarity" in top_match
        assert "distance" in top_match
        assert top_match["similarity"] > 0.4

    def test_retrieval_k_limits(self, temp_store: VectorStore):
        chunks = [
            TextChunk(
                chunk_id=f"chunk_{i:04d}",
                text=f"Information segment number {i} regarding machine learning.",
                source="corpus.txt",
                chunk_index=i,
            )
            for i in range(8)
        ]
        temp_store.index_chunks(chunks)

        # Retrieve top 3
        results = temp_store.retrieve_chunks("machine learning", k=3)
        assert len(results) == 3

    def test_retrieval_when_fewer_chunks_than_k(self, temp_store: VectorStore):
        chunks = [
            TextChunk(
                chunk_id="solo_chunk",
                text="Single available knowledge item.",
                source="solo.txt",
                chunk_index=0,
            )
        ]
        temp_store.index_chunks(chunks)

        results = temp_store.retrieve_chunks("knowledge item", k=4)
        assert len(results) == 1
        assert results[0]["chunk_id"] == "solo_chunk"

    def test_persistence_across_instances(self):
        temp_dir = tempfile.mkdtemp()
        try:
            store1 = VectorStore(persist_directory=temp_dir, collection_name="persist_col")
            chunks = [
                TextChunk(
                    chunk_id="persist_01",
                    text="Persistent data stored safely on disk.",
                    source="persist.txt",
                    chunk_index=0,
                )
            ]
            store1.index_chunks(chunks)
            assert store1.count() == 1
            store1.close()
            del store1
            gc.collect()

            # Simulate process restart by creating a new VectorStore instance on the same path
            store2 = VectorStore(persist_directory=temp_dir, collection_name="persist_col")
            assert store2.count() == 1
            results = store2.retrieve_chunks("stored safely", k=1)
            assert len(results) == 1
            assert results[0]["chunk_id"] == "persist_01"
            store2.close()
            del store2
            gc.collect()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
