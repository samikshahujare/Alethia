"""Regression tests for document updates, duplicate detection, BM25 synchronization, and SOP grounding.

Verifies:
1. Updated document with same filename detects content change and replaces old chunks.
2. Duplicate unchanged upload is detected and skipped without re-embedding.
3. Removal of outdated chunks prevents orphaned chunks.
4. BM25 synchronization after document replacement reflects fresh corpus.
5. SOP singular and plural queries retrieve the definition chunk.
6. Exact SOP definition appears in generator context.
7. Answer generation produces definition and Judge approves it.
8. Judge rejects answers when evidence is absent (Rule 3).
9. Source attribution accurately reports document and chunk ID.
"""

import pytest
from src.chunker import TextChunk
from src.document_loader import Document
from src.text_cleaner import clean_document
from src.chunker import chunk_document
from src.store import VectorStore, get_vector_store
from src.hybrid_retriever import HybridRetriever
from src.prompts import (
    GENERATE_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    format_generate_prompt,
    format_judge_prompt,
    format_context_from_chunks,
)
from src.nodes import judge_node, generate_node
from src.state import GraphState


@pytest.fixture
def store():
    return get_vector_store()


def test_updated_document_same_filename_detects_change(tmp_path):
    """Test that uploading modified content with the same filename re-indexes."""
    test_store = VectorStore(persist_directory=tmp_path / "test_chroma", collection_name="test_col")
    
    doc1 = Document(text="Initial section A content. Initial section B content.", metadata={"source": "test.txt"})
    chunks1 = chunk_document(clean_document(doc1))
    indexed1 = test_store.index_chunks(chunks1)
    assert indexed1 == len(chunks1)

    # Modify content with same filename
    doc2 = Document(text="Updated section A content with new information. Completely changed section B.", metadata={"source": "test.txt"})
    chunks2 = chunk_document(clean_document(doc2))
    
    # Duplicate detection should detect change and re-index
    indexed2 = test_store.index_chunks(chunks2, overwrite=False)
    assert indexed2 == len(chunks2)
    
    # Total count should match chunks2, not chunks1 + chunks2
    assert test_store.count() == len(chunks2)


def test_duplicate_unchanged_upload_skips(tmp_path):
    """Test that uploading the exact same document twice is detected and skipped."""
    test_store = VectorStore(persist_directory=tmp_path / "test_chroma_dup", collection_name="test_col_dup")
    
    doc = Document(text="Static content for duplicate testing that does not change.", metadata={"source": "static.txt"})
    chunks = chunk_document(clean_document(doc))
    
    first_index = test_store.index_chunks(chunks, overwrite=False)
    assert first_index == len(chunks)
    
    # Second upload with identical content
    second_index = test_store.index_chunks(chunks, overwrite=False)
    assert second_index == 0


def test_removal_of_outdated_chunks(tmp_path):
    """Test that shorter updated document does not leave orphaned old chunks."""
    test_store = VectorStore(persist_directory=tmp_path / "test_chroma_orphan", collection_name="test_col_orphan")
    
    long_text = "Paragraph 1 is here. " * 50 + "\n\n" + "Paragraph 2 is here. " * 50
    doc_long = Document(text=long_text, metadata={"source": "shrink.txt"})
    chunks_long = chunk_document(clean_document(doc_long))
    assert len(chunks_long) > 1
    test_store.index_chunks(chunks_long)
    count_long = test_store.count()
    
    # Shorter version
    short_text = "Just a short single sentence document."
    doc_short = Document(text=short_text, metadata={"source": "shrink.txt"})
    chunks_short = chunk_document(clean_document(doc_short))
    assert len(chunks_short) < count_long
    
    test_store.index_chunks(chunks_short, overwrite=False)
    assert test_store.count() == len(chunks_short)


def test_bm25_synchronization_after_document_replacement(tmp_path):
    """Test that BM25 index accurately reflects document replacement."""
    test_store = VectorStore(persist_directory=tmp_path / "test_chroma_bm25", collection_name="test_col_bm25")
    retriever = HybridRetriever(vector_store=test_store)
    
    doc1 = Document(text="Old unique token: ZYXWVUT", metadata={"source": "rep.txt"})
    test_store.index_chunks(chunk_document(clean_document(doc1)))
    retriever.sync_bm25()
    
    assert len(retriever.retrieve_bm25("ZYXWVUT", k=1)) == 1
    
    # Replace document
    doc2 = Document(text="New unique token: ABCDEFGH", metadata={"source": "rep.txt"})
    test_store.index_chunks(chunk_document(clean_document(doc2)))
    retriever.sync_bm25()
    
    assert len(retriever.retrieve_bm25("ZYXWVUT", k=1)) == 0
    assert len(retriever.retrieve_bm25("ABCDEFGH", k=1)) == 1


def test_sop_singular_plural_retrieval(store):
    """Test that both 'What is SOP?' and 'What are SOPs?' retrieve chunk 0006 at rank 1."""
    retriever = HybridRetriever(vector_store=store)
    retriever.sync_bm25()
    
    for q in ["What is SOP?", "What are SOPs?"]:
        results = retriever.retrieve(q, k=4)
        assert len(results) > 0
        top = results[0]
        assert top["chunk_id"] == "Tell_me_about_yourself_txt_chunk_0006"
        assert "step-by-step process" in top["text"]


def test_exact_sop_definition_in_generator_context(store):
    """Test that the retrieved context contains the exact definition for SOP."""
    retriever = HybridRetriever(vector_store=store)
    chunks = retriever.retrieve("What is SOP?", k=4)
    context = format_context_from_chunks(chunks)
    
    assert "What is SOP?" in context
    assert "A documented step-by-step process that everyone follows to ensure work is consistent and accurate." in context


def test_correct_sop_answer_generation_and_judge():
    """Test that generator creates definition and Judge approves it when grounded."""
    context = (
        "[Document Chunk 1 | Source: Tell me about yourself.txt | ID: Tell_me_about_yourself_txt_chunk_0006]\n"
        "What is SOP? A documented step-by-step process that everyone follows to ensure work is consistent and accurate."
    )
    mock_draft = "An SOP is a documented step-by-step process that everyone follows to ensure work is consistent and accurate."
    
    mock_llm_gen = lambda messages, **kwargs: mock_draft
    state: GraphState = {
        "question": "What is SOP?",
        "original_question": "What is SOP?",
        "chunks": [{"chunk_id": "Tell_me_about_yourself_txt_chunk_0006", "text": "What is SOP? A documented step-by-step process that everyone follows to ensure work is consistent and accurate.", "source": "Tell me about yourself.txt"}],
        "trace": [],
        "attempts": 0,
    }
    
    gen_state = generate_node(state, llm_func=mock_llm_gen)
    assert "step-by-step process" in gen_state["draft"]
    
    mock_judge_llm = lambda messages, **kwargs: "VERDICT: PASS\nREASON: Answer directly quotes the definition from the context."
    judge_state = judge_node(gen_state, llm_func=mock_judge_llm)
    assert judge_state["verdict"] == "PASS"


def test_judge_rejects_when_evidence_absent():
    """Test that Judge rejects an invented definition when context does not contain it."""
    context = (
        "[Document Chunk 1 | Source: Tell me about yourself.txt | ID: Tell_me_about_yourself_txt_chunk_0009]\n"
        "I led a campus placement dashboard project where I managed placement data, ensured data accuracy, created SOPs, and developed dashboards."
    )
    invented_draft = "An SOP is a Standard Operating Procedure used in enterprise supply chain management."
    
    # Judge should output FAIL for unsupported claims
    mock_judge_llm = lambda messages, **kwargs: "VERDICT: FAIL\nREASON: The context does not define SOP or mention supply chain management."
    state: GraphState = {
        "question": "What is SOP?",
        "draft": invented_draft,
        "chunks": [{"chunk_id": "chunk_0009", "text": "created SOPs", "source": "Tell me about yourself.txt"}],
        "trace": [],
        "attempts": 0,
    }
    res = judge_node(state, llm_func=mock_judge_llm)
    assert res["verdict"] == "FAIL"


def test_source_attribution_correct(store):
    """Test that source attribution accurately names Tell me about yourself.txt."""
    retriever = HybridRetriever(vector_store=store)
    chunks = retriever.retrieve("What is SOP?", k=1)
    assert chunks[0]["source"] == "Tell me about yourself.txt"
    assert chunks[0]["chunk_id"] == "Tell_me_about_yourself_txt_chunk_0006"

