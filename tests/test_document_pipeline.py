"""Unit tests for Alethia Document Processing Pipeline.

Tests:
- Document loading (.txt files, directory loading).
- Error handling on non-.txt and empty files.
- Text cleaning and normalization.
- Sentence-aware chunking, overlap, metadata preservation, and unique ID assignment.
"""

import tempfile
from pathlib import Path
import pytest

from src.document_loader import Document, load_document, load_documents_from_directory
from src.text_cleaner import clean_text, clean_document
from src.chunker import TextChunk, chunk_text, chunk_document


class TestDocumentLoader:
    """Test suite for src.document_loader."""

    def test_load_valid_txt_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Sample document content for testing.")
            temp_path = f.name

        try:
            doc = load_document(temp_path)
            assert isinstance(doc, Document)
            assert "Sample document content" in doc.text
            assert doc.source == Path(temp_path).name
            assert doc.metadata["char_count"] == len(doc.text)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_reject_non_txt_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".pdf", delete=False, encoding="utf-8") as f:
            f.write("Dummy PDF content")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                load_document(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_reject_empty_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("   \n\t  \n  ")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Document is empty"):
                load_document(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_documents_from_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir)
            (dir_path / "doc1.txt").write_text("Document one text.", encoding="utf-8")
            (dir_path / "doc2.txt").write_text("Document two text.", encoding="utf-8")
            (dir_path / "ignored.md").write_text("Markdown file.", encoding="utf-8")

            docs = load_documents_from_directory(dir_path)
            assert len(docs) == 2
            sources = [d.source for d in docs]
            assert "doc1.txt" in sources
            assert "doc2.txt" in sources


class TestTextCleaner:
    """Test suite for src.text_cleaner."""

    def test_whitespace_and_newline_normalization(self):
        raw = "This   is   a    sentence.\n\n\n\nWith    too   many   newlines."
        cleaned = clean_text(raw)
        assert "This is a sentence." in cleaned
        assert "\n\n" in cleaned
        assert "\n\n\n" not in cleaned
        assert "With too many newlines." in cleaned

    def test_control_character_removal(self):
        # Insert null byte, bell, escape
        raw = "Hello\x00 World\x07!\x1b Test"
        cleaned = clean_text(raw)
        assert cleaned == "Hello World! Test"

    def test_preserves_punctuation_and_unicode(self):
        raw = 'Transformer achieves 28.4 BLEU! Does it work? Yes: "scaled dot-product" — résumé.'
        cleaned = clean_text(raw)
        assert cleaned == raw

    def test_clean_document_updates_metadata(self):
        doc = Document(text="  Hello   world!  ", metadata={"source": "test.txt"})
        cleaned_doc = clean_document(doc)
        assert cleaned_doc.text == "Hello world!"
        assert cleaned_doc.metadata["source"] == "test.txt"
        assert cleaned_doc.metadata["cleaned_char_count"] == 12


class TestChunker:
    """Test suite for src.chunker."""

    def test_short_text_produces_single_valid_chunk(self):
        text = "This is a brief short document."
        chunks = chunk_text(text, source="short.txt", chunk_size=400)
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].chunk_index == 0
        assert chunks[0].chunk_id == "short_txt_chunk_0000"
        assert chunks[0].source == "short.txt"

    def test_long_text_splits_into_multiple_chunks(self):
        sentences = [
            f"Sentence number {i} provides detailed contextual explanation for the neural network architecture."
            for i in range(15)
        ]
        text = " ".join(sentences)
        assert len(text) > 1000

        chunks = chunk_text(text, source="long.txt", chunk_size=400, chunk_overlap=60)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.text) > 0
            # Allow minor leeway for sentence boundary accommodation
            assert len(chunk.text) <= 500

    def test_chunk_ids_are_unique(self):
        sentences = [f"Unique information statement {i} about transformers." for i in range(20)]
        text = " ".join(sentences)
        chunks = chunk_text(text, source="unique_test.txt", chunk_size=300)
        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_source_metadata_is_preserved(self):
        doc = Document(
            text="First sentence about attention. Second sentence about feedforward networks.",
            metadata={"source": "transformer.txt", "author": "Vaswani"},
        )
        chunks = chunk_document(doc, chunk_size=100)
        for chunk in chunks:
            assert chunk.source == "transformer.txt"
            assert chunk.metadata["source"] == "transformer.txt"
            assert chunk.metadata["author"] == "Vaswani"
            assert "chunk_index" in chunk.metadata

    def test_empty_or_whitespace_text_returns_empty_list(self):
        assert chunk_text("") == []
        assert chunk_text("    \n\n  ") == []

