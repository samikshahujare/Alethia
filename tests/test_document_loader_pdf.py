"""Tests for PDF and multi-format document loading capabilities in Alethia."""

import io
from pathlib import Path
import pytest
from pypdf import PdfWriter

from src.document_loader import (
    Document,
    load_document,
    load_document_from_bytes,
    create_document_from_text,
    load_documents_from_directory,
)


def test_create_document_from_text():
    """Verify direct text document creation with metadata."""
    doc = create_document_from_text("Quarterly financial results: Profit grew by 15%.", title="q3_results.txt")
    assert isinstance(doc, Document)
    assert doc.source == "q3_results.txt"
    assert "Profit grew by 15%" in doc.text
    assert doc.metadata.get("char_count") == len(doc.text)
    assert doc.metadata.get("file_type") == "txt"


def test_create_document_from_text_appends_txt_extension():
    """Verify document title gets .txt appended if missing extension."""
    doc = create_document_from_text("Sample notes content", title="meeting_notes")
    assert doc.source == "meeting_notes.txt"


def test_create_document_from_empty_text_raises_value_error():
    """Verify empty text raises ValueError."""
    with pytest.raises(ValueError, match="empty"):
        create_document_from_text("   \n\t  ")


def test_load_document_from_bytes_txt():
    """Verify loading text document from bytes."""
    data = "Deep learning architectures have transformed NLP.".encode("utf-8")
    doc = load_document_from_bytes(data, "nlp.txt")
    assert doc.source == "nlp.txt"
    assert doc.metadata.get("file_type") == "txt"
    assert "transformed NLP" in doc.text


def test_load_document_from_bytes_empty_raises():
    """Verify empty bytes raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        load_document_from_bytes(b"", "empty.txt")


def test_load_document_from_bytes_blank_pdf_raises():
    """Verify blank PDF with no extractable text raises appropriate ValueError."""
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    pdf_bytes = buf.getvalue()

    with pytest.raises(ValueError, match="no extractable text"):
        load_document_from_bytes(pdf_bytes, "blank.pdf")


def test_load_document_rejects_unsupported_extension(tmp_path: Path):
    """Verify loader rejects unsupported extensions like .csv or .docx."""
    bad_file = tmp_path / "data.csv"
    bad_file.write_text("a,b,c", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(bad_file)

