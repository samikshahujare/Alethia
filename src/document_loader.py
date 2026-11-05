"""Document loader module for Alethia.

Responsible for loading and validating raw text and PDF documents from disk or memory,
preserving source metadata and enforcing file integrity.
"""

from dataclasses import dataclass, field
import io
from pathlib import Path
from typing import Any, Dict, List, Union


@dataclass
class Document:
    """Represents a loaded text document with accompanying metadata."""

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def source(self) -> str:
        """Convenience property for accessing document source."""
        return self.metadata.get("source", "unknown")


def _extract_text_from_pdf_stream(stream: io.BytesIO, filename: str) -> Document:
    """Extract text and metadata from a PDF byte stream using pypdf.

    Args:
        stream: In-memory byte stream of the PDF file.
        filename: Name or identifier of the source file.

    Returns:
        Document instance containing extracted text and page metadata.

    Raises:
        ValueError: If PDF is empty or contains no extractable text.
        RuntimeError: If PDF parsing fails.
    """
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise RuntimeError("pypdf is required to process PDF files. Please install pypdf.") from e

    try:
        reader = PdfReader(stream)
        num_pages = len(reader.pages)
        if num_pages == 0:
            raise ValueError(f"PDF document has no pages: {filename}")

        page_texts = []
        for page_idx, page in enumerate(reader.pages, 1):
            extracted = page.extract_text() or ""
            if extracted.strip():
                page_texts.append(extracted.strip())

        full_text = "\n\n".join(page_texts).strip()
        if not full_text:
            raise ValueError(
                f"PDF document '{filename}' contains no extractable text (it may be scanned/image-only)."
            )

        return Document(
            text=full_text,
            metadata={
                "source": filename,
                "file_type": "pdf",
                "page_count": num_pages,
                "char_count": len(full_text),
            },
        )
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Error parsing PDF file '{filename}': {e}") from e


def load_document(file_path: Union[str, Path]) -> Document:
    """Load a single .txt or .pdf document from disk.

    Args:
        file_path: Path to the .txt or .pdf file.

    Returns:
        Document instance containing raw text and metadata.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file format is unsupported, or if file is empty.
        RuntimeError: If file cannot be read due to permission or I/O errors.
    """
    path = Path(file_path).resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Document file not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in (".txt", ".pdf"):
        raise ValueError(
            f"Unsupported file type '{path.suffix}'. Alethia supports '.txt' and '.pdf' files."
        )

    if suffix == ".pdf":
        try:
            with open(path, "rb") as f:
                stream = io.BytesIO(f.read())
            doc = _extract_text_from_pdf_stream(stream, path.name)
            doc.metadata["file_path"] = str(path)
            return doc
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise RuntimeError(f"Error reading PDF file {path.name}: {e}") from e

    # Handling .txt files
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            raise RuntimeError(f"Failed to read file {path.name}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error reading file {path.name}: {e}") from e

    cleaned_text = content.strip()
    if not cleaned_text:
        raise ValueError(f"Document is empty: {path.name}")

    return Document(
        text=content,
        metadata={
            "source": path.name,
            "file_type": "txt",
            "file_path": str(path),
            "char_count": len(content),
        },
    )


def load_document_from_bytes(data: bytes, filename: str) -> Document:
    """Load and parse document text from raw bytes (supporting .txt and .pdf).

    Args:
        data: Raw file byte array.
        filename: Original file name.

    Returns:
        Document instance with extracted text and metadata.
    """
    if not data or not len(data):
        raise ValueError(f"File '{filename}' is empty.")

    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return _extract_text_from_pdf_stream(io.BytesIO(data), filename)

    # Default to text decoding
    try:
        content = data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            content = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            content = data.decode("latin-1")

    cleaned = content.strip()
    if not cleaned:
        raise ValueError(f"Document '{filename}' contains no readable text.")

    return Document(
        text=content,
        metadata={
            "source": filename,
            "file_type": "txt" if suffix == ".txt" else "other",
            "char_count": len(content),
        },
    )


def create_document_from_text(text: str, title: str = "pasted_note.txt") -> Document:
    """Create a Document instance directly from raw text input.

    Args:
        text: Raw user-provided text string.
        title: Optional document title / identifier.

    Returns:
        Document instance.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("Provided text is empty.")

    safe_title = title.strip()
    if not safe_title.endswith((".txt", ".md")):
        safe_title = f"{safe_title}.txt"

    return Document(
        text=cleaned,
        metadata={
            "source": safe_title,
            "file_type": "txt",
            "char_count": len(cleaned),
        },
    )


def load_documents_from_directory(dir_path: Union[str, Path]) -> List[Document]:
    """Load all .txt and .pdf documents from a directory.

    Args:
        dir_path: Path to directory containing documents.

    Returns:
        List of loaded Document objects. Empty or unreadable files are skipped.

    Raises:
        FileNotFoundError: If directory does not exist.
        NotADirectoryError: If dir_path is not a directory.
    """
    path = Path(dir_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path}")

    matching_files = sorted(list(path.glob("*.txt")) + list(path.glob("*.pdf")))
    if not matching_files:
        return []

    documents: List[Document] = []
    for file in matching_files:
        try:
            doc = load_document(file)
            documents.append(doc)
        except Exception:
            # Skip invalid or empty files in bulk directory mode
            continue

    return documents


