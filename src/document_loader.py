"""Document loader module for Alethia.

Responsible for loading and validating raw text documents from disk,
preserving source metadata and enforcing file integrity.
"""

from dataclasses import dataclass, field
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


def load_document(file_path: Union[str, Path]) -> Document:
    """Load a single .txt document from disk.

    Args:
        file_path: Path to the .txt file.

    Returns:
        Document instance containing raw text and metadata.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file is not a .txt file, or if file is empty.
        RuntimeError: If file cannot be read due to permission or I/O errors.
    """
    path = Path(file_path).resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Document file not found: {path}")

    if path.suffix.lower() != ".txt":
        raise ValueError(
            f"Unsupported file type '{path.suffix}'. Alethia Phase 2 document loader only supports '.txt' files."
        )

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Fallback with error handling for non-strict UTF-8
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
            "file_path": str(path),
            "char_count": len(content),
        },
    )


def load_documents_from_directory(dir_path: Union[str, Path]) -> List[Document]:
    """Load all .txt documents from a directory.

    Args:
        dir_path: Path to directory containing .txt documents.

    Returns:
        List of loaded Document objects. Empty files are skipped or rejected.

    Raises:
        FileNotFoundError: If directory does not exist.
        NotADirectoryError: If dir_path is not a directory.
    """
    path = Path(dir_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path}")

    txt_files = sorted(list(path.glob("*.txt")))
    if not txt_files:
        return []

    documents: List[Document] = []
    for file in txt_files:
        try:
            doc = load_document(file)
            documents.append(doc)
        except ValueError as e:
            # Handle empty document or invalid format with a warning/skip in bulk mode
            continue

    return documents

