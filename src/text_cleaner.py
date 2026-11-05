"""Text cleaning and normalization module for Alethia.

Prepares raw document text for chunking and embedding by removing control characters,
normalizing whitespace, and standardizing newlines while preserving punctuation and semantics.
"""

import re
from typing import Optional
from src.document_loader import Document


def clean_text(text: Optional[str]) -> str:
    """Clean and normalize raw text string.

    Rules applied:
    - Replaces non-printable control characters except standard whitespace.
    - Normalizes carriage returns (`\r\n` -> `\n`, `\r` -> `\n`).
    - Compresses runs of three or more newlines down to double newlines (paragraph breaks).
    - Compresses runs of multiple horizontal spaces and tabs into a single space.
    - Trims leading and trailing whitespace per line and overall.
    - Preserves all standard punctuation, quotes, sentence boundaries, and unicode characters.

    Args:
        text: Raw text string to clean.

    Returns:
        Cleaned and normalized text string. Returns empty string if input is None or empty.
    """
    if not text:
        return ""

    if not isinstance(text, str):
        text = str(text)

    # 1. Strip unwanted control characters (ASCII 0-8, 11-12, 14-31, 127)
    # Retain \t (9) and \n (10)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # 2. Normalize carriage return newlines
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Normalize horizontal whitespace (tabs and spaces) within lines
    lines = cleaned.split("\n")
    normalized_lines = []
    for line in lines:
        # Replace consecutive horizontal whitespace characters with a single space
        norm_line = re.sub(r"[ \t]+", " ", line).strip()
        normalized_lines.append(norm_line)

    cleaned = "\n".join(normalized_lines)

    # 4. Normalize consecutive newlines (max 2 consecutive newlines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def clean_document(doc: Document) -> Document:
    """Clean the text of a Document while preserving and updating its metadata.

    Args:
        doc: Document object to clean.

    Returns:
        New Document instance with cleaned text and updated metadata.
    """
    cleaned_content = clean_text(doc.text)
    new_metadata = dict(doc.metadata)
    new_metadata["original_char_count"] = doc.metadata.get("char_count", len(doc.text))
    new_metadata["cleaned_char_count"] = len(cleaned_content)

    return Document(text=cleaned_content, metadata=new_metadata)

