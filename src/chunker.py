"""Text chunking module for Alethia.

Splits cleaned text into semantically cohesive, paragraph- and sentence-boundary-aware chunks
with configurable character sizes, overlap, and deterministic unique IDs.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from src.document_loader import Document


@dataclass
class TextChunk:
    """Represents an indexed text chunk with metadata."""

    chunk_id: str
    text: str
    source: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary format."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences while respecting common sentence terminators.

    Uses regex lookbehind for punctuation followed by whitespace.
    """
    if not text:
        return []

    sentence_pattern = r"(?<=[.!?])\s+"
    sentences = [s.strip() for s in re.split(sentence_pattern, text) if s.strip()]
    return sentences if sentences else [text.strip()]


def _split_long_sentence(sentence: str, max_size: int) -> List[str]:
    """Fallback to split an unusually long sentence by words if it exceeds chunk_size."""
    words = sentence.split()
    parts: List[str] = []
    current_part: List[str] = []
    current_len = 0

    for word in words:
        word_len = len(word) + (1 if current_part else 0)
        if current_len + word_len > max_size and current_part:
            parts.append(" ".join(current_part))
            current_part = [word]
            current_len = len(word)
        else:
            current_part.append(word)
            current_len += word_len

    if current_part:
        parts.append(" ".join(current_part))

    return parts


def _chunk_single_segment(
    sentences: List[str],
    source_clean: str,
    source: str,
    chunk_size: int,
    chunk_overlap: int,
    start_index: int,
    meta: Dict[str, Any],
) -> List[TextChunk]:
    """Chunk a sequence of sentences within a single paragraph or continuous block."""
    chunks: List[TextChunk] = []
    current_sentences: List[str] = []
    current_length = 0

    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_len = len(sentence) + (1 if current_sentences else 0)

        if current_length + sentence_len <= chunk_size or not current_sentences:
            current_sentences.append(sentence)
            current_length += sentence_len
            i += 1
        else:
            chunk_content = " ".join(current_sentences).strip()
            if chunk_content:
                chunk_idx = start_index + len(chunks)
                chunk_id = f"{source_clean}_chunk_{chunk_idx:04d}"
                chunk_meta = dict(meta)
                chunk_meta.update({
                    "source": source,
                    "chunk_index": chunk_idx,
                    "char_count": len(chunk_content),
                })
                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        text=chunk_content,
                        source=source,
                        chunk_index=chunk_idx,
                        metadata=chunk_meta,
                    )
                )

            overlap_sentences: List[str] = []
            overlap_len = 0
            for prev_s in reversed(current_sentences):
                added_len = len(prev_s) + (1 if overlap_sentences else 0)
                if overlap_len + added_len <= chunk_overlap:
                    overlap_sentences.insert(0, prev_s)
                    overlap_len += added_len
                else:
                    break

            current_sentences = overlap_sentences
            current_length = overlap_len

    if current_sentences:
        chunk_content = " ".join(current_sentences).strip()
        if chunk_content:
            chunk_idx = start_index + len(chunks)
            chunk_id = f"{source_clean}_chunk_{chunk_idx:04d}"
            chunk_meta = dict(meta)
            chunk_meta.update({
                "source": source,
                "chunk_index": chunk_idx,
                "char_count": len(chunk_content),
            })
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    text=chunk_content,
                    source=source,
                    chunk_index=chunk_idx,
                    metadata=chunk_meta,
                )
            )

    return chunks


def chunk_text(
    text: str,
    source: str = "unknown",
    chunk_size: int = 400,
    chunk_overlap: int = 60,
    base_metadata: Optional[Dict[str, Any]] = None,
) -> List[TextChunk]:
    """Split text into paragraph- and sentence-aware chunks of approximately `chunk_size` characters.

    Respects paragraph breaks (`\\n\\n`) to preserve semantic coherence and topic continuity,
    while applying sentence boundaries and overlap within larger passages.

    Args:
        text: Raw or cleaned text to chunk.
        source: Source document filename or identifier.
        chunk_size: Target maximum characters per chunk (default: 400).
        chunk_overlap: Approximate character overlap to preserve inter-chunk context (default: 60).
        base_metadata: Optional dictionary of metadata from parent Document to attach.

    Returns:
        List of TextChunk instances with unique chunk IDs and index numbering.
    """
    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and less than chunk_size")

    source_clean = re.sub(r"[^a-zA-Z0-9_-]", "_", source).strip("_")
    meta = dict(base_metadata or {})

    # Split into paragraphs first to honor discourse structure
    paragraphs = [p.strip() for p in text.strip().split("\n\n") if p.strip()]

    # If there are no multiple paragraphs, handle as a single block
    if len(paragraphs) <= 1:
        raw_sentences = _split_into_sentences(text.strip())
        sentences: List[str] = []
        for s in raw_sentences:
            if len(s) > chunk_size:
                sentences.extend(_split_long_sentence(s, chunk_size))
            else:
                sentences.append(s)
        return _chunk_single_segment(
            sentences=sentences,
            source_clean=source_clean,
            source=source,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            start_index=0,
            meta=meta,
        )

    # For multi-paragraph texts: chunk each paragraph respecting boundaries
    all_chunks: List[TextChunk] = []
    for para in paragraphs:
        raw_sentences = _split_into_sentences(para)
        sentences = []
        for s in raw_sentences:
            if len(s) > chunk_size:
                sentences.extend(_split_long_sentence(s, chunk_size))
            else:
                sentences.append(s)

        para_chunks = _chunk_single_segment(
            sentences=sentences,
            source_clean=source_clean,
            source=source,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            start_index=len(all_chunks),
            meta=meta,
        )
        all_chunks.extend(para_chunks)

    return all_chunks


def chunk_document(
    doc: Document,
    chunk_size: int = 400,
    chunk_overlap: int = 60,
) -> List[TextChunk]:
    """Chunk a Document instance preserving its metadata and source information."""
    return chunk_text(
        text=doc.text,
        source=doc.source,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        base_metadata=doc.metadata,
    )
