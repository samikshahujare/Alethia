"""Okapi BM25 implementation for Alethia.

Implements standard Okapi BM25 lexical search over chunked documents
synchronized with the vector store.
"""

from collections import Counter, defaultdict
import math
import re
from typing import Any, Dict, List, Optional, Set

# Common question words and function stopwords for search filtering
QUESTION_STOPWORDS: Set[str] = {
    "what", "is", "are", "the", "a", "an", "in", "of", "to", "for", "and", "or",
    "does", "do", "did", "how", "why", "which", "where", "when", "who", "whom",
    "this", "that", "these", "those",
}


def tokenize(text: str) -> List[str]:
    """Tokenize and normalize text for BM25 lexical indexing and search.

    Normalizes to lowercase, extracts alphanumeric words and decimals,
    normalizes plurals (e.g. 'sops' -> 'sop', 'optimizers' -> 'optimizer'),
    and indexes sub-tokens for compound underscore terms (e.g. 'beta_1' -> 'beta_1', 'beta', '1').

    Args:
        text: Input text string.

    Returns:
        List of normalized string tokens.
    """
    if not text or not text.strip():
        return []

    # Match alphanumeric words, compound underscore identifiers, and decimal numbers
    raw_tokens = re.findall(r"\b[a-zA-Z0-9_]+(?:\.[0-9]+)?\b", text.lower())
    tokens: List[str] = []

    for t in raw_tokens:
        tokens.append(t)
        # Lightweight plural/singular normalization (e.g. 'sops' -> 'sop')
        if len(t) >= 4 and t.endswith("s") and not t.endswith("ss"):
            tokens.append(t[:-1])
        # If token contains underscores, also yield sub-parts to support partial matching
        if "_" in t:
            for sub in t.split("_"):
                if sub:
                    tokens.append(sub)
                    if len(sub) >= 4 and sub.endswith("s") and not sub.endswith("ss"):
                        tokens.append(sub[:-1])

    return tokens


class BM25Index:
    """Okapi BM25 inverted index with document-length normalization."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """Initialize empty BM25 index.

        Args:
            k1: Term frequency saturation parameter (default: 1.5).
            b: Document length normalization parameter (default: 0.75).
        """
        self.k1 = float(k1)
        self.b = float(b)

        # Storage for indexed documents
        self.doc_count: int = 0
        self.avg_doc_len: float = 0.0
        self.total_tokens: int = 0

        # Document registry: chunk_id -> raw document record
        self.documents: Dict[str, Dict[str, Any]] = {}
        # Document token lengths: chunk_id -> int
        self.doc_lengths: Dict[str, int] = {}
        # Document term frequencies: chunk_id -> Counter(token -> int)
        self.doc_term_freqs: Dict[str, Counter] = {}
        # Inverted index document frequencies: token -> set(chunk_ids)
        self.term_docs: Dict[str, Set[str]] = defaultdict(set)
        # Precomputed IDF cache: token -> float
        self._idf_cache: Dict[str, float] = {}

    def clear(self) -> None:
        """Clear all indexed documents and reset statistics."""
        self.doc_count = 0
        self.avg_doc_len = 0.0
        self.total_tokens = 0
        self.documents.clear()
        self.doc_lengths.clear()
        self.doc_term_freqs.clear()
        self.term_docs.clear()
        self._idf_cache.clear()

    def idf(self, term: str) -> float:
        """Calculate Robertson-Sparck Jones BM25 Inverse Document Frequency.

        Formula:
            IDF(q_i) = ln(1 + (N - n(q_i) + 0.5) / (n(q_i) + 0.5))

        This formulation guarantees strictly positive IDF values for all terms
        present in the corpus, preventing negative weights for common terms.

        Args:
            term: Normalized query token.

        Returns:
            Positive float IDF score, or 0.0 if term is not in vocabulary.
        """
        if term in self._idf_cache:
            return self._idf_cache[term]

        n_q = len(self.term_docs.get(term, ()))
        if n_q == 0 or self.doc_count == 0:
            return 0.0

        n = float(self.doc_count)
        # Robertson-Sparck Jones formula with +1 smoothing
        val = math.log(1.0 + (n - float(n_q) + 0.5) / (float(n_q) + 0.5))
        self._idf_cache[term] = val
        return val

    def index_documents(self, documents: List[Dict[str, Any]]) -> int:
        """Index a list of document chunk dictionaries.

        Expected keys per document:
            - chunk_id (str): Unique chunk identifier.
            - text (str): Text content.
            - source (str, optional): Source document identifier.
            - chunk_index (int, optional): Order within document.
            - metadata (dict, optional): Additional chunk metadata.

        Args:
            documents: List of document dictionaries.

        Returns:
            Total count of indexed documents.
        """
        if not documents:
            return self.doc_count

        self._idf_cache.clear()

        for doc in documents:
            chunk_id = str(doc.get("chunk_id", "")).strip()
            if not chunk_id:
                continue

            text = str(doc.get("text", ""))
            tokens = tokenize(text)
            doc_len = len(tokens)

            # Store document record
            self.documents[chunk_id] = {
                "chunk_id": chunk_id,
                "text": text,
                "source": doc.get("source", "unknown"),
                "chunk_index": doc.get("chunk_index", -1),
                "metadata": doc.get("metadata", {}),
            }

            self.doc_lengths[chunk_id] = doc_len
            tf = Counter(tokens)
            self.doc_term_freqs[chunk_id] = tf

            # Update inverted index
            for term in tf.keys():
                self.term_docs[term].add(chunk_id)

        self.doc_count = len(self.documents)
        self.total_tokens = sum(self.doc_lengths.values())
        self.avg_doc_len = (self.total_tokens / self.doc_count) if self.doc_count > 0 else 0.0

        return self.doc_count

    def search(
        self,
        query: str,
        k: int = 4,
        source_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search the indexed corpus for a query using Okapi BM25 scoring.

        Args:
            query: Query string.
            k: Maximum number of results to return (default: 4).
            source_filter: Optional source document name to filter results by.

        Returns:
            List of result dictionaries sorted by BM25 score descending,
            with chunk_id ascending as deterministic tie-breaker.
            Only returns documents with score > 0.0.
        """
        if not query or not query.strip():
            return []

        if self.doc_count == 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # Focus candidate retrieval on content tokens when available to avoid stopword swamping
        content_tokens = [qt for qt in query_tokens if qt not in QUESTION_STOPWORDS]
        candidate_search_tokens = content_tokens if content_tokens else query_tokens

        # Find candidate documents that contain at least one content query term
        candidate_ids: Set[str] = set()
        for qt in candidate_search_tokens:
            candidate_ids.update(self.term_docs.get(qt, ()))

        if not candidate_ids:
            return []

        # Filter by source if requested
        if source_filter:
            candidate_ids = {
                cid for cid in candidate_ids
                if self.documents.get(cid, {}).get("source") == source_filter
            }
            if not candidate_ids:
                return []

        scores: Dict[str, float] = defaultdict(float)

        # Precompute query term IDF values
        query_idfs = {qt: self.idf(qt) for qt in set(query_tokens)}

        for chunk_id in candidate_ids:
            doc_len = float(self.doc_lengths.get(chunk_id, 0))
            tf_map = self.doc_term_freqs.get(chunk_id, {})

            # Document length normalization factor B
            denom_norm = 1.0 - self.b + self.b * (doc_len / self.avg_doc_len if self.avg_doc_len > 0 else 1.0)

            score = 0.0
            for qt in query_tokens:
                f = float(tf_map.get(qt, 0))
                if f > 0.0:
                    idf_val = query_idfs.get(qt, 0.0)
                    # Standard BM25 term weighting
                    tf_component = (f * (self.k1 + 1.0)) / (f + self.k1 * denom_norm)
                    score += idf_val * tf_component

            if score > 0.0:
                scores[chunk_id] = score

        if not scores:
            return []

        # Sort candidate documents: score descending, then chunk_id ascending for deterministic tie-breaking
        sorted_candidates = sorted(
            scores.items(),
            key=lambda item: (-item[1], item[0]),
        )

        results: List[Dict[str, Any]] = []
        for chunk_id, score in sorted_candidates[:k]:
            doc_info = self.documents[chunk_id]
            results.append({
                "chunk_id": chunk_id,
                "text": doc_info["text"],
                "source": doc_info["source"],
                "chunk_index": doc_info["chunk_index"],
                "bm25_score": round(score, 4),
                "metadata": doc_info["metadata"],
            })

        return results

    @classmethod
    def from_documents(
        cls,
        documents: List[Dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> "BM25Index":
        """Factory method to construct a populated BM25Index from document dicts.

        Args:
            documents: List of document chunk dictionaries.
            k1: BM25 k1 parameter.
            b: BM25 b parameter.

        Returns:
            Populated BM25Index instance.
        """
        index = cls(k1=k1, b=b)
        index.index_documents(documents)
        return index

    @classmethod
    def from_chromadb(
        cls,
        store: Optional[Any] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> "BM25Index":
        """Factory method to synchronize BM25Index with the authoritative ChromaDB collection.

        Reads indexed documents and metadata from ChromaDB without altering,
        overwriting, or deleting any ChromaDB data.

        Args:
            store: Optional VectorStore instance (defaults to get_vector_store()).
            k1: BM25 k1 parameter.
            b: BM25 b parameter.

        Returns:
            Populated BM25Index instance synchronized with ChromaDB.
        """
        from src.store import get_vector_store

        target_store = store or get_vector_store()
        records = target_store.collection.get(include=["documents", "metadatas"])

        documents: List[Dict[str, Any]] = []
        ids = records.get("ids", [])
        docs = records.get("documents", [])
        metas = records.get("metadatas", [])

        for i, chunk_id in enumerate(ids):
            text = docs[i] if i < len(docs) else ""
            meta = metas[i] if i < len(metas) else {}
            documents.append({
                "chunk_id": chunk_id,
                "text": text,
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index", -1),
                "metadata": meta,
            })

        return cls.from_documents(documents, k1=k1, b=b)

