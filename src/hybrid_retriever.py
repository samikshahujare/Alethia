"""Hybrid retrieval combining dense semantic search and Okapi BM25.

Fuses dense ChromaDB semantic search and sparse BM25 lexical rankings
using Reciprocal Rank Fusion (RRF).
"""

from typing import Any, Dict, List, Optional, Union
from src.bm25 import BM25Index
from src.store import VectorStore, get_vector_store

DEFAULT_RRF_K = 60
DEFAULT_DENSE_WEIGHT = 1.0
DEFAULT_BM25_WEIGHT = 1.0


def calculate_rrf_score(
    dense_rank: Optional[int],
    bm25_rank: Optional[int],
    k_rrf: int = DEFAULT_RRF_K,
    dense_weight: float = DEFAULT_DENSE_WEIGHT,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
) -> float:
    """Calculate Reciprocal Rank Fusion (RRF) score for a document.

    Formula:
        RRF(d) = (dense_weight / (k_rrf + dense_rank)) + (bm25_weight / (k_rrf + bm25_rank))

    Only adds a ranking contribution if the document exists in that ranking
    (1-based rank). Missing ranks contribute 0.0.

    Args:
        dense_rank: 1-based rank in dense retrieval results, or None if not retrieved.
        bm25_rank: 1-based rank in BM25 retrieval results, or None if not retrieved.
        k_rrf: RRF smoothing constant (default: 60).
        dense_weight: Weight multiplier for dense rank (default: 1.0).
        bm25_weight: Weight multiplier for BM25 rank (default: 1.0).

    Returns:
        Combined RRF score as a float rounded to 6 decimal places.
    """
    score = 0.0
    if dense_rank is not None and dense_rank > 0:
        score += float(dense_weight) / float(k_rrf + dense_rank)
    if bm25_rank is not None and bm25_rank > 0:
        score += float(bm25_weight) / float(k_rrf + bm25_rank)
    return round(score, 6)


class HybridRetriever:
    """Orchestrates hybrid sparse-dense retrieval with Reciprocal Rank Fusion."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        bm25_index: Optional[BM25Index] = None,
        k_rrf: int = DEFAULT_RRF_K,
        dense_weight: float = DEFAULT_DENSE_WEIGHT,
        bm25_weight: float = DEFAULT_BM25_WEIGHT,
    ):
        """Initialize HybridRetriever.

        Args:
            vector_store: Optional VectorStore instance. Defaults to persistent singleton.
            bm25_index: Optional BM25Index instance. If None, lazily built from vector_store.
            k_rrf: RRF smoothing constant (default: 60).
            dense_weight: Dense ranking weight (default: 1.0).
            bm25_weight: BM25 ranking weight (default: 1.0).
        """
        self.vector_store = vector_store or get_vector_store()
        self._bm25_index = bm25_index
        self.k_rrf = int(k_rrf)
        self.dense_weight = float(dense_weight)
        self.bm25_weight = float(bm25_weight)

    @property
    def bm25_index(self) -> BM25Index:
        """Retrieve or lazily initialize the synchronized BM25Index."""
        if self._bm25_index is None:
            self._bm25_index = BM25Index.from_chromadb(self.vector_store)
        return self._bm25_index

    def sync_bm25(self) -> int:
        """Synchronize the in-memory BM25 index with the authoritative ChromaDB collection.

        Returns:
            Count of indexed chunks in the refreshed BM25 index.
        """
        self._bm25_index = BM25Index.from_chromadb(self.vector_store)
        return self._bm25_index.doc_count

    def retrieve_dense(
        self,
        query: str,
        k: int = 4,
        source_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Perform dense-only semantic search via ChromaDB vector store.

        Args:
            query: User query string.
            k: Maximum number of chunks to return (default: 4).
            source_filter: Optional document source name to scope search.

        Returns:
            List of chunk dictionaries with dense diagnostics and ranking.
        """
        clean_q = query.strip() if query else ""
        if not clean_q:
            return []

        try:
            if source_filter:
                raw_chunks = self.vector_store.retrieve_chunks(clean_q, k=k, where={"source": source_filter})
            else:
                raw_chunks = self.vector_store.retrieve_chunks(clean_q, k=k)
        except Exception:
            return []

        results: List[Dict[str, Any]] = []
        for rank, chunk in enumerate(raw_chunks, 1):
            dist = chunk.get("distance")
            sim = chunk.get("similarity")
            if dist is not None and sim is None:
                sim = round(max(0.0, 1.0 - float(dist)), 4)

            results.append({
                "chunk_id": chunk.get("chunk_id", ""),
                "text": chunk.get("text", ""),
                "source": chunk.get("source", "unknown"),
                "chunk_index": chunk.get("chunk_index", -1),
                "metadata": chunk.get("metadata", {}),
                "dense_rank": rank,
                "bm25_rank": None,
                "dense_distance": round(float(dist), 4) if dist is not None else None,
                "dense_similarity": round(float(sim), 4) if sim is not None else None,
                "bm25_score": None,
                "rrf_score": calculate_rrf_score(
                    dense_rank=rank,
                    bm25_rank=None,
                    k_rrf=self.k_rrf,
                    dense_weight=self.dense_weight,
                    bm25_weight=self.bm25_weight,
                ),
                "retrieval_strategy": "dense",
            })

        return results

    def retrieve_bm25(
        self,
        query: str,
        k: int = 4,
        source_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Perform sparse-only lexical search via BM25Index.

        Args:
            query: User query string.
            k: Maximum number of chunks to return (default: 4).
            source_filter: Optional document source name to scope search.

        Returns:
            List of chunk dictionaries with BM25 diagnostics and ranking.
        """
        clean_q = query.strip() if query else ""
        if not clean_q:
            return []

        if source_filter:
            raw_chunks = self.bm25_index.search(clean_q, k=k, source_filter=source_filter)
        else:
            raw_chunks = self.bm25_index.search(clean_q, k=k)

        results: List[Dict[str, Any]] = []
        for rank, chunk in enumerate(raw_chunks, 1):
            results.append({
                "chunk_id": chunk.get("chunk_id", ""),
                "text": chunk.get("text", ""),
                "source": chunk.get("source", "unknown"),
                "chunk_index": chunk.get("chunk_index", -1),
                "metadata": chunk.get("metadata", {}),
                "dense_rank": None,
                "bm25_rank": rank,
                "dense_distance": None,
                "dense_similarity": None,
                "bm25_score": chunk.get("bm25_score"),
                "rrf_score": calculate_rrf_score(
                    dense_rank=None,
                    bm25_rank=rank,
                    k_rrf=self.k_rrf,
                    dense_weight=self.dense_weight,
                    bm25_weight=self.bm25_weight,
                ),
                "retrieval_strategy": "bm25",
            })

        return results

    def retrieve_hybrid(
        self,
        query: str,
        k: int = 4,
        candidate_depth: Optional[int] = None,
        source_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining dense semantic search and BM25 via RRF.

        Steps:
        1. Query dense ChromaDB store for top candidate chunks (optionally scoped to source).
        2. Query sparse BM25 index for top candidate chunks (optionally scoped to source).
        3. Merge candidate sets by chunk_id, deduplicating records.
        4. Calculate RRF score based strictly on ordinal ranks:
           RRF(d) = dense_weight/(60 + dense_rank) + bm25_weight/(60 + bm25_rank)
        5. Sort by RRF score descending, breaking ties deterministically by chunk_id ascending.
        6. Return top-k results with complete diagnostic metadata.

        Args:
            query: User query string.
            k: Number of final results to return (default: 4).
            candidate_depth: Number of candidates to fetch from each retriever
                             before fusion. Defaults to max(k * 3, 20).
            source_filter: Optional document source name to scope search.

        Returns:
            List of top-k fused chunk dictionaries.
        """
        clean_q = query.strip() if query else ""
        if not clean_q:
            return []

        # Candidate depth bounds candidate pool for fusion
        pool_size = candidate_depth or max(k * 3, 20)

        dense_results: List[Dict[str, Any]] = []
        try:
            if source_filter:
                dense_results = self.vector_store.retrieve_chunks(clean_q, k=pool_size, where={"source": source_filter})
            else:
                dense_results = self.vector_store.retrieve_chunks(clean_q, k=pool_size)
        except Exception:
            dense_results = []

        if source_filter:
            bm25_results = self.bm25_index.search(clean_q, k=pool_size, source_filter=source_filter)
        else:
            bm25_results = self.bm25_index.search(clean_q, k=pool_size)

        if not dense_results and not bm25_results:
            return []

        # Candidate map: chunk_id -> unified record
        candidates: Dict[str, Dict[str, Any]] = {}

        for rank, chunk in enumerate(dense_results, 1):
            cid = chunk.get("chunk_id", "")
            if not cid:
                continue

            dist = chunk.get("distance")
            sim = chunk.get("similarity")
            if dist is not None and sim is None:
                sim = round(max(0.0, 1.0 - float(dist)), 4)

            candidates[cid] = {
                "chunk_id": cid,
                "text": chunk.get("text", ""),
                "source": chunk.get("source", "unknown"),
                "chunk_index": chunk.get("chunk_index", -1),
                "metadata": chunk.get("metadata", {}),
                "dense_rank": rank,
                "bm25_rank": None,
                "dense_distance": round(float(dist), 4) if dist is not None else None,
                "dense_similarity": round(float(sim), 4) if sim is not None else None,
                "bm25_score": None,
            }

        for rank, chunk in enumerate(bm25_results, 1):
            cid = chunk.get("chunk_id", "")
            if not cid:
                continue

            if cid in candidates:
                # Merge BM25 rank and score into existing dense candidate
                candidates[cid]["bm25_rank"] = rank
                candidates[cid]["bm25_score"] = chunk.get("bm25_score")
            else:
                # Document retrieved by BM25 only
                candidates[cid] = {
                    "chunk_id": cid,
                    "text": chunk.get("text", ""),
                    "source": chunk.get("source", "unknown"),
                    "chunk_index": chunk.get("chunk_index", -1),
                    "metadata": chunk.get("metadata", {}),
                    "dense_rank": None,
                    "bm25_rank": rank,
                    "dense_distance": None,
                    "dense_similarity": None,
                    "bm25_score": chunk.get("bm25_score"),
                }

        for cid, record in candidates.items():
            record["rrf_score"] = calculate_rrf_score(
                dense_rank=record["dense_rank"],
                bm25_rank=record["bm25_rank"],
                k_rrf=self.k_rrf,
                dense_weight=self.dense_weight,
                bm25_weight=self.bm25_weight,
            )
            record["retrieval_strategy"] = "hybrid_rrf"

        sorted_candidates = sorted(
            candidates.values(),
            key=lambda c: (-c["rrf_score"], c["chunk_id"]),
        )

        return sorted_candidates[:k]

    def retrieve(
        self,
        query: str,
        k: int = 4,
        strategy: str = "hybrid",
        source_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Unified retrieval entry point supporting multiple strategies.

        Args:
            query: User query string.
            k: Maximum number of chunks to return (default: 4).
            strategy: One of 'hybrid' (default), 'dense', or 'bm25'.
            source_filter: Optional source document name to filter retrieval by.

        Returns:
            List of retrieved chunk dictionaries.
        """
        strat = (strategy or "hybrid").lower().strip()
        if strat in ("dense", "semantic"):
            return self.retrieve_dense(query=query, k=k, source_filter=source_filter)
        if strat in ("bm25", "sparse"):
            return self.retrieve_bm25(query=query, k=k, source_filter=source_filter)
        return self.retrieve_hybrid(query=query, k=k, source_filter=source_filter)


_DEFAULT_HYBRID_RETRIEVER: Optional[HybridRetriever] = None


def get_hybrid_retriever(
    vector_store: Optional[VectorStore] = None,
    bm25_index: Optional[BM25Index] = None,
    k_rrf: int = DEFAULT_RRF_K,
    dense_weight: float = DEFAULT_DENSE_WEIGHT,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
) -> HybridRetriever:
    """Retrieve or create the default HybridRetriever singleton."""
    global _DEFAULT_HYBRID_RETRIEVER
    if _DEFAULT_HYBRID_RETRIEVER is None or vector_store is not None or bm25_index is not None:
        retriever = HybridRetriever(
            vector_store=vector_store,
            bm25_index=bm25_index,
            k_rrf=k_rrf,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
        )
        if vector_store is None and bm25_index is None:
            _DEFAULT_HYBRID_RETRIEVER = retriever
        return retriever
    return _DEFAULT_HYBRID_RETRIEVER


def hybrid_retrieve(
    query: str,
    k: int = 4,
    strategy: str = "hybrid",
    retriever: Optional[HybridRetriever] = None,
    source_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Convenience function to perform hybrid or single-strategy retrieval."""
    active_retriever = retriever or get_hybrid_retriever()
    return active_retriever.retrieve(query=query, k=k, strategy=strategy, source_filter=source_filter)

