"""Semantic retriever module for Alethia.

Provides semantic similarity search over indexed chunks stored in ChromaDB
independent of any LLM or generation model.
"""

from typing import Any, Dict, List, Optional
from src.store import VectorStore, get_vector_store, retrieve_chunks
from src.hybrid_retriever import hybrid_retrieve, get_hybrid_retriever, HybridRetriever
from src.decomposition import decompose_question, retrieve_decomposed


def retrieve(
    question: str,
    k: int = 4,
    store: Optional[VectorStore] = None,
    strategy: str = "dense",
) -> List[Dict[str, Any]]:
    """Retrieve top-k relevant chunks for a question.

    Args:
        question: User query string.
        k: Maximum number of chunks to return (default: 4).
        store: Optional VectorStore instance.
        strategy: Retrieval strategy ('dense' [default for backward compatibility], 'hybrid', or 'bm25').

    Returns:
        List of retrieved chunk dictionaries.
    """
    if strategy == "dense":
        return retrieve_chunks(question=question, k=k, store=store)
    return hybrid_retrieve(query=question, k=k, strategy=strategy)


__all__ = [
    "retrieve",
    "retrieve_chunks",
    "hybrid_retrieve",
    "get_hybrid_retriever",
    "HybridRetriever",
    "decompose_question",
    "retrieve_decomposed",
]

