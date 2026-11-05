"""ChromaDB vector store and embedding generation for Alethia."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import hashlib
import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

from src.chunker import TextChunk

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_COLLECTION_NAME = "alethia_documents"
DEFAULT_DB_DIR = Path(__file__).resolve().parent.parent / "chroma_db"

# Global cache for the embedding model to avoid reloading weights repeatedly
_CACHED_EMBEDDER: Optional[SentenceTransformer] = None


def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    """Retrieve or initialize the SentenceTransformer embedding model."""
    global _CACHED_EMBEDDER
    if _CACHED_EMBEDDER is None:
        try:
            _CACHED_EMBEDDER = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            _CACHED_EMBEDDER = SentenceTransformer(model_name)
    return _CACHED_EMBEDDER


class VectorStore:
    """Manages persistent ChromaDB vector storage and semantic retrieval."""

    def __init__(
        self,
        persist_directory: Optional[Union[str, Path]] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
    ):
        """Initialize ChromaDB client with persistent disk storage.

        Args:
            persist_directory: Directory for ChromaDB persistence. Defaults to project root 'chroma_db'.
            collection_name: Name of the ChromaDB collection.
            model_name: Name of the SentenceTransformer model.
        """
        self.persist_directory = Path(persist_directory or DEFAULT_DB_DIR).resolve()
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.model_name = model_name

        self.client: ClientAPI = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection: Collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = get_embedding_model(model_name)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate normalized vector embeddings for a list of text strings.

        Args:
            texts: List of text strings.

        Returns:
            List of embedding vectors (list of floats).
        """
        if not texts:
            return []
        try:
            embeddings = self.embedder.encode(texts, normalize_embeddings=True)
            return embeddings.tolist()
        except Exception as e:
            raise RuntimeError(f"Embedding generation failed: {e}") from e

    def index_chunks(self, chunks: List[TextChunk], overwrite: bool = False) -> int:
        """Store document chunks in ChromaDB with intelligent duplicate and content-change detection.

        If a document with the same source name already exists:
        - If the content has changed (different chunk count, altered text, or changed hash):
          safely removes outdated chunks for only that source, regenerates embeddings, and re-indexes.
        - If the content is identical and overwrite=False:
          safely skips re-indexing to prevent duplicate embeddings.
        - If overwrite=True:
          always replaces existing chunks for that source.

        Args:
            chunks: List of TextChunk objects to index.
            overwrite: If True, always replaces existing document chunks.
                       If False (default), updates changed documents and skips unchanged duplicates.

        Returns:
            Count of newly indexed or re-indexed chunks.
        """
        if not chunks:
            return 0

        chunks_by_source: Dict[str, List[TextChunk]] = {}
        for chunk in chunks:
            src = str(chunk.source)
            if src not in chunks_by_source:
                chunks_by_source[src] = []
            chunks_by_source[src].append(chunk)

        chunks_to_index: List[TextChunk] = []

        for src, doc_chunks in chunks_by_source.items():
            incoming_hash = hashlib.sha256("".join(c.text for c in doc_chunks).encode("utf-8")).hexdigest()[:16]

            for c in doc_chunks:
                if not hasattr(c, "metadata") or c.metadata is None:
                    c.metadata = {}
                c.metadata["doc_hash"] = incoming_hash

            if overwrite:
                self.delete_document(src)
                chunks_to_index.extend(doc_chunks)
                continue

            try:
                existing_records = self.collection.get(
                    where={"source": src},
                    include=["documents", "metadatas"],
                )
            except Exception:
                existing_records = {"ids": [], "documents": [], "metadatas": []}

            existing_ids = existing_records.get("ids", [])
            existing_docs = existing_records.get("documents", [])
            existing_metas = existing_records.get("metadatas", [])

            if not existing_ids:
                chunks_to_index.extend(doc_chunks)
            else:
                existing_hash = None
                if existing_metas and isinstance(existing_metas[0], dict):
                    existing_hash = existing_metas[0].get("doc_hash")

                content_changed = False
                if existing_hash and existing_hash != incoming_hash:
                    content_changed = True
                elif len(existing_ids) != len(doc_chunks):
                    content_changed = True
                else:
                    indexed_pairs = []
                    for doc, meta in zip(existing_docs, existing_metas):
                        c_idx = meta.get("chunk_index", 0) if isinstance(meta, dict) else 0
                        indexed_pairs.append((c_idx, doc))
                    indexed_pairs.sort(key=lambda x: x[0])
                    existing_full_text = "".join(doc for _, doc in indexed_pairs)
                    incoming_full_text = "".join(c.text for c in doc_chunks)
                    if existing_full_text != incoming_full_text:
                        content_changed = True

                if content_changed:
                    self.delete_document(src)
                    chunks_to_index.extend(doc_chunks)
                else:
                    pass

        if not chunks_to_index:
            return 0

        texts = [chunk.text for chunk in chunks_to_index]
        chunk_ids = [chunk.chunk_id for chunk in chunks_to_index]

        metadatas = []
        for chunk in chunks_to_index:
            meta: Dict[str, Any] = {
                "source": str(chunk.source),
                "chunk_index": int(chunk.chunk_index),
                "char_count": int(len(chunk.text)),
                "doc_hash": chunk.metadata.get("doc_hash", ""),
            }
            for k, v in chunk.metadata.items():
                if k not in meta and isinstance(v, (str, int, float, bool)):
                    meta[k] = v
            metadatas.append(meta)

        embeddings = self.generate_embeddings(texts)

        self.collection.upsert(
            ids=chunk_ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(chunks_to_index)

    def count(self) -> int:
        """Return total number of chunks currently stored in the collection."""
        return self.collection.count()

    def delete_document(self, source: str) -> int:
        """Delete all indexed chunks associated with a specific document source.

        Args:
            source: Source filename or identifier.

        Returns:
            Number of chunks removed.
        """
        if not source:
            return 0
        try:
            records = self.collection.get(where={"source": source})
            ids = records.get("ids", [])
            if ids:
                self.collection.delete(ids=ids)
            return len(ids)
        except Exception:
            return 0

    def reindex_document(self, chunks: List[TextChunk]) -> int:
        """Safely re-index a document by removing outdated chunks and inserting new chunks.

        Prevents orphaned chunks if the document was edited to be shorter.

        Args:
            chunks: New list of TextChunk objects for the document.

        Returns:
            Number of newly indexed chunks.
        """
        if not chunks:
            return 0
        source = chunks[0].source
        self.delete_document(source)
        return self.index_chunks(chunks, overwrite=False)

    def clear(self) -> None:
        """Clear all records from the collection."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def close(self) -> None:
        """Release underlying collection and client resources."""
        self.collection = None
        self.client = None

    def retrieve_chunks(
        self,
        question: str,
        k: int = 4,
        where: Optional[Dict[str, Any]] = None,
        source_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve the top k most relevant chunks for a question.

        Args:
            question: User query string.
            k: Number of chunks to retrieve (default: 4).
            where: Optional ChromaDB metadata filter dictionary (e.g. {"source": "doc.txt"}).
            source_filter: Optional source document name to filter results by.

        Returns:
            List of retrieved chunk dictionaries containing chunk_id, text,
            source, chunk_index, distance, similarity, and metadata.

        Raises:
            ValueError: If the query is empty or whitespace only.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty or whitespace only.")

        total_chunks = self.count()
        if total_chunks == 0:
            return []

        n_results = min(k, total_chunks)
        if n_results <= 0:
            return []

        query_embedding = self.generate_embeddings([question.strip()])[0]

        query_kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        filter_dict = where
        if source_filter and not filter_dict:
            filter_dict = {"source": source_filter}
        if filter_dict:
            query_kwargs["where"] = filter_dict

        results = self.collection.query(**query_kwargs)

        retrieved_items: List[Dict[str, Any]] = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        docs = results["documents"][0] if results.get("documents") else []
        metas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        for i, chunk_id in enumerate(ids):
            text = docs[i] if i < len(docs) else ""
            meta = metas[i] if i < len(metas) else {}
            distance = float(distances[i]) if i < len(distances) else 0.0

            similarity = round(max(0.0, 1.0 - distance), 4)

            retrieved_items.append({
                "chunk_id": chunk_id,
                "text": text,
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index", -1),
                "distance": round(distance, 4),
                "similarity": similarity,
                "metadata": meta,
            })

        return retrieved_items


_DEFAULT_STORE: Optional[VectorStore] = None


def get_vector_store(
    persist_directory: Optional[Union[str, Path]] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> VectorStore:
    """Retrieve or create the default VectorStore instance."""
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None or persist_directory is not None:
        store = VectorStore(
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
        if persist_directory is None:
            _DEFAULT_STORE = store
        return store
    return _DEFAULT_STORE


def retrieve_chunks(
    question: str,
    k: int = 4,
    store: Optional[VectorStore] = None,
    where: Optional[Dict[str, Any]] = None,
    source_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Convenience function to retrieve chunks using default or custom VectorStore."""
    target_store = store or get_vector_store()
    return target_store.retrieve_chunks(question=question, k=k, where=where, source_filter=source_filter)
