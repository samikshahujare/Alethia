"""Index all documents from data/ into the persistent ChromaDB collection."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.document_loader import load_documents_from_directory
from src.text_cleaner import clean_document
from src.chunker import chunk_document
from src.store import get_vector_store


def index_corpus():
    store = get_vector_store()
    print(f"Current chunks before indexing: {store.count()}")

    docs = load_documents_from_directory("data")
    print(f"Loaded {len(docs)} documents from data/")

    total_chunks = 0
    newly_indexed = 0

    for doc in docs:
        cleaned = clean_document(doc)
        chunks = chunk_document(cleaned)
        total_chunks += len(chunks)
        added = store.index_chunks(chunks, overwrite=False)
        newly_indexed += added
        print(f"  {doc.source}: {len(chunks)} chunks generated, {added} newly indexed")

    print(f"\nTotal chunks in ChromaDB now: {store.count()} (Newly indexed: {newly_indexed})")


if __name__ == "__main__":
    index_corpus()

