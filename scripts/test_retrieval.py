"""Phase 2 Retrieval Demonstration Script.

Demonstrates:
1. Loading the fixed corpus from data/transformer_summary.txt.
2. Cleaning the text using the text cleaning module.
3. Splitting the document into sentence-boundary-aware chunks.
4. Generating dense embeddings using 'all-MiniLM-L6-v2'.
5. Persisting chunks into ChromaDB.
6. Executing semantic similarity search for a sample query without calling any LLM.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.document_loader import load_document
from src.text_cleaner import clean_document
from src.chunker import chunk_document
from src.store import VectorStore


def run_demo() -> None:
    corpus_path = PROJECT_ROOT / "data" / "transformer_summary.txt"
    db_path = PROJECT_ROOT / "chroma_db"

    print("=" * 75)
    print("ALETHIA PHASE 2: DOCUMENT PROCESSING & SEMANTIC RETRIEVAL DEMO")
    print("=" * 75)

    # 1. Document Loading
    print(f"\n[1] Loading document from: {corpus_path.relative_to(PROJECT_ROOT)}")
    raw_doc = load_document(corpus_path)
    print(f"    Loaded source: {raw_doc.source} ({len(raw_doc.text)} characters)")

    # 2. Text Cleaning
    print("\n[2] Cleaning and normalizing text...")
    cleaned_doc = clean_document(raw_doc)
    print(f"    Original character count: {cleaned_doc.metadata.get('original_char_count')}")
    print(f"    Cleaned character count:  {cleaned_doc.metadata.get('cleaned_char_count')}")

    # 3. Chunking
    print("\n[3] Splitting into sentence-aware chunks (target size: 400 chars, overlap: 60 chars)...")
    chunks = chunk_document(cleaned_doc, chunk_size=400, chunk_overlap=60)
    print(f"    Total chunks generated: {len(chunks)}")
    for c in chunks:
        print(f"    - Chunk ID: {c.chunk_id:<28} Length: {len(c.text):>3} chars | Start: {c.text[:45]}...")

    # 4. ChromaDB Storage & Embeddings
    print(f"\n[4] Initializing ChromaDB vector store at: {db_path.relative_to(PROJECT_ROOT)}")
    store = VectorStore(persist_directory=db_path, collection_name="alethia_documents")
    indexed_new = store.index_chunks(chunks, overwrite=True)
    print(f"    New chunks indexed:      {indexed_new}")
    print(f"    Total chunks in store:   {store.count()}")

    # 5. Semantic Retrieval
    sample_question = "How many encoder and decoder layers does the Transformer use?"
    k = 4
    print(f"\n[5] Executing semantic search for query:")
    print(f'    "{sample_question}"')
    print(f"    Requesting top {k} relevant chunks...\n")

    results = store.retrieve_chunks(sample_question, k=k)

    print("-" * 75)
    print(f"RETRIEVAL RESULTS ({len(results)} chunks returned):")
    print("-" * 75)

    for rank, res in enumerate(results, 1):
        print(f"\n--- Rank #{rank} ---")
        print(f"Chunk ID:          {res['chunk_id']}")
        print(f"Source Filename:   {res['source']}")
        print(f"Chunk Index:       {res['chunk_index']}")
        print(f"Cosine Similarity: {res['similarity']:.4f}  (Distance: {res['distance']:.4f})")
        print("Text Content:")
        print(f'"{res["text"]}"')

    print("\n" + "=" * 75)
    print("PHASE 2 RETRIEVAL DEMO COMPLETE (No LLM called)")
    print("=" * 75)


if __name__ == "__main__":
    run_demo()

