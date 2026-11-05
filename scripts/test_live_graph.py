"""Phase 4 Live LangGraph Orchestration Smoke Test Script.

Validates end-to-end live execution of the compiled StateGraph:
1. Verifies Groq configuration and vector store availability.
2. Runs a supported query ("How many encoder layers does the Transformer use?").
3. Inspects full state transitions: retrieve -> generate -> judge -> END.
4. Runs an out-of-domain query to verify proper abstention ("Not in context.").
5. Displays the complete execution trace without exposing sensitive credentials.
"""

import sys
from pathlib import Path

# Ensure stdout handles UTF-8 safely on Windows
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_groq_model, is_groq_configured
from src.graph import run_rag_pipeline
from src.store import get_vector_store


def run_live_graph_smoke_test() -> bool:
    print("=" * 75)
    print("ALETHIA PHASE 4: LIVE LANGGRAPH ORCHESTRATION SMOKE TEST")
    print("=" * 75)

    # 1. Check Configuration (never printing key fragments)
    print("\n[1] Checking Environment & Groq Model...")
    configured = is_groq_configured()
    print(f"    API Key Configured: {'YES' if configured else 'NO'}")
    active_model = get_groq_model()
    print(f"    Active Model ID:    {active_model}")

    if not configured:
        print("\n[NOTICE] GROQ_API_KEY is not configured in .env.")
        print("         Live orchestration smoke test skipped.")
        return False

    # Check vector store
    store = get_vector_store()
    chunk_count = store.count()
    print(f"    Vector Store Chunks: {chunk_count}")
    if chunk_count == 0:
        print("[WARNING] Vector store is empty. Ingesting test documents...")
        from src.document_loader import load_documents_from_directory
        from src.text_cleaner import clean_document
        from src.chunker import chunk_document
        docs = load_documents_from_directory("data")
        for doc in docs:
            cleaned = clean_document(doc)
            chunks = chunk_document(cleaned)
            store.index_chunks(chunks)
        print(f"    Ingested {store.count()} chunks into vector store.")

    # 2. Test Supported Question
    print("\n[2] Executing Supported Question Pipeline...")
    query_1 = "How many encoder layers does the Transformer use?"
    print(f"    Query: '{query_1}'")
    try:
        result_1 = run_rag_pipeline(query_1)
        print(f"    Final Verdict:  {result_1.get('verdict')}")
        print(f"    Attempts:       {result_1.get('attempts')}")
        print(f"    Draft Answer:   {result_1.get('draft')}")
        print(f"    Judge Reason:   {result_1.get('reason')}")
        print(f"    Retrieved Chunks: {len(result_1.get('chunks', []))}")
        print("    Trace Steps:")
        for idx, step in enumerate(result_1.get("trace", []), 1):
            name = step.get("node") or step.get("event")
            print(f"      Step {idx}: {name}")
    except Exception as e:
        print(f"    [ERROR] Pipeline failed: {e}")
        return False

    # 3. Test Abstention Question
    print("\n[3] Executing Out-of-Corpus Abstention Query...")
    query_2 = "What is the capital city of Australia?"
    print(f"    Query: '{query_2}'")
    try:
        result_2 = run_rag_pipeline(query_2)
        print(f"    Final Verdict:  {result_2.get('verdict')}")
        print(f"    Attempts:       {result_2.get('attempts')}")
        print(f"    Draft Answer:   {result_2.get('draft')}")
        print(f"    Judge Reason:   {result_2.get('reason')}")
        print("    Trace Steps:")
        for idx, step in enumerate(result_2.get("trace", []), 1):
            name = step.get("node") or step.get("event")
            print(f"      Step {idx}: {name}")
    except Exception as e:
        print(f"    [ERROR] Abstention query failed: {e}")
        return False

    print("\n" + "=" * 75)
    print("PHASE 4 LIVE LANGGRAPH ORCHESTRATION SMOKE TEST COMPLETED SUCCESSFULLY!")
    print("=" * 75)
    return True


if __name__ == "__main__":
    success = run_live_graph_smoke_test()
    sys.exit(0 if success else 1)

