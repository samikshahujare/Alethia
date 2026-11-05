"""Live End-to-End Verification of Alethia with Real Groq API Calls.

Executes live queries against the active ChromaDB store and records:
- Configured model (openai/gpt-oss-20b)
- Live end-to-end latency per query
- Retrieved chunk IDs and sources
- Generated answer draft
- Judge verdict and reason
- Factual verification against document contents
"""

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunker import TextChunk
from src.config import get_groq_model, is_groq_configured
from src.graph import run_rag_pipeline
from src.hybrid_retriever import get_hybrid_retriever
from src.store import get_vector_store


def run_live_verification():
    print("=" * 80)
    print("ALETHIA LIVE END-TO-END ACCURACY AND API VERIFICATION")
    print("=" * 80)
    print(f"Groq Configured:   {is_groq_configured()}")
    model_name = get_groq_model()
    print(f"Configured Model:  {model_name}")

    if not is_groq_configured():
        print("ERROR: Groq is not configured.")
        return

    test_queries = [
        {
            "id": "LIVE-01",
            "name": "Question Extraction",
            "query": "Which questions are present in this file?",
            "check": lambda d, v: "21 total" in d and v == "PASS",
        },
        {
            "id": "LIVE-02",
            "name": "SOP Query",
            "query": "What is SOP?",
            "check": lambda d, v: ("created sops" in d.lower() or "placement dashboard" in d.lower() or "not in context" in d.lower()) and v == "PASS",
        },
        {
            "id": "LIVE-03",
            "name": "KYC Query",
            "query": "What is KYC?",
            "check": lambda d, v: "know your customer" in d.lower() and v == "PASS",
        },
        {
            "id": "LIVE-04",
            "name": "Skills Extraction",
            "query": "List all skills mentioned in this file.",
            "check": lambda d, v: len(d) > 30 and v == "PASS",
        },
        {
            "id": "LIVE-05",
            "name": "Document Summary",
            "query": "Summarize this document.",
            "check": lambda d, v: len(d) > 50 and v == "PASS",
        },
        {
            "id": "LIVE-06",
            "name": "Out-of-Scope Query",
            "query": "What is the capital of Australia?",
            "check": lambda d, v: "not in context" in d.lower() and v == "PASS",
        },
        {
            "id": "LIVE-07",
            "name": "Multi-Part Query (Supported + Missing)",
            "query": "What is KYC and what is the population of Mars?",
            "check": lambda d, v: "know your customer" in d.lower() and "not in context" in d.lower() and v == "PASS",
        },
    ]

    results = []

    for item in test_queries:
        qid = item["id"]
        qname = item["name"]
        q = item["query"]
        print(f"\n[{qid}] Executing {qname}: '{q}'")

        t0 = time.perf_counter()
        try:
            res = run_rag_pipeline(q)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            draft = res.get("draft", "")
            verdict = res.get("verdict", "")
            reason = res.get("reason", "")
            chunks = res.get("chunks", [])
            chunk_ids = [c.get("chunk_id") for c in chunks]
            sources = sorted(list({c.get("source") for c in chunks if c.get("source")}))

            passed = item["check"](draft, verdict)

            print(f"  Latency:    {latency_ms:.1f} ms")
            print(f"  Verdict:    {verdict}")
            print(f"  Reason:     {reason}")
            print(f"  Draft:      {draft[:140]}...")
            print(f"  Sources:    {sources}")
            print(f"  Chunks:     {chunk_ids[:3]}")
            print(f"  Status:     {'PASS' if passed else 'FAIL'}")

            results.append({
                "id": qid,
                "name": qname,
                "query": q,
                "latency_ms": round(latency_ms, 1),
                "verdict": verdict,
                "reason": reason,
                "draft": draft,
                "sources": sources,
                "chunk_ids": chunk_ids,
                "passed": passed,
            })
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {str(e)}")
            results.append({
                "id": qid,
                "name": qname,
                "query": q,
                "error": str(e),
                "passed": False,
            })

    # Test 8: Document Lifecycle (Upload, Index, Remove, Re-query)
    print(f"\n[LIVE-08] Testing Document Upload, Index, Remove & Re-query Lifecycle...")
    store = get_vector_store()
    retriever = get_hybrid_retriever()
    temp_doc = "temp_compliance_audit_v1.txt"
    temp_chunk = TextChunk(
        chunk_id=f"{temp_doc.replace('.', '_')}_chunk_0000",
        text="Audit standard 704 mandates dual authorization for fund transfers exceeding $50,000.",
        source=temp_doc,
        chunk_index=0,
        metadata={"source": temp_doc, "char_count": 89},
    )

    t0 = time.perf_counter()
    # 1. Index
    store.index_chunks([temp_chunk], overwrite=True)
    retriever.sync_bm25()

    # 2. Retrieve
    q_temp = "What does Audit standard 704 mandate?"
    before_chunks = retriever.retrieve_hybrid(q_temp, k=2)
    found_before = any(c.get("chunk_id") == temp_chunk.chunk_id for c in before_chunks)

    # 3. Delete
    deleted_count = store.delete_document(temp_doc)
    retriever.sync_bm25()

    # 4. Re-query
    after_chunks = retriever.retrieve_hybrid(q_temp, k=2)
    found_after = any(c.get("chunk_id") == temp_chunk.chunk_id for c in after_chunks)
    lifecycle_latency_ms = (time.perf_counter() - t0) * 1000.0

    passed_lifecycle = found_before and (deleted_count >= 1) and (not found_after)
    print(f"  Indexed & Retrieved before deletion: {found_before}")
    print(f"  Deleted count:                       {deleted_count}")
    print(f"  Retrieved after deletion:            {found_after}")
    print(f"  Lifecycle Latency:                   {lifecycle_latency_ms:.1f} ms")
    print(f"  Status:                              {'PASS' if passed_lifecycle else 'FAIL'}")

    results.append({
        "id": "LIVE-08",
        "name": "Document Lifecycle",
        "query": q_temp,
        "latency_ms": round(lifecycle_latency_ms, 1),
        "passed": passed_lifecycle,
    })

    # Summary
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.get("passed"))
    avg_latency = round(sum(r.get("latency_ms", 0.0) for r in results) / total_tests, 1)

    print("\n" + "=" * 80)
    print(f"LIVE VERIFICATION SUMMARY: {passed_tests}/{total_tests} Tests Passed.")
    print(f"Average Live Pipeline Latency: {avg_latency} ms")
    print("=" * 80)

    # Save to JSON
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "live_verification_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "average_latency_ms": avg_latency,
            "results": results,
        }, f, indent=2)
    print(f"Saved results to {out_file}")


if __name__ == "__main__":
    run_live_verification()

