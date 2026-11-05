"""Test live Groq execution on key queries: SOP, KYC, multi-part, etc."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import time
from src.graph import run_rag_pipeline
from src.config import get_groq_model, is_groq_configured

print(f"Groq Configured: {is_groq_configured()}")
print(f"Configured Model: {get_groq_model()}")

queries = [
    "What is SOP?",
    "What is KYC?",
    "Which questions are present in this file?",
    "List all skills mentioned in this file.",
    "Summarize this document.",
    "What is the capital of Australia?",
    "What is KYC and what is the population of Mars?",
]

for q in queries:
    print("\n" + "=" * 70)
    print(f"QUERY: {q}")
    print("=" * 70)
    t0 = time.perf_counter()
    try:
        res = run_rag_pipeline(q)
        latency = (time.perf_counter() - t0) * 1000.0
        print(f"LATENCY: {latency:.1f} ms")
        print(f"VERDICT: {res.get('verdict')}")
        print(f"REASON:  {res.get('reason')}")
        print(f"DRAFT:\n{res.get('draft')}")
        chunks = res.get("chunks", [])
        print(f"RETRIEVED CHUNKS ({len(chunks)}):")
        for c in chunks[:3]:
            print(f"  - [{c.get('chunk_id')}]: {c.get('text', '')[:90]}...")
    except Exception as e:
        print(f"ERROR: {type(e).__name__} - {e}")
