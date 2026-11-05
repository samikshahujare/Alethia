import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.graph import run_rag_pipeline

q = "Why does standard Adam implement weight decay incorrectly, and what optimizer was used to train the base Transformer?"
res = run_rag_pipeline(q)

print("Final attempts:", res.get("attempts"))
print("Final verdict:", res.get("verdict"))
print("Final draft:", res.get("draft"))

trace = res.get("trace", [])
for i, t in enumerate(trace):
    print(f"[{i}] Node: {t.get('node', t.get('event'))}")
    if t.get("node") == "retrieve_node":
        print(f"    Chunks: {t.get('chunk_ids')}")
    elif t.get("node") == "generate_node":
        print(f"    Draft: {t.get('draft')[:100]}...")
    elif t.get("node") == "judge_node":
        print(f"    Verdict: {t.get('verdict')} | Reason: {t.get('reason')}")
    elif t.get("node") == "rewrite_node":
        print(f"    Rewritten: {t.get('rewritten_question')}")

