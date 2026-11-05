import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.graph import run_rag_pipeline

q = "Why does standard Adam implement weight decay incorrectly when paired with L2 regularization, and what optimizer was used to train the base Transformer?"
state = run_rag_pipeline(q)

print("Attempts:", state.get("attempts"))
print("Final Verdict:", state.get("verdict"))
draft = str(state.get("draft")).encode('ascii', 'backslashreplace').decode('ascii')
print("Final Draft:", draft)

trace = state.get("trace", [])
for i, t in enumerate(trace):
    node = t.get("node", t.get("event"))
    print(f"[{i}] {node}")
    if node == "retrieve_node":
        print(f"    Chunks: {t.get('chunk_ids')}")
    elif node == "generate_node":
        d = str(t.get("draft")).encode('ascii', 'backslashreplace').decode('ascii')
        print(f"    Draft: {d}")
    elif node == "judge_node":
        r = str(t.get("reason")).encode('ascii', 'backslashreplace').decode('ascii')
        print(f"    Verdict: {t.get('verdict')} | Reason: {r}")
    elif node == "rewrite_node":
        rew = str(t.get("rewritten_question")).encode('ascii', 'backslashreplace').decode('ascii')
        print(f"    Rewritten: {rew}")

