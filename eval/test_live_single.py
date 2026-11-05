import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import time
from src.graph import run_rag_pipeline

t0 = time.perf_counter()
res = run_rag_pipeline("What is KYC and what is the population of Mars?")
total_t = (time.perf_counter() - t0) * 1000.0

print(f"Total Pipeline Latency: {total_t:.1f} ms")
print(f"Verdict: {res.get('verdict')}")
print(f"Reason:  {res.get('reason')}")
print(f"Draft:\n{res.get('draft')}")
print("\nExecution Trace:")
for step in res.get("trace", []):
    print(" ", step.get("node") or step.get("event"), "->", {k: v for k, v in step.items() if k not in ("node", "event", "draft", "raw_response")})
