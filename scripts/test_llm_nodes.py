"""Phase 3 Live Groq API Smoke Test Script.

Validates end-to-end live LLM execution on Groq:
1. Verifies API key configuration (without printing any key fragments).
2. Tests Generate Node on supplied context.
3. Tests Judge Node on both a supported answer and an unsupported answer.
4. Tests Rewrite Node on an unclear query.

Requires a valid GROQ_API_KEY in .env to perform live inference.
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

from src.config import get_groq_model, is_groq_configured, validate_groq_config
from src.nodes import generate_node, judge_node, rewrite_node
from src.state import GraphState, create_initial_state


def run_live_smoke_test() -> bool:
    print("=" * 75)
    print("ALETHIA PHASE 3: LIVE GROQ LLM NODES SMOKE TEST")
    print("=" * 75)

    # 1. Verify Configuration (strictly no key fragments printed)
    print("\n[1] Checking Groq Configuration...")
    configured = is_groq_configured()
    print(f"    API Key Configured: {'YES' if configured else 'NO'}")
    active_model = get_groq_model()
    print(f"    Active Model ID:    {active_model}")

    if not configured:
        print("\n[NOTICE] GROQ_API_KEY is not configured in .env.")
        print("         To execute live Groq API inference, add your key to .env:")
        print("         GROQ_API_KEY=your_key")
        print("\nLive smoke test skipped (API key required for live inference).")
        return False

    all_passed = True

    # 2. Test Generate Node
    print(f"\n[2] Testing Generate Node (Model: {active_model}, Temp: 0.1)...")
    sample_chunks = [
        {
            "chunk_id": "smoke_chunk_001",
            "source": "smoke_test.txt",
            "text": "The Transformer features six encoder layers and six decoder layers. It is optimized with Adam.",
        }
    ]
    gen_state = create_initial_state("How many encoder layers does the Transformer use?")
    gen_state["chunks"] = sample_chunks

    try:
        gen_result = generate_node(gen_state)
        print(f"    Question:     '{gen_result['question']}'")
        print(f"    Draft Output: '{gen_result['draft']}'")
        if gen_result["draft"] and "six" in gen_result["draft"].lower():
            print("    [PASS] Generate Node produced context-grounded answer.")
        else:
            print("    [FAIL] Generate Node output unexpected.")
            all_passed = False
    except Exception as e:
        print(f"    [FAIL] Generate Node raised exception: {e}")
        all_passed = False

    # 3A. Test Judge Node - Supported Answer
    print(f"\n[3A] Testing Judge Node on Supported Answer (Model: {active_model}, Temp: 0.0)...")
    judge_state_pass: GraphState = {
        "question": "How many encoder layers does the Transformer use?",
        "chunks": sample_chunks,
        "draft": "The Transformer uses six encoder layers.",
        "trace": [],
    }

    try:
        judge_res_pass = judge_node(judge_state_pass)
        print(f"    Draft:   '{judge_state_pass['draft']}'")
        print(f"    Verdict: {judge_res_pass['verdict']}")
        print(f"    Reason:  {judge_res_pass['reason']}")
        if judge_res_pass["verdict"] == "PASS":
            print("    [PASS] Judge correctly approved supported answer.")
        else:
            print("    [FAIL] Judge rejected supported answer.")
            all_passed = False
    except Exception as e:
        print(f"    [FAIL] Judge Node raised exception: {e}")
        all_passed = False

    # 3B. Test Judge Node - Unsupported Answer
    print(f"\n[3B] Testing Judge Node on Unsupported / Hallucinated Answer (Model: {active_model}, Temp: 0.0)...")
    judge_state_fail: GraphState = {
        "question": "How many encoder layers does the Transformer use?",
        "chunks": sample_chunks,
        "draft": "The model uses twenty-four encoder layers and was trained on Mars.",
        "trace": [],
    }

    try:
        judge_res_fail = judge_node(judge_state_fail)
        print(f"    Draft:   '{judge_state_fail['draft']}'")
        print(f"    Verdict: {judge_res_fail['verdict']}")
        print(f"    Reason:  {judge_res_fail['reason']}")
        if judge_res_fail["verdict"] == "FAIL":
            print("    [PASS] Judge correctly caught unsupported claim.")
        else:
            print("    [FAIL] Judge incorrectly approved unsupported claim.")
            all_passed = False
    except Exception as e:
        print(f"    [FAIL] Judge Node raised exception: {e}")
        all_passed = False

    # 4. Test Rewrite Node
    print(f"\n[4] Testing Rewrite Node (Model: {active_model}, Temp: 0.2)...")
    rewrite_state: GraphState = {
        "question": "layers?",
        "original_question": "layers?",
        "reason": "The query was too brief to retrieve specific layer architecture details.",
        "attempts": 0,
        "trace": [],
    }

    try:
        rewrite_res = rewrite_node(rewrite_state)
        print(f"    Original Question:  '{rewrite_res['original_question']}'")
        print(f"    Rewritten Question: '{rewrite_res['question']}'")
        print(f"    Attempts Count:     {rewrite_res['attempts']}")
        if rewrite_res["question"] and rewrite_res["question"] != "layers?":
            print("    [PASS] Rewrite Node formulated a more specific query.")
        else:
            print("    [FAIL] Rewrite Node did not reformulate query.")
            all_passed = False
    except Exception as e:
        print(f"    [FAIL] Rewrite Node raised exception: {e}")
        all_passed = False

    print("\n" + "=" * 75)
    if all_passed:
        print("PHASE 3 LIVE GROQ SMOKE TEST: ALL NODES FUNCTIONING [PASS]")
    else:
        print("PHASE 3 LIVE GROQ SMOKE TEST: ISSUES DETECTED [FAIL]")
    print("=" * 75)

    return all_passed


if __name__ == "__main__":
    success = run_live_smoke_test()
    sys.exit(0 if success else 1)
