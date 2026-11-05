"""Live verification script for hardened Judge logic against Groq API.

Tests all five required scenarios live against the configured Groq model:
1. Grounded complete answer -> PASS
2. Correct abstention on partial retrieval -> PASS
3. Unsupported hallucinated claim on missing part -> FAIL
4. Unnecessary abstention when all info present -> FAIL
5. Contradiction of context -> FAIL

Runs separately from automated offline unit tests.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import get_groq_model, validate_groq_config
from src.nodes import judge_node
from src.state import GraphState


def run_live_judge_scenarios():
    print("=" * 80)
    print("ALETHIA STEP 4.1: LIVE HARDENED JUDGE VERIFICATION")
    print("=" * 80)

    cfg = validate_groq_config()
    model = get_groq_model()
    print(f"Model: {model}")
    print(f"Groq Config Valid: {cfg['is_valid']}")
    if not cfg["is_valid"]:
        raise RuntimeError("Groq API key not configured.")

    context_part_a = "Training was conducted using the Adam optimizer with warmup."
    context_part_b = "Standard Adam uses beta_1 equals 0.9 and beta_2 equals 0.999."
    full_context = [{"text": context_part_a}, {"text": context_part_b}]
    partial_context = [{"text": context_part_b}]

    test_cases = [
        {
            "scenario": "Scenario 1: Fully Grounded Multi-Part",
            "question": "What optimizer was used in Attention Is All You Need, and what are standard Adam's beta values?",
            "chunks": full_context,
            "draft": "The Transformer was trained using the Adam optimizer with warmup. Default parameters for Adam are beta_1 = 0.9 and beta_2 = 0.999.",
            "expected_verdict": "PASS",
        },
        {
            "scenario": "Scenario 2: Correct Abstention on Partial Retrieval",
            "question": "What optimizer was used in Attention Is All You Need, and what are standard Adam's beta values?",
            "chunks": partial_context,
            "draft": "Default parameters for standard Adam are beta_1 = 0.9 and beta_2 = 0.999. The optimizer used in Attention Is All You Need is not in context.",
            "expected_verdict": "PASS",
        },
        {
            "scenario": "Scenario 3: Unsupported Claim on Missing Part",
            "question": "What optimizer was used in Attention Is All You Need, and what are standard Adam's beta values?",
            "chunks": partial_context,
            "draft": "Default parameters for standard Adam are beta_1 = 0.9 and beta_2 = 0.999. The Transformer used SGD with Nesterov momentum.",
            "expected_verdict": "FAIL",
        },
        {
            "scenario": "Scenario 4: Unnecessary Abstention When All Info Present",
            "question": "What optimizer was used in Attention Is All You Need, and what are standard Adam's beta values?",
            "chunks": full_context,
            "draft": "Not in context.",
            "expected_verdict": "FAIL",
        },
        {
            "scenario": "Scenario 5: Contradiction of Context",
            "question": "What optimizer was used in Attention Is All You Need, and what are standard Adam's beta values?",
            "chunks": full_context,
            "draft": "The Transformer used RMSprop, and Adam beta_1 is 0.5.",
            "expected_verdict": "FAIL",
        },
    ]

    all_passed = True
    results = []

    for idx, tc in enumerate(test_cases, 1):
        print(f"\n[{idx}/5] {tc['scenario']}")
        print(f"  Question: {tc['question']}")
        print(f"  Chunks Count: {len(tc['chunks'])}")
        print(f"  Draft: {tc['draft']}")

        state: GraphState = {
            "question": tc["question"],
            "chunks": tc["chunks"],
            "draft": tc["draft"],
            "trace": [],
        }

        updated = judge_node(state)
        actual_verdict = updated.get("verdict", "")
        reason = updated.get("reason", "")
        expected = tc["expected_verdict"]

        passed = (actual_verdict == expected)
        all_passed = all_passed and passed
        print(f"  Actual Verdict: {actual_verdict} (Expected: {expected}) -> {'PASS' if passed else 'FAIL'}")
        print(f"  Reason: {reason}")

        results.append({
            "scenario": tc["scenario"],
            "expected": expected,
            "actual": actual_verdict,
            "reason": reason,
            "matches": passed,
        })

    print("\n" + "=" * 80)
    print(f"LIVE HARDENED JUDGE RESULTS: {'ALL 5 SCENARIOS PASSED' if all_passed else 'SOME SCENARIOS FAILED'}")
    print("=" * 80)
    return results


if __name__ == "__main__":
    run_live_judge_scenarios()

